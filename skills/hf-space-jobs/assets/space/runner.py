"""Run one job, upload its log and results, then pause the Space.

A small web server shows the log, which also satisfies the Space health check.
"""

from __future__ import annotations

import http.server
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi

JOB = os.environ.get("JOB", "example")
JOB_ID = os.environ.get("JOB_ID", "unnamed")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "")
SPACE_ID = os.environ.get("SPACE_ID", "")
OUT = Path("/home/user/out")
LOG = OUT / "log.txt"


class LogPage(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = LOG.read_bytes()[-200_000:] if LOG.exists() else b"starting"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def say(text: str) -> None:
    with LOG.open("a") as handle:
        handle.write(f"[runner] {text}\n")
    print(f"[runner] {text}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    threading.Thread(
        target=http.server.ThreadingHTTPServer(("0.0.0.0", 7860), LogPage).serve_forever, daemon=True
    ).start()
    api = HfApi()
    prefix = f"runs/{JOB_ID}"
    try:
        if RESULTS_REPO and api.file_exists(RESULTS_REPO, f"{prefix}/log.txt"):
            say(f"{JOB_ID} already has results in {RESULTS_REPO}; not running it again.")
        else:
            say(f"job={JOB} id={JOB_ID}")
            with LOG.open("a") as handle:
                code = subprocess.call(
                    [sys.executable, f"jobs/{JOB}.py", str(OUT)], stdout=handle, stderr=subprocess.STDOUT
                )
            say(f"job exited with code {code}")
            if code != 0:  # keep the JOB_ID free so the job can be run again
                prefix = f"runs/{JOB_ID}-failed-{time.strftime('%Y%m%dT%H%M%S')}"
            if RESULTS_REPO:
                say(f"uploading to {RESULTS_REPO}/{prefix}")
                api.upload_folder(repo_id=RESULTS_REPO, folder_path=str(OUT), path_in_repo=prefix)
    except Exception as exc:  # always reach the pause below
        say(f"runner error: {type(exc).__name__}: {exc}")
    try:
        if SPACE_ID:
            say("pausing the Space")
            api.pause_space(SPACE_ID)
    except Exception as exc:
        say(f"could not pause: {type(exc).__name__}: {exc}; the sleep timer is the backstop")
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
