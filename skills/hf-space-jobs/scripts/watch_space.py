"""Wait on a private Space's log page, then print the log's tail and the Space's state.

    watch_space.py OWNER/NAME [--pattern REGEX] [--minutes N]

Stops when a log line matches the pattern, when the runner finishes ("pausing
the Space" or "runner error"), or when the page has served no log for ten
minutes. Only lines starting with "[" are kept, which drops progress bars.
Run it in the background for long waits. The token is read from HF_TOKEN.

Needs huggingface_hub, for example: uv run --with huggingface_hub python watch_space.py ...
"""

from __future__ import annotations

import argparse
import os
import re
import time
import urllib.request


def fetch(url: str, token: str) -> list[str]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:
        return []
    return [line for line in text.replace("\r", "\n").split("\n") if line.startswith("[")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("space", help="OWNER/NAME of the Space")
    parser.add_argument("--pattern", default=None, help="also stop when a log line matches this regular expression")
    parser.add_argument("--minutes", type=int, default=720, help="longest time to wait")
    parser.add_argument("--tail", type=int, default=40, help="log lines to print at the end")
    args = parser.parse_args()

    token = os.environ["HF_TOKEN"]
    url = f"https://{re.sub(r'[^a-z0-9]+', '-', args.space.lower())}.hf.space/"
    finished = re.compile("pausing the Space|runner error" + (f"|job exited|{args.pattern}" if args.pattern else ""))
    deadline, empty, lines = time.time() + 60 * args.minutes, 0, []
    while time.time() < deadline:
        lines = fetch(url, token)
        if any(finished.search(line) for line in lines):
            break
        empty = 0 if lines else empty + 1
        if empty >= 20:  # a paused or failed Space serves no log
            break
        time.sleep(30)
    print("\n".join(line[:250] for line in lines[-args.tail :]))

    from huggingface_hub import HfApi

    runtime = HfApi(token=token).get_space_runtime(args.space)
    print(f"Space stage: {runtime.stage} | hardware: {runtime.hardware} | sleep cap (s): {runtime.sleep_time}")


if __name__ == "__main__":
    main()
