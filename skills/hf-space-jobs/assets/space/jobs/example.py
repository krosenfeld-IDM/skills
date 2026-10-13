"""Template job: shows the conventions the runner and the watcher rely on.

Run by the runner as ``python jobs/example.py OUT``. Everything left in OUT is
uploaded to ``runs/<JOB_ID>/`` in the results repository when the job exits.
Larger intermediate results are uploaded here as they finish, so a restarted
job resumes instead of starting over.

Local check of the code path, with no GPU and no upload:
    EXAMPLE_DRY_RUN=1 python jobs/example.py out
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path(__file__).resolve().parent.parent  # the Space root; data files are in ROOT / "data"
JOB_ID = os.environ.get("JOB_ID", "unnamed")
RESULTS_REPO = os.environ.get("RESULTS_REPO", "")
DRY_RUN = os.environ.get("EXAMPLE_DRY_RUN") == "1"
PARTS = int(os.environ.get("EXAMPLE_PARTS", 2 if DRY_RUN else 10))  # set with --variable EXAMPLE_PARTS=...
UPLOAD = bool(RESULTS_REPO) and not DRY_RUN


def say(text: str) -> None:
    # Timestamped lines starting with "[" are what the watcher keeps; progress bars are dropped.
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# Stop before any large download if the Space came up without a GPU.
if not torch.cuda.is_available() and not DRY_RUN:
    sys.exit("No GPU: stopping before the download.")

api = HfApi()
prefix = f"parts/{JOB_ID}"
done = {name for name in api.list_repo_files(RESULTS_REPO) if name.startswith(prefix + "/")} if UPLOAD else set()
summary = {"job_id": JOB_ID, "parts": []}
start = time.time()
for number in range(PARTS):
    name = f"{prefix}/part_{number:04d}.json"
    if name in done:
        say(f"{name} exists; skipped")
        continue
    result = {"part": number, "value": float(torch.randn(1000).mean())}  # the real work goes here
    if result["value"] != result["value"]:  # fail early and loudly rather than waste the run
        raise RuntimeError(f"NaN in part {number}")
    path = OUT / f"part_{number:04d}.json"
    path.write_text(json.dumps(result))
    if UPLOAD:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=RESULTS_REPO)
        path.unlink()  # already uploaded; keep it out of the runner's final upload
    summary["parts"].append(name)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    say(f"part {number + 1}/{PARTS} done, {time.time() - start:.0f}s elapsed")
say("finished")
