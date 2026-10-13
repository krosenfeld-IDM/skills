"""Create or update a job-running Hugging Face Space and start one job on paid hardware.

    deploy_space.py init DIR
        Copy the Space template (Dockerfile, runner, example job) into DIR.

    deploy_space.py run --dir DIR --space OWNER/NAME --job NAME --job-id ID [options] [--start]
        Stage DIR plus any --add and --data files. Without --start, print the plan
        and create nothing. With --start, create the private Space and results
        repository, upload, and request the GPU, which starts billing.

The token comes from HF_TOKEN and needs write access to both repositories. It is
also stored as a Space secret so the job can upload results and pause itself.

Needs huggingface_hub, for example: uv run --with huggingface_hub python deploy_space.py ...
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "space"
RESERVED = ("JOB", "JOB_ID", "RESULTS_REPO")


def init(args: argparse.Namespace) -> None:
    target = Path(args.dir)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} is not empty; not overwriting it.")
    shutil.copytree(TEMPLATE, target, dirs_exist_ok=True)
    print(f"Copied the Space template to {target}. Edit requirements.txt and add jobs under {target / 'jobs'}.")


def run(args: argparse.Namespace) -> None:
    source = Path(args.dir)
    if not (source / "jobs" / f"{args.job}.py").exists():
        raise SystemExit(f"{source / 'jobs' / (args.job + '.py')} does not exist.")
    results = args.results or f"{args.space}-runs"

    stage = Path(tempfile.mkdtemp(prefix="space_stage_"))
    shutil.copytree(source, stage, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".git"))
    for entry in args.add:  # SRC or SRC:DEST, DEST relative to the Space root
        src, _, dest = entry.partition(":")
        target = stage / (dest or Path(src).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, target, dirs_exist_ok=True) if Path(src).is_dir() else shutil.copy(src, target)
    (stage / "data").mkdir(exist_ok=True)
    for name in args.data:
        shutil.copy(name, stage / "data" / Path(name).name)

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in files) / 2**20
    variables = {"JOB": args.job, "JOB_ID": args.job_id, "RESULTS_REPO": results}
    variables.update(pair.split("=", 1) for pair in args.variable)
    print(f"Space {args.space} (private, Docker, {args.hardware}, sleeps after {args.sleep_seconds}s idle)")
    print(f"Results repository {results} (private)")
    print(f"Variables {variables}")
    print(f"Files ({size:.1f} MiB): " + ", ".join(str(p.relative_to(stage)) for p in files))
    if not args.start:
        print("Dry run: nothing was created. Add --start to create the Space and start billed hardware.")
        return

    from huggingface_hub import HfApi

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(results, repo_type="model", private=True, exist_ok=True)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.add_space_secret(args.space, "HF_TOKEN", token)
    for key in api.get_space_variables(args.space):  # job settings left by an earlier run
        if key not in variables and key not in RESERVED:
            api.delete_space_variable(args.space, key)
    for key, value in variables.items():
        api.add_space_variable(args.space, key, value)
    # Hardware first, so the job never starts on the default CPU.
    api.request_space_hardware(args.space, args.hardware, sleep_time=args.sleep_seconds)
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(stage), commit_message=f"Run {args.job_id}")
    runtime = api.get_space_runtime(args.space)
    if runtime.stage == "PAUSED":  # a push does not wake a Space paused by an earlier job
        runtime = api.restart_space(args.space)
    print(f"Requested {args.hardware}; stage={runtime.stage}. Log: https://huggingface.co/spaces/{args.space}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("init", help="copy the Space template into a directory")
    start.add_argument("dir")
    start.set_defaults(function=init)
    job = commands.add_parser("run", help="stage a Space directory and (with --start) run one job")
    job.add_argument("--dir", required=True, help="Space directory created by init")
    job.add_argument("--space", required=True, help="OWNER/NAME of the Space")
    job.add_argument("--results", default=None, help="results repository; default <space>-runs")
    job.add_argument("--job", required=True, help="script name in jobs/, without .py")
    job.add_argument("--job-id", required=True, help="unique name for this run")
    job.add_argument("--hardware", default="a100-large", help="Space hardware flavor")
    job.add_argument("--sleep-seconds", type=int, default=3600, help="idle cap; set it above the job's length")
    job.add_argument("--add", action="append", default=[], metavar="SRC[:DEST]",
                     help="extra file or directory to place in the Space, such as project code the job imports")
    job.add_argument("--data", action="append", default=[], help="file to place in the Space's data/ directory")
    job.add_argument("--variable", action="append", default=[], metavar="KEY=VALUE", help="extra Space variable for the job")
    job.add_argument("--start", action="store_true", help="create, upload and start billed hardware")
    job.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
