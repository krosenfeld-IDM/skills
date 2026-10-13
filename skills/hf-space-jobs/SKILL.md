---
name: hf-space-jobs
description: "Run GPU batch jobs (fine-tuning, feature extraction, large batched inference, throughput tests) on a private Hugging Face Space that uploads its results and pauses itself. Use whenever work needs a rented GPU via Hugging Face Spaces: setting one up, launching a run, checking whether a job is done, resuming, downloading results, or changing the sleep timer or hardware."
---

# GPU batch jobs on a Hugging Face Space

A Space with paid hardware can serve as a rented GPU for one-off jobs. The
pattern here: a private Docker Space runs **one job**, uploads its log and
outputs to a private results repository, then **pauses itself** so billing
stops. A small web page shows the live log.

Bundled files (paths relative to this skill):

| File | Purpose |
| --- | --- |
| `scripts/deploy_space.py` | `init` a Space directory from the template; `run` stages it and starts a job |
| `scripts/watch_space.py` | wait on the Space's log, then print its tail and the Space's state |
| `assets/space/` | template: `Dockerfile`, `runner.py`, `requirements.txt`, Space card, `jobs/example.py` |

Both scripts need `huggingface_hub` (`uv run --with huggingface_hub python ...`).

## The token

Everything authenticates with `HF_TOKEN`. It needs write access to the account
or organization that owns the Space. The deploy script also stores it as a
Space secret so the job can upload and pause itself.

Never print the token. If the user updated it after the session started, the
inherited environment is stale; re-read it from their shell profile into the
command's environment, for example
`export HF_TOKEN=$(zsh -ic 'print -r -- $HF_TOKEN' 2>/dev/null | tail -1)`.

## Starting a job costs money

`run --start` requests paid hardware. Launch only when the user has asked for
that run; approval for one run does not cover the next, larger one. Without
`--start` the script prints its plan and creates nothing, which is a good way
to show the user what will happen.

Tell the user the expected duration as you launch, estimated from measured
speed rather than guesswork. If nothing has been measured on this hardware
yet, run a short throughput job first: estimates made before measuring are
easily several times too optimistic.

## Setting up

```
python scripts/deploy_space.py init path/to/project/space
```

This copies the template into the project. Then:

- Pin every version in `requirements.txt`, matching the environment the
  results must transfer to.
- Write jobs under `jobs/` (see "Writing a job").
- Name the Space after what it runs (the model, for instance) so several can
  coexist. The results repository defaults to `<space>-runs`.

## Launching

```
python scripts/deploy_space.py run --dir path/to/project/space \
    --space OWNER/NAME --job train --job-id train-001 \
    --data data/train.parquet --add src/mymodel.py \
    --variable TRAIN_EXAMPLES=50000 \
    --sleep-seconds 43200 --start
```

- `--job` is a script name in `jobs/`. `--job-id` names this run and must be
  new: the runner refuses to rerun an ID that already has
  `runs/<id>/log.txt`, which protects finished results when a Space restarts.
  A failed job uploads to `runs/<id>-failed-<time>` and leaves the ID free.
- `--data FILE` copies a file into the Space's `data/` directory; `--add
  SRC[:DEST]` copies project code or other files anywhere in the Space. Both
  repeat. Staged files are uploaded to the (private) Space repository, so
  check the printed file list for anything that should not leave the machine.
- `--variable KEY=VALUE` passes settings to the job as environment variables.
  Variables left by an earlier run are deleted on each deploy, so a run sees
  only what its own command set.
- `--sleep-seconds` is the backstop if self-pausing fails. Set it well above
  the expected job length; the default hour is too short for training.
- `--hardware` takes a Spaces hardware flavor (default `a100-large`).

The script requests hardware before uploading files, so the job never starts
on the free CPU, and restarts the Space if an earlier job left it paused.

## Watching a run

Wait with the watcher rather than polling by hand. For long waits run it in
the background so the session is notified when it exits:

```
python scripts/watch_space.py OWNER/NAME --pattern "step 20/"   # first progress, or early failure
python scripts/watch_space.py OWNER/NAME --minutes 720           # until the runner finishes
```

Use both stages for anything long. The first catches failures in the opening
minutes (build errors, out of memory, a broken data path) while they are
cheap; the second waits for the end. Building the image and loading a model
can take ten minutes or more before the first job output, so an empty log
early on is normal.

When a job ends, confirm two things before reporting: the Space stage is
`PAUSED` (billing has stopped), and the expected files are in the results
repository (`HfApi().list_repo_files(results_repo)`). The watcher prints the
stage.

The job does not depend on the local session. If a watcher is lost, start
another one.

## Getting results

```
python -c "
from huggingface_hub import snapshot_download
snapshot_download('OWNER/NAME-runs', allow_patterns=['runs/<job-id>/*'], local_dir='data/runs')"
```

Download into a directory Git ignores. Small outputs are under
`runs/<job-id>/`; anything a job uploaded itself is under the prefix it chose.
Record the run's settings, results and reproduce command wherever the project
keeps such notes.

## Changing a running Space

- **Sleep timer:** `HfApi().set_space_sleep_time(space, seconds)`. In one
  observed case the stage briefly read `RUNNING_BUILDING` and the job carried
  on uninterrupted. Check the log afterwards rather than assuming either way.
- **Hardware:** changing it restarts the container, and the job with it.
- **Stopping early:** `HfApi().pause_space(space)`. Ask first unless the job
  is clearly broken; only a resumable job keeps its work.

## Writing a job

A job is `jobs/<name>.py`, run as `python jobs/<name>.py OUT`. Everything left
in `OUT` is uploaded to `runs/<JOB_ID>/` when it exits. It reads `JOB_ID` and
`RESULTS_REPO` from the environment. `assets/space/jobs/example.py` shows the
conventions in about sixty lines; start from it.

What has mattered in practice:

- **Exit before any large download when there is no GPU.** A Space that boots
  on CPU by mistake should fail in seconds.
- **Give the job a dry-run switch** that shrinks everything to a few small
  inputs and skips uploads, and run it locally before paying for a GPU. Most
  bugs are in data handling and bookkeeping and show up there for free. Local
  numerics may differ (half precision on a laptop GPU can produce NaN where
  the real hardware does not); the dry run checks the code path, not the
  numbers.
- **Make it resumable if it runs longer than about half an hour.** Upload
  intermediate results (shards, training state) to the results repository as
  they finish, and skip or reload them on start. Delete each uploaded file
  from `OUT` so the runner does not upload it again. Every upload stays in
  the repository's history, so save large training state every half hour
  rather than every few minutes.
- **Fail early and loudly.** Check on the first training step that the
  trainable parameters actually receive a gradient, and stop on a NaN loss. A
  silent failure otherwise costs the whole run.
- **Batch by token count, not example count,** when input lengths vary
  widely: group inputs of similar length under a token budget. Fixed-size
  batches run out of memory on the long tail. Memory is not reliably released
  after an out-of-memory error, so when probing limits run each configuration
  in its own process.
- **Match the target environment where results must transfer.** If an
  optional accelerated code path exists only on the Space, use it for
  training if you like, but produce evaluation predictions with the path the
  target environment will use.
- **Print progress as timestamped lines starting with `[`.** The watcher
  keeps only those lines, which filters out download and upload progress bars.
