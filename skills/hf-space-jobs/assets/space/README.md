---
title: GPU batch job
emoji: 🧪
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Runs one batch job on the Space's GPU, uploads its log and results to a private
repository, then pauses the Space so billing stops. The page shows the live log.

Settings (Space variables): `JOB` names a script in `jobs/`, `JOB_ID` names the
run, `RESULTS_REPO` is where results go. `HF_TOKEN` is a Space secret with write
access. A `JOB_ID` that already has results is not run again, so waking the Space
does not repeat a finished job.
