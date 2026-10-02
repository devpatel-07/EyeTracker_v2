# Blink mini-project: development handoff

**September 30 status:** the project owner accepts current fixed-crop blink-filter performance for continued development. Independent-recording validation is deferred; it is not a prerequisite for implementing the next development feature. The optional pupil ROI tracker is implemented and documented in [ROI_DEVELOPMENT.md](ROI_DEVELOPMENT.md), including processing order, how to run it, and measured results. Fixed crops remain the default. Earlier status reports and requests for fresh data below describe prior milestones, not additional work required before trying tracking.

**September 28 update:** no new image rule was promoted after component ablations. New runs preserve raw pupil observations on skipped frames for diagnosis; 185 tests pass. See the final section of PROJECT_GUIDE.md. The September 24 run below remains the prior behavioral reference.

**September 24 update:** the earlier image/recovery gate and disabled-by-default robot-command gate are implemented. The final default run is `20260924T184324_1ba59c` (17/94 bad admissions, 11/176 good rejections; 180 tests). Stronger border/brightness rules regressed and remain experimental. See the latest section of PROJECT_GUIDE.md for policy selection, timing changes, and the live robot integration contract. Older results below are historical.

Status: working development prototype; independent performance validation remains open.

## What is complete

- Pupil detection, per-eye filtering and recovery, geometry protections and explicit model reliability diagnostics.
- Conservative crop-boundary and upper-boundary checks, plus a broad upper-pupil contamination check. All preserve the original pupil fit and model weights.
- Human label storage, baseline/filter execution, scoring, and a clickable mistake list with reviewer notes and source frame numbers.
- Automated regression tests (164 passing) and full-video comparisons. Recovery thresholds are unchanged.

## Reproducible development result

Run: `20260918T023223_8c0e2e`. Reviewer: ryan. Frozen label revision: 406.
282 starter labels; 270 certain labels scored, 12 uncertain labels excluded. These are selected development frames from one recording pair, not 270 independent experimental subjects. Adjacent frames are correlated.

| Version | Unusable admitted | Usable rejected | Mean measured recovery delay |
|---|---:|---:|---:|
| Original filter | 35/94 | 6/176 | 19.0 ms |
| First boundary update | 34/94 | 7/176 | 23.8 ms |
| Current baseline | 65/94 | 1/176 | 8.3 ms |
| Current filter | 17/94 | 11/176 | 33.3 ms |

Both new modes process 1,267 paired frames (2,534 eye records per mode), on CPU, minimum/recovery confidence 0.7/0.7, recovery hold 0.05 seconds. Mean recovery delay excludes wholly missed bad intervals and intervals with no observed acceptance; read the full JSON for censored events, denominators and per-eye results. These are development results, not real-world accuracy or a speed benchmark.

## Inspect and reproduce

1. Restart `Blink Review.command` to load the new source.
2. Choose reviewer ryan and run `20260918T023223_8c0e2e` in Read results, then Score my current labels.
3. Below the summary, the mistakes table shows the exact eye/frame, confidence, algorithm reason and reviewer note. Open frame jumps to the image without changing its label. Rescore after editing labels.
4. Download the report and labels. The run folder also contains the exact label snapshot used for this handoff; current labels can produce different scores.
5. To reproduce inference, use Try settings: minimum 0.7, recovery confidence 0.7, recovery duration 0.05, frame limit 0. Then run both modes and score the same label snapshot.

```sh
cd /path/to/EyeTracker_v2
.venv/bin/python -m unittest test_pupil_detection -q
```

## What the team must finish

1. **Review the tradeoff.** Inspect new false rejections and remaining missed coverage with a teammate. The image rule can confuse illumination gradients with coverage; low contrast and segmentation retreat from crop boundaries remain failure modes. A human-usable image can still fail geometry checks or be withheld during recovery.
2. **Agree on acceptance targets before unseen testing.** Record the maximum acceptable bad-frame admission rate, good-frame loss and recovery delay, plus minimum coverage of usable/unusable examples. The software does not choose those application requirements for you.
3. **Collect fresh paired videos.** Include open eyes, natural blinks and partial closures under representative gaze and lighting conditions. Preserve the configured resolution, rotations and ROI, or explicitly create a new configuration and matching labels. Obtain independent labels before reviewing algorithm outcomes. Do not repurpose the supplied development recordings as held-out data.
4. **Freeze and evaluate.** Record the Git commit, source fingerprint and settings. Run the held-out workflow below, label both eyes, run the full paired comparison, then score. Settings and implementation are frozen by the first held-out run. If you tune after seeing those outcomes, treat that data as development and acquire another evaluation set.
5. **Reproduce on another team computer.** Publish the reviewed source and instructions on the team branch, install dependencies in a new environment and run tests. pye3d may require Eigen 3/build tools; the current review server uses Unix file locking and is not yet native-Windows compatible. Do not copy `.venv` between computers.

```sh
.venv/bin/python blink_validation.py --split held_out \
  --left-video "/absolute/path/new_lefteye.mp4" \
  --right-video "/absolute/path/new_righteye.mp4"
```

Use the agreed frozen settings in Try settings before starting the first held-out run. Start a separate reviewer identity for independent reviews; joint consensus review is not independent agreement. Labels and reports live in ignored `blink_validation_data/`, so back them up separately from Git.

## Scope and claims

The present output establishes software behavior and development filtering performance only. It does not validate physiological blink counts, generalization to other recordings, or accurate gaze. Per-camera checkerboard captures and independent known gaze targets remain necessary for calibrated 3D/gaze claims; those are separate from finishing the blink-filter evaluation.

Do not mark the research mini-project validated until the fresh-recording result meets the agreed criteria and a teammate can reproduce the workflow. No new recording or human judgment has been invented to fill these gaps.
