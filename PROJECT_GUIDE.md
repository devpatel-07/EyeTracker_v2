# EyeTracker v2 project guide

**September 30 status:** the project owner accepts current fixed-crop blink-filter performance for continued development. Independent-recording validation is deferred; it is not a prerequisite for implementing the next development feature. The optional pupil ROI tracker is implemented and documented in [ROI_DEVELOPMENT.md](ROI_DEVELOPMENT.md), including processing order, how to run it, and measured results. Fixed crops remain the default. Earlier status reports and requests for fresh data below describe prior milestones, not additional work required before trying tracking.

**Latest development handoff:** see [BLINK_HANDOFF.md](BLINK_HANDOFF.md) for the current results, mistake navigation and remaining validation checklist. Historical results below remain for context.

## File organization update (September 17, 2026)

This is a mechanical extraction: algorithm bodies, defaults, launch commands,
labels and report formats are unchanged. Existing imports from
`pupil_detection` and `blink_validation` remain supported.

| File | What to edit here |
|---|---|
| `pupil_detection.py` | Neural inference, pupil ellipses and image-based eyelid evidence |
| `pupil_quality.py` | Shared observation records, frame acceptance and temporal recovery |
| `blink_validation.py` | Local server, label persistence and comparison-run orchestration |
| `blink_scoring.py` | Scoring records, reviewer agreement and report text |
| `blink_review.html` | Browser interface, with CSS and JavaScript in the same file |

The new quality module is included in the pipeline source fingerprint. New runs
therefore have a different fingerprint; run both comparison modes together.
Historical paired runs and their saved labels remain usable. Restart the review
server when you want it to load the reorganized files. Keep the HTML file beside
`blink_validation.py` when sharing the project. No new dependencies are needed.

The Word handbook predates this extraction; its method explanations still apply,
but use this table for the new locations. Historical step notes below describe
the implementation at the time of each step.

This project processes paired eye-camera recordings. A TinyUNet segments the
pupil in each eye's region of interest (ROI), OpenCV fits an ellipse, camera
calibration corrects its geometry, and a separate pye3d temporal model for each
eye estimates the eye center, pupil center, and pupil diameter. OpenCV windows
show the results; optional console output exposes measurements and status.

## Current completion checklist

**The blink-filter software is implemented and running. Human validation is the
next unfinished part.** Its job is to keep unsuitable eye images out of pye3d,
the model that learns 3D eye geometry from a sequence of pupil measurements.
A **frame** is one image from the video. A **gate** is the accept/skip decision
made before that image's measurement reaches the 3D model.

“Implemented and tested” below means that the code exists and its behavior has
been checked. It does not mean that every real blink is detected or every
accepted measurement is correct. The numbered steps match the detailed sections
later in this guide; those sections also contain historical results from earlier
versions.

| Step | What is finished, in plain language | Main code to read | What still needs evidence |
| --- | --- | --- | --- |
| 1. Look for possible eyelid closure | The detector looks for a supported eyelid edge and whether it covers the pupil. It adds an explanation without changing the original image or pupil measurement. Implemented and tested. | `pupil_detection.py`: `detect_eyelid_closure()`, `PupilDetector.detect()` | These are experimental appearance cues, not confirmed physiological blink events. Partial pupil coverage remains difficult. |
| 2. Judge one frame | Missing/invalid ellipses, low pupil confidence, and possible closure/coverage can produce a skip decision with a reason. Implemented and tested. | `pupil_detection.py`: `FrameQualityDecision`, `assess_pupil_quality()` | Human labels are needed to measure how often the decision is wrong. A missing pupil alone does not establish a blink. |
| 3. Wait for recovery | After rejection, the eye must produce sufficiently strong measurements for a short continuous period before updating the model again. Each eye has its own memory. Implemented and tested. | `pupil_detection.py`: `TemporalQualityTracker.update()` | The 0.60/0.70 confidence cutoffs and 0.05-second hold are engineering starting points, not validated biological thresholds. |
| 4. Enforce the decision | Rejected and recovering frames actually skip the pye3d call. Skips preserve the model's history and have no current 3D geometry; the next accepted frame uses its real video timestamp. Implemented, tested and exercised on both complete recordings. | `eye_pipeline.py`: `process_frame_loop()`; `eye_model_estimation.py`: `skip_update()` | Successful execution proves the gate works, not that it chooses the correct real frames. |
| 5. Protect corrected coordinates | After lens correction, invalid ellipse geometry or an out-of-bounds center is rejected before pye3d stores it. This rejection uses the same recovery logic. Implemented and tested. | `eye_pipeline.py`: `_prepare_corrected_model_input()`; `eye_model_estimation.py`: `assess_pye3d_input_geometry()` | This protects the model's input contract; it does not establish whether the physical camera calibration is correct. |
| 6. Explain low confidence and test known geometry | Native fitted values and the exact failed/unavailable pye3d checks are exported and displayed. Synthetic pupils with known answers test rotations, coordinate offsets and 3D reconstruction. Implemented and tested, including installed pye3d. | `eye_model_estimation.py`: `diagnose_model_output()`; `test_pupil_detection.py`: `KnownCameraGeometryTests`, `KnownPye3DGeometryTests` | The exact failed checks are identified, but the recordings still fail many eye-center ranges. Physical accuracy has not been established. |
| 7a. Correct camera math and prepare separate calibration | The full ellipse is normalized for unequal focal lengths. Checkerboard collection/fitting, camera IDs, provenance and mismatch checks are available. Implemented and tested with synthetic and rendered checkerboards. | `eye_model_estimation.py`: `_focal_scales()`, `_scaled_ellipse_shape()`; `calibration.py`: `calibrate_from_recordings()`, `fit_checkerboard_calibration()`, `CameraCalibration.load()` | Actual checkerboard recordings are still needed from each physical camera. Both eyes currently use the legacy file with unverified identity. |
| 7b. Make reliability explicit | Finite geometry, pye3d range checks, temporal consistency and camera identity are separate statuses. A drift warning does not stop otherwise eligible observations from helping the model learn. Implemented and tested. | `eye_model_estimation.py`: `EyeModelEstimate`, `ModelStabilityMonitor`; `feature_output.py`: `frame_record()`, `create_output_frame()` | “Stable,” “ready,” or “checks passed” does not mean measured gaze accuracy. Stability thresholds need reference-data validation. |
| 7c. Prepare independent gaze evaluation | Saved directions can be compared with independent reference directions. Optional rotation fitting uses earlier samples; later samples provide angular error and coverage. Implemented and tested with synthetic references. | `gaze_validation.py`: `fit_direction_rotation()`, `evaluate_gaze()` | Measured target directions and the physical camera/target geometry are still missing. No real gaze-accuracy result exists yet. |
| Review and comparison tools | The local app saves separate reviewers' labels, runs baseline and filtered pipelines, scores errors/recovery, and compares reviewers. The launcher and a complete comparison are ready. | `blink_validation.py`: `save_label()`, `start_run()`, `score_records()`, `compare_records()`, `reviewer_agreement()`; `Blink Review.command` | Humans must supply the labels, use development results to choose settings, then evaluate fresh recordings with frozen settings. |

The latest automated suite passed **157 tests**. The saved default comparison
`20260917T145404_ac78e5` completed **1,267 paired frames in each mode**, producing
2,534 eye/frame records per mode. Its artifacts are in
`blink_validation_data/664d205892bd8028f01e/runs/20260917T145404_ac78e5/`.

| Saved run | Measurements submitted to pye3d | Skipped measurements | Finite geometry returned |
| --- | ---: | ---: | ---: |
| Baseline: confidence and geometry protections | 2,409 | 125 | 2,402 |
| Blink filtering: adds eyelid checks and recovery | 2,304 | 230 | 2,299 |

**The extra 105 skips are not yet proven correct.** At setup, the app's human
label directory is empty and its saved report says `no_scorable_human_labels`.
The old development observations discussed later are not prefilled as human
truth. The filtered export still has 2,303 range-failing model calls out of
2,304; after five seconds the only failed range is eye-center y. It also records
14 right-eye drift warnings. These are visible findings, not problems silently
fixed by accepting finite geometry. Every camera identity remains `unverified`.

Your next three blink-project tasks are:

1. **Label the 282 starter eye frames.** The six suggested intervals contain
   141 frames per eye. Each reviewer uses their own name and judges the same
   frames independently as usable, unusable or uncertain. Finish both eyes;
   discuss disagreements afterward rather than copying another person's labels.
   These selected development intervals are a starting review set, not a random
   sample of all future recordings.
2. **Score, inspect and tune on development data.** Use the saved comparison
   first. Inspect bad frames admitted, good frames discarded and recovery delays.
   Change one setting at a time only when the errors justify it, rerun the pair,
   then score the same human labels. Some partial-coverage errors may require
   better image evidence rather than a different confidence cutoff.
3. **Evaluate fresh recordings with frozen settings.** After development choices
   are settled, collect unseen data and use `--split held_out`. Review it
   independently and run the complete comparison. Keep those results for final
   evaluation instead of using them to choose new settings. The app refuses the
   known development recordings and freezes the held-out settings/implementation.

Physical per-camera checkerboards and measured gaze targets are a **separate
requirement for calibrated 3D/gaze claims**. You can begin the three blink-review
tasks now. No dedicated blur classifier or physiological blink-event counter is
currently implemented; the current work judges measurement eligibility.

## Where to start

| File | Responsibility | Useful extension points |
| --- | --- | --- |
| `eye_pipeline.py` | Presets, startup, paired frame loop, cleanup | Configuration, synchronization, export integration |
| `video_preparation.py` | File selection, rotation preview, capture metadata | Input sources, ROI controls, orientation setup |
| `pupil_detection.py` | TinyUNet loading, pupil measurement, experimental eyelid diagnostics | Model replacement, preprocessing, confidence scoring, closure evidence |
| `calibration.py` | Per-camera checkerboard fitting, provenance, coordinate/distortion transforms | Camera handling, geometry validation |
| `eye_model_estimation.py` | Adapt observations to pye3d; maintain model state and normalize results | 3D estimation, quality criteria, coordinate conversion |
| `feature_output.py` | Overlays, console formatting, structured frame records | Display styles and output presentation |
| `gaze_validation.py` | Offline comparison to independent reference directions | Held-out angular error and coverage |
| `blink_validation.py` | Local review interface, saved human labels, paired runs and reports | Review workflow, frame-filter scoring, reviewer agreement |

Read the module docstrings first, then `run_pipeline()` and
`process_frame_loop()` to follow the calls across these files.

## Running the current workflow

Use the dependency versions in `requirements.txt`. The combination validated on
this project is Python 3.12, NumPy 2.2.6, desktop OpenCV 4.12.0.88, PyTorch 2.14.0,
and pye3d 0.3.2. OpenCV 5 changes one-dimensional matrix behavior and crashes
pye3d 0.3.2's Kalman filter on its second update, so the estimator stops early
with a clear version error if OpenCV 5 is installed.

On Apple Silicon, pye3d 0.3.2 has no native PyPI wheel for Python 3.12. The
working project environment contains a source-built copy. Preserve that
environment when possible; rebuilding pye3d requires Eigen 3.x rather than the
incompatible Eigen 5 package currently supplied by Homebrew. The requirements
file records runtime versions but cannot install this system build dependency.
The project does not include the TinyUNet training implementation.

From the project directory, run:

```sh
python eye_pipeline.py
```

1. Choose the left and right recordings, unless their paths are set in the
   presets at the top of `eye_pipeline.py`.
2. Review the four rotation panels. ROI rectangles are fixed presets; the GUI
   rotates frames but does not provide ROI editing. Calibration panels are
   source-video previews, not calibration images or an automatic calibration.
3. Confirm the rotations. Both captures rewind so the preview frames are
   included in processing.
4. Watch the two output windows. Press `q` in an OpenCV window to stop. Processing
   also stops when either capture ends/fails or a positive `MAX_FRAMES` is reached.

GUI choices apply to the current run. To persist them, edit the presets.
`TEXT_OUTPUT = True` enables console records; the current code does not save
CSV data or rendered video. `--output run.jsonl` now saves structured results;
`--headless` bypasses windows and uses the configured rotations. `WAIT_MS` adds a window-event/key wait after
processing and is not a playback-rate controller.

## Easy blink validation workspace

Double-click **`Blink Review.command`** in Finder. It opens a local browser page
and uses this project's existing `.venv`; no new packages or online account are
required. Begin on **Start here** and follow the guided next-step button. Keep
the launcher terminal open while reviewing. Control-C stops the
server and any active comparison. Closing only the browser does not stop the
server. Your completed labels and runs remain on disk.

**Continue guided labeling →** opens the next unreviewed starter frame. It
works through the intervals, covering the left and right eye in each one. Once
all 282 starter eye frames have a saved label, the guide sends you to Read
results. An uncertain label records an honest judgment and counts as reviewed;
it is excluded from mistake-rate calculations. Progress always belongs to the
currently selected reviewer.

1. **Label frames.** Enter a short reviewer name (letters, numbers, hyphens or
   underscores) and choose Start / resume. Select an eye and review interval.
   The six starter intervals cover 282 eye frames across both eyes. They cover
   earlier development examples, but labels are
   intentionally blank. Judge the raw pupil crop without algorithm overlays.
   Use **1 = usable, 2 = unusable, 3 = uncertain**, or click the buttons. Arrow
   keys step frames. Each choice saves immediately and normally advances one
   frame. Use Previous to correct a label. Type a note before choosing, or use
   Save current label + note. Full-frame view provides context; labeling is
   enabled only in crop view. Use a different reviewer name for each person.
2. **Read results.** A full default comparison of the current recordings is
   already saved from setup. Select it and click Score my current labels.
   The report gives bad frames admitted, good frames discarded, recovery delays,
   per-eye counts, reason groups, output coverage, and descriptive model-center
   movement and runtime. Add labels and score again without rerunning inference.
   Uncertain/unreviewed frames do not enter mistake-rate denominators. A category
   with no examples says so instead of displaying an invented zero-percent error.
   Downloads include labels and a detailed JSON report; a readable Markdown
   report is also saved automatically. Reports compare reviewers where both
   labeled the same frames and list disagreements for discussion.
3. **Try settings.** Use this after reading the initial results. Change one
   setting at a time and click Run baseline + blink filter to generate another
   comparison. Both runs process the same videos, start fresh pye3d models,
   and keep pupil confidence and geometry protections. Baseline bypasses only
   the extra eyelid checks and recovery hold. Progress appears in the browser.
   You can keep labeling while runs execute, though this can affect timing.
   A frame limit of 0 means the complete recording; a short run cannot score
   labels from later frames. Return to Read results to score the new pair.

All saves live under **`blink_validation_data/<dataset-id>/`**, which is ignored
by Git. `recording.json` binds the labels to video hashes, rotation and ROI;
`labels/<reviewer>.json` holds one person's reviewed judgments; `runs/<run-id>/`
contains settings, both exports/logs/summaries and reports. Back up this folder
with your project. Revisiting with the same name resumes your labels, and the
name field suggests existing reviewers. Reviewer names are case-insensitive.
Atomic saves plus file locking protect labels from interrupted writes and two
launchers saving concurrently. A stale tab must reload the reviewer before it
can overwrite newer work.

These recordings were already used during development. Treat their scores as
preliminary; the application will not label them held-out data. Partial pupil
coverage is a known weakness and may require improving the image checks rather
than raising confidence thresholds. Smaller model-center movements do not prove
better gaze accuracy; known target data is still needed for that claim.

The launcher uses the only left-eye/right-eye MP4 pair in this folder. To select
another pair, use the existing virtual environment in Terminal:

```sh
.venv/bin/python blink_validation.py \
  --left-video /path/to/new_lefteye.mp4 \
  --right-video /path/to/new_righteye.mp4
```

Configure ROI, rotations, calibration paths and camera identities in
`eye_pipeline.py` before reviewing a new recording. Changes to video bytes,
rotation or crop create a separate review workspace. For **fresh evaluation
recordings only**, add `--split held_out`. Choose settings using development data
first. The first held-out comparison freezes thresholds and implementation;
later changes are refused for that workspace, and only full-recording runs are
allowed. The software enforces this bookkeeping, but independent collection and
careful human review remain necessary.

If you prefer direct pipeline commands, new options include `--filter-mode
baseline|filtered`, `--min-confidence`, `--recovery-confidence`,
`--recovery-duration`, `--run-summary`, and `--progress-every`. The ordinary
pipeline still defaults to filtered mode. Its JSONL now records comparison mode,
mask threshold, device and implementation fingerprint. The comparison refuses
mismatched recordings, calibration, settings or source implementations.

The review workflow uses `blink_validation.py` for the local server and label
persistence, `blink_scoring.py` for scoring, and `blink_review.html` for the interface.
`eye_pipeline.py` owns the actual baseline/filtered execution, and the existing
`test_pupil_detection.py` holds all tests. `gaze_validation.py` remains the
separate tool for measured 3D reference directions; it is not required to score
blink-filter decisions.

## How a frame becomes a measurement

```text
Read left/right frame pair → rotate full frames → crop each ROI
    → grayscale, resize, normalize → TinyUNet logits → sigmoid probabilities
    → resize probabilities to ROI → threshold mask → largest plausible contour
    → ellipse + observation confidence → restore full-frame coordinates
    → image-level quality checks → sample boundary, undistort, refit ellipse
    → check the corrected pye3d image domain → update per-eye recovery state
    → either update pye3d or return an explicit skipped estimate
    → scale 3D lengths; transform projections back to display pixels
    → assess per-eye temporal consistency without stopping model learning
    → save requested JSONL records; draw overlays unless running headless
```

Both eyes share one segmentation network in evaluation mode. Each has its own
calibration object and pye3d model history. Calls are sequential and interleaved
by stage; no stereo triangulation or shared binocular coordinate system is
implemented.

## Coordinate and data contracts

- **Image sizes:** `(width, height)`; NumPy image shapes are `(height, width)`
  or `(height, width, channels)`. OpenCV color frames use BGR channel order.
- **ROIs:** `(x, y, width, height)` in the full frame after `FRAME_ROTATION`.
  They must fit entirely inside that frame; detection rejects invalid crops.
- **Ellipses:** `((center_x, center_y), (axis_0, axis_1), angle_degrees)`.
  Axis lengths are full diameters in pixels. Detector output includes the ROI
  offset and still has lens distortion.
- **Rotations:** raw calibration image → `CALIBRATION_ROTATION` → source video
  → `FRAME_ROTATION` → processed frame. Their sum modulo four quarter-turns
  defines the total calibration-to-processed rotation.
- **Lens correction:** rotate points into the raw calibration orientation,
  apply distortion math there, and rotate back. Only geometry is corrected;
  the image used for display stays distorted. Ellipse correction samples 72
  boundary points by default and refits, so it is an approximation.
- **pye3d input:** the adapter orders ellipse axes minor then major, scales the
  complete ellipse to a virtual camera with `f = (fx + fy) / 2`, and shifts
  its principal point to the image midpoint. The coordinate equations are
  `x_model = (x - cx) * f/fx + width/2` and similarly for y. An affine shape
  transform updates both axes and angle, preserving rays when `fx != fy`.
  Merely averaging the focal lengths without transforming pixels was an older
  approximation, replaced in step 7. The grayscale frame itself is
  neither undistorted nor shifted; review that alignment if changing pye3d's
  image-based search behavior. After the shift, the ellipse center must remain
  in the half-open rectangle `[0, width) x [0, height)`; the pipeline checks
  this before pye3d can store the observation. The complete fitted ellipse is
  allowed to extend beyond the image because partial visibility alone does not
  prove that its inferred center is unusable. Corneal refraction correction is
  explicitly disabled; this is separate from camera lens-distortion correction.
- **3D output:** `EyeModelEstimate` contains optional camera-space XYZ values
  and pupil diameter, scaled by `EYE_RADIUS_MM / PYE3D_REFERENCE_EYE_RADIUS_MM`.
  Lengths depend on the assumed eye radius; changing that preset scales reported
  lengths without changing pye3d's internal model or its displayed projection.
  Left/right values belong to different camera frames and cannot be directly
  combined without a transform between those frames.
- **Display output:** projected model geometry is inverse-scaled, shifted back and distorted
  again to align with the rotated source frame. The drawn ray extends the 2D
  eye-center-to-pupil-center segment; it is not a calibrated screen gaze point.

## Confidence, timing, and missing data

`MASK_THRESHOLD` is a per-pixel segmentation cutoff. Observation confidence is
the mean model probability inside the selected contour multiplied by the
contour/ellipse intersection-over-union. This is a heuristic quality score,
separate from pye3d's `model_confidence`.

No acceptable contour produces `ellipse=None`, `blink=True`, and `confidence=0`.
The legacy `blink` flag also covers bad framing and failed segmentation; it
is not a confirmed physiological blink. Separate experimental eyelid evidence
is now attached to the observation (see below). A low-confidence ellipse can
still be drawn even when it is excluded from the 3D update.

Missing or low-confidence observations skip the temporal update and produce
an empty estimate, while the estimator retains its internal history. Accepted
update timestamps must strictly increase. `ready=True` requires finite,
positive-depth geometry and a positive pupil diameter; the code does not impose
a separate model-confidence threshold or prove that the model has converged.
`update_time_ms` measures only `update_and_detect`, not total frame latency.

`model_diagnostics` separately explains the default pye3d output-range checks.
It preserves native values before eye-radius scaling and lists failed or
unavailable checks. A skipped frame has `model_diagnostics=None`; a model call
that produces unusable geometry can still carry diagnostics for investigation.
The overlay distinguishes geometry availability from these range checks. Neither
`ready=True` nor `model_confidence=1.0` proves that gaze measurements are accurate.

The loop pairs equal frame indices and derives timestamps as `frame_index / fps`.
Matching FPS metadata does not establish synchronized start times, repair dropped
frames, or account for variable-rate recordings. New sessions should create new
estimators rather than reuse an old temporal model with timestamps reset to zero.

## Included assets and assumptions

- The two supplied MP4 files report 1920 × 1080 at 30 FPS. The left has 1,271
  frames; the right has 1,267. The default loop processes at most 1,267 pairs.
- Default frame rotations produce 1080 × 1920 images with ROI
  `(0, 100, 1080, 698)` for each eye.
- `camera_calibration.npz` contains a 1920 × 1080 calibration with unknown device
  provenance. The loader reads its intrinsics but does not promote its stored
  error to an accuracy claim. Presets prefer `left_camera_calibration.npz` and
  `right_camera_calibration.npz` when present, otherwise fall back to this legacy
  file and warn that identity is unverified. Resolution checks do not prove that a
  calibration belongs to a particular camera or that its orientation is correct.
- The active checkpoint is
  `models/finetuned_2026-08-25/pupil_unet_best.pt`, with 16 base channels and a
  320 × 192 grayscale input. The other checkpoint and both training-history JSON
  files are reference assets; the pipeline does not read the histories or JPG.
- The checkpoint loader expects `config` with `base_channels`, `input_width`,
  and `input_height`, plus `model_state_dict` matching TinyUNet. Its three pooling
  levels require input dimensions divisible by eight. Preprocessing stretches
  the ROI to that size and divides 8-bit pixels by 255; preserve the training
  convention when changing models.
- `DEVICE = "auto"` selects CUDA if available, otherwise CPU. It does not
  automatically select Apple's MPS backend.
- `__pycache__` contains generated Python bytecode, not source to maintain.

## Building on the project

### Experimental eyelid-closure preview

The first implementation stays in `pupil_detection.py`, with a small overlay in
`feature_output.py`. It adds no runtime module, model weight, or dependency.
`PupilDetector.detect()` now attaches an `EyelidObservation` to its result:

```python
observation = pupil_detector.detect(frame, roi)
print(observation.eyelid.state)
print(observation.eyelid.reason)
```

`detect_eyelid_closure()` smooths a small grayscale copy of the eye crop, looks
for a bright-to-dark transition shared across many columns, and checks its
position relative to the measured pupil. When a reliable pupil is available,
edge support must come from both sides outside it. The image and neural-network
input are not modified. Purple dots show supported edge samples; the text line
is a preview, independent of pye3d's ready status.

| State | Meaning |
| --- | --- |
| `unknown` | Insufficient or ambiguous evidence; no open/closed claim |
| `no_closure_evidence` | Supported edge does not overlap the measured pupil |
| `occlusion_possible` | Supported edge overlaps the upper part of the pupil |
| `closed_possible` | Broad low edge and an unreliable pupil suggest closure |

These are appearance candidates, not blink events. A single frame cannot tell
closing from reopening. The low-edge rule assumes the supplied upright, fixed
eye crops; different framing, roll, eyelashes and downward gaze can confuse it.
It is not an anatomical eyelid-gap measurement. The 0.60 pupil-reliability cutoff
inside this diagnostic is separate from the eye model's acceptance threshold.
During the original preview-only step, these candidates did not reject model
inputs. **In the current filtered pipeline, steps 2–4 use closure/coverage
candidates as rejection evidence and enforce the resulting decision.** The
detector still preserves the original ellipse, confidence and legacy blink
flag. Its helper is stateless, so sharing the detector between eyes is safe;
the step-3 quality trackers keep separate history per eye. Baseline comparison
mode bypasses the added eyelid checks and recovery hold.

On 183 selected frames from the bundled recordings, the prototype flagged 25
of 31 pupil-hidden frames and 7 of 23 partially covered frames. None of 129
apparently open or lateral-gaze controls received a closure flag, but many
returned `unknown`. These development checks used the same recordings inspected
while developing the method, not independent validation or blink-event accuracy.
Partial coverage remains a substantial limitation. During the original eyelid
preview change, pye3d was not installed in the available Python environment, so
that step tested pupil inference and display rendering separately. Later sections
record the complete real-video pye3d runs performed after the environment repair.

Run the automated suite without loading the bundled recordings:

```sh
python -m unittest test_pupil_detection -v
```

The original eyelid checks do not need pye3d. The expanded suite also exercises
the actual pye3d geometry when installed; those specific tests are explicitly
skipped when it is unavailable.

They cover unsupported/blank/noisy input, isolated glare, pupil-size changes,
partial coverage, crop boundaries, coordinate offsets, input preservation and
independence between calls. Six real-frame regression checks also confirmed
that existing pupil ellipses, confidence scores and legacy flags were unchanged.

Research context: [Chen and Epps (2019)](https://www.frontiersin.org/journals/ict/articles/10.3389/fict.2019.00018/full)
explains why missing pupils, eyelashes and near-field IR appearance complicate
blink detection. This lightweight preview is not a reproduction of their
trained shape-model method.

### Frame-quality decision (step 2)

This step also lives in `pupil_detection.py`; no `quality_gate.py` is needed.
`FrameQualityDecision` stores a proposed action and its reason. The function
`assess_pupil_quality()` reads an existing pupil observation and returns that
decision without changing the observation or calling the eye model:

```python
pupil = pupil_detector.detect(frame, roi)
decision = assess_pupil_quality(pupil, min_confidence=0.60)
print(decision.allow_model_update, decision.reason)
```

The checks run in order, so the first failure supplies the reason:

1. Reject nonfinite/out-of-range pupil confidence.
2. Reject a missing ellipse, then malformed/nonfinite geometry or nonpositive axes.
3. Reject pupil confidence below the supplied whole-observation threshold.
4. Propose skipping `closed_possible` or `occlusion_possible` eyelid evidence.
5. Otherwise propose accepting the measurement. Unknown/missing eyelid evidence
   is explicitly reported as uncertain; acceptance means only that these checks
   passed. The legacy `blink` boolean is not separate evidence and is ignored.

A bad configured threshold raises `ValueError`. A bad observation returns a
skip decision. The function has no temporal memory: it cannot identify a
whole blink, distinguish closing/reopening, or enforce recovery delays.

The step-3 tracker calls this function with the pipeline's `MIN_CONFIDENCE` for
each eye. `feature_output.py` displays the quality decision and reason. With `TEXT_OUTPUT=True`,
the console records whether the model input was accepted or skipped, along with
`quality_state` and `quality_reason`. The optional output-function arguments preserve
older calls that supply no decision.

The single-frame function itself only returns a decision; the pipeline now enforces
that decision after the temporal tracker described below. Rejected/recovering frames
are not submitted to pye3d.

The prior eyelid code now includes input/output examples and inline explanations
of grayscale/array dimensions, resizing, signed brightness differences, local
maximum searches, rotated ellipse extents, Boolean masks, axis reductions, row
selection and coordinate restoration. Its existing executable behavior is
unchanged. Added tests cover the decision policy and paired-frame-loop integration,
including enforcement of skip decisions and independence between the two eyes.
All tests remain in the existing `test_pupil_detection.py`.

### Per-eye recovery tracking (step 3)

`TemporalQualityTracker` also lives in `pupil_detection.py`. It remembers whether
this eye recently failed a quality check, so one promising frame cannot instantly
end recovery. `eye_pipeline.py` creates one tracker for the left eye and another
for the right eye before entering its frame loop. They share no quality history.

```python
tracker = TemporalQualityTracker(
    min_confidence=0.60,        # Cutoff while measurements remain usable.
    recovery_confidence=0.70,   # Stronger score needed after any rejection.
    recovery_duration_s=0.05,  # Required span of consecutive strong samples.
    max_gap_s=1.5 / fps,       # Largest permitted gap in supplied timestamps.
)
# Repeat for EVERY consecutive frame; do not recreate the tracker here.
decision = tracker.update(pupil, timestamp_s=frame_index / fps)
print(decision.state, decision.allow_model_update, decision.reason)
```

| Quality state | What it means | Pipeline action |
| --- | --- | --- |
| `usable` | Current checks pass and no recovery hold remains | Accept |
| `rejected` | The current single-frame checks fail | Skip immediately |
| `recovering` | Current checks pass, but stronger/longer evidence is still needed | Skip |

For example, a rejected frame at time 0 followed by scores of 0.80 at 1/30,
2/30 and 3/30 seconds produces `recovering`, `recovering`, then `usable`.
Three strong samples span about 67 ms at 30 FPS, exceeding the configured 50 ms.
A score below 0.70 restarts this hold even if it passes the ordinary 0.60 cutoff.
Once usable, a score of 0.65 may pass again. Using different thresholds depending
on the current state is called **hysteresis**. If recovery scores never reach
0.70, the tracker keeps waiting; this can discard usable measurements too.

`update()` returns a `FrameQualityDecision`; it never edits the observation.
`recovery_elapsed_s` reports the strong-sample span while waiting, and is zero
outside the hold. All temporal acceptances must also pass the single-frame
checks. At startup, ordinary checks apply immediately. `reset()` forgets quality
history when seeking/restarting; it has no effect on the separate 3D model.
Nonfinite, negative, duplicate or decreasing timestamps raise `ValueError`
without changing history. Large timestamp gaps restart recovery.

The optional `additional_rejection_reason` argument lets a later camera-geometry
check reject an otherwise acceptable observation. Raw pupil or eyelid failures
keep priority. A corrected-geometry failure enters the same per-eye recovery
state, but it is still a measurement-quality failure rather than proof of a
physiological blink.

The pipeline currently estimates time as `frame_index / fps`, assuming uniform
sampling. Consequently it cannot detect missing capture frames or variable-rate
presentation-time gaps. The tracker's gap handling works when supplied timestamps
actually expose a gap. Wall-clock processing time is unsuitable for video replay.
No future frames are consulted and decisions are never backdated. These states
describe measurement quality; they do not identify physiological blink phases.

The overlay adds `quality state` and the elapsed recovery span. With
`TEXT_OUTPUT=True`, logs include `quality_state` and `recovery_elapsed_s`.
The pipeline enforces these decisions before pye3d. A current image-level
rejection avoids lens correction; an otherwise valid recovery frame is corrected
for the later geometry check but still returns an explicit skipped estimate.
Accepted frames resume using their real later video timestamp. No new runtime
files or dependencies were added.

Validation: the step-3 validation had 42 passing tests, including time boundaries, interrupted recovery,
independent eyes, causal decisions and unchanged 3D-model inputs in a paired-loop
test. Replaying all 2,538 eye frames from both recordings proposes 97 immediate
rejections and 43 recovery holds. On 282 newly annotated frames, 63 were too
uncertain to score. Of 54 visibly unusable frames, single-frame checks would
accept 32; adding recovery reduces that to 27. Of 165 apparently usable frames,
rejections rise from 1 to 9. Thus recovery helps some transitions but still
misses substantial partial coverage and costs usable data. These are same-subject,
same-recording development results, not independent accuracy validation.

The 0.70 and 0.05 defaults are engineering choices, frozen before scoring these
new labels. Research informs the approach, not those numerical thresholds:
[Attivissimo et al. (2023), section 3.7](https://iris.poliba.it/bitstream/11589/261920/1/2023_Performance_evaluation_of_image_processing_algorithms_for_eye_blinking_detection_pdfeditoriale.pdf)
describes hysteresis for blink classification;
[Pupil Core's implementation](https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/blink_detection.py)
uses confidence history and separate onset/offset activity thresholds. Neither
establishes settings for this U-Net score. No upstream code was copied.
[Nyström et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38424292/)
also distinguishes pupil-data loss from eye-openness measurements, motivating
our separation of quality states from blink-event claims.

### Enforced pye3d quality gate (step 4)

`eye_pipeline.py` now uses each eye's `FrameQualityDecision` as an actual gate.
When `allow_model_update` is false, that eye skips
`EyeModelEstimator.update()`. A current raw failure also avoids lens correction;
a raw-valid recovery frame may already have been corrected for the step-5 check.
`EyeModelEstimator.skip_update()` returns an
empty `EyeModelEstimate` with a status such as `quality rejected: no pupil` or
`quality recovering: waiting for stable pupil measurements`. It does **not** call
`pye3d.update_and_detect()` and does **not** advance `last_timestamp`, so the next
accepted sample enters pye3d with its real later video timestamp. The two eyes are
still independent: one eye can be skipped while the other updates normally.

The display now says `model input: accepted` or `model input: skipped` rather than
`would accept/would skip`. With `TEXT_OUTPUT=True`, console records include
`model_input`, `quality_state`, `quality_reason`, and `recovery_elapsed_s`. Raw
eyelid evidence remains labeled as a preview because `EyelidObservation` is still
experimental evidence rather than a confirmed physiological blink classifier.

The step-4 validation contained 46 passing tests. Those tests verify that skipped
frames never call the pye3d boundary, do not advance model history, that the next
accepted frame keeps its actual timestamp, that left/right gating is independent,
and that display/log output reports the enforced action.

Before the post-calibration check in step 5, the full paired recordings were run
headlessly through the real TinyUNet,
calibration, temporal gate, and pye3d 0.3.2. The run completed all 1,267 frame
pairs (2,534 eye frames) without an exception. The left eye submitted 1,188
frames to pye3d and skipped 79; the right eye submitted 1,206 and skipped 61.
Of the 2,394 submitted frames, 2,389 returned finite geometry and five were
normal warm-up results. All 140 skipped records contained no 3D geometry, took
zero pye3d update time, and left the model timestamp unchanged. Ninety-eight
frame pairs accepted one eye while skipping the other, confirming that the two
gates run independently.

This establishes execution and gate behavior, not measurement accuracy. Some
early pye3d estimates were anatomically implausible while the model was still
settling, including a left-eye pupil diameter above 270 mm even though pye3d
reported the result as ready. Downstream work must therefore add convergence and
plausibility criteria instead of treating `ready=True` as validated gaze data.

### Post-calibration pye3d domain gate (step 5)

OpenCV's lens correction can legitimately return coordinates outside the source
image; it transforms points but does not clip them. pye3d, however, assigns
observations to spatial bins using its supplied image size. The pipeline now
checks the corrected center in exactly the coordinates passed to pye3d:

```python
f = (fx + fy) / 2
model_x = (corrected_x - principal_x) * f / fx + width / 2
model_y = (corrected_y - principal_y) * f / fy + height / 2
valid = 0 <= model_x < width and 0 <= model_y < height
```

This is the current coordinate conversion, including the unequal-focal
normalization added in step 7. Here `fx` and `fy` are the camera's horizontal
and vertical focal lengths in pixels; `principal_x` and `principal_y` locate its
optical center. The old shift-only formula is valid only when the two focal
lengths are equal. The adapter also transforms the full ellipse's axes and angle,
not just its center.

`assess_pye3d_input_geometry()` in `eye_model_estimation.py` also rejects
malformed, nonfinite, or nonpositive ellipse geometry. `eye_pipeline.py`
performs the cheap image-level checks first, undistorts an otherwise eligible
ellipse once, then gives any corrected-domain failure to that eye's existing
temporal tracker. The final decision controls both the displayed status and the
real pye3d call. `EyeModelEstimator.update()` repeats the geometry check
defensively so another caller cannot bypass it or advance the model timestamp
with an invalid observation.

The rule checks the center, not the complete rotated ellipse outline. Requiring
every inferred boundary point to fit would reject 552 otherwise accepted
observations in these recordings, including plausible partially visible pupils.
No evidence currently supports that much stronger policy. Aspect ratio, visible
boundary fraction, and ellipse-refit residual are useful future diagnostics, but
they need labeled data before becoming rejection thresholds.

The implementation follows pye3d's
[detector coordinate conversion](https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/detector_3d.py)
and [spatial observation storage](https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/observation.py),
and OpenCV's [`undistortPoints` contract](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).
The temporal isolation matters because the published pye3d method improves its
fit by integrating pupil contours over time; see
[Dierkes, Kassner and Bulling (2019)](https://www.hcics.simtech.uni-stuttgart.de/publications/dierkes19_etra/).

All 54 automated tests pass. They cover half-open boundaries with an off-center
principal point, malformed corrected ellipses, direct-estimator protection,
timestamp preservation, normal recovery after a geometry failure, raw-reason
priority, paired-eye independence, and the rendered/logged action. A complete
CPU replay then processed all 1,267 paired frames:

| Result | Left | Right | Total |
| --- | ---: | ---: | ---: |
| Submitted to pye3d | 1,180 | 1,126 | 2,306 |
| Skipped by the final gate | 87 | 141 | 228 |
| Direct corrected-center rejections | 6 | 73 | 79 |
| Recovery-held frames | 23 | 29 | 52 |
| Finite, structurally ready estimates | 1,176 | 1,125 | 2,301 |
| Normal pye3d warm-up results | 4 | 1 | 5 |

Every skipped record had zero pye3d time and no 3D geometry, and all accepted
timestamps remained strictly increasing. The exact `run_pipeline()` entry point
also reached EOF and rendered 1,267 outputs per eye with no pye3d out-of-bounds
message. The guard found more than the eight observations that previously caused
warnings because pye3d checks a flattened bin number; some individually invalid
horizontal bins can still produce an in-range flattened number.

On the 219 previously hand-scored, non-ambiguous frames, the new rule did not
catch an additional visibly unusable pupil: 27 of 54 were still accepted. It
increased rejection of apparently usable frames from 9 to 11 of 165. This step
therefore fixes the pye3d input contract and model bookkeeping; it does not
improve measured blink discrimination.

The public [Cambridge pupil dataset](https://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/datasets/)
provides labeled 2D pupil ellipses, and its
[EyeRender data](https://www.cl.cam.ac.uk/research/rainbow/projects/eyemodelfit/)
provides synthetic gaze and pupil ground truth. They are valuable future tests
for detection and ellipse-to-3D mathematics, respectively. They do not include
the two physical camera calibrations used here, so they cannot validate this
camera-specific post-undistortion boundary. This step instead uses synthetic
boundary tests, the complete paired recordings with their calibration, and the
existing blind visual labels. The resulting artifacts are in
`EyeTracker_v2_real_video_validation/post_calibration_gate` in the workspace.

### Model-output diagnostics and known-geometry validation (step 6)

`diagnose_model_output(raw)` in `eye_model_estimation.py` inspects the dictionary
returned by pye3d before the project's length scaling. It returns immutable
`ModelDiagnostics` attached to the existing `EyeModelEstimate`. The input gate
still decides which observations train the model; a low output confidence does
not prevent the next eligible observation from improving the fit.

The pye3d 0.3.2 implementation uses these inclusive native-scale ranges:

| Diagnostic name | Allowed values |
| --- | --- |
| `eye_center_x` | -15 to +15 native mm |
| `eye_center_y` | -10 to +10 native mm |
| `eye_center_z` | 15 to 75 native mm |
| `pupil_diameter` | 1 to 9 native mm |
| `gaze_phi` | `degrees(phi) + 90` between -90 and +90 |
| `gaze_theta` | `degrees(theta) - 90` between -80 and +80 |

One failed range normally produces `model_confidence=0.1`; this is a categorical
warning, not a 10% probability or a convergence score. If the pupil normal cannot
produce valid angles, pye3d writes zero angle placeholders. The diagnostics
inspect the normal too and mark those angle checks unavailable. Upstream's later
position/diameter checks can overwrite its invalid-angle confidence of 0.0 with
0.1, so inspecting the confidence number alone loses information.

Set `TEXT_OUTPUT=True` in `eye_pipeline.py` to print the new fields:

```text
model_checks=failed model_failed_checks=('eye_center_y',)
native_eye_center_mm=(-4.44, -10.88, 25.44)
```

The first field is `passed`, `failed`, or `unavailable`. Failed checks and
unavailable checks are both retained if they coexist. `feature_output.py` also
displays `geometry available | model checks failed` with the failed names, so a
finite result is no longer presented as an unqualified green ready indicator.
`ready` and the existing `status` field remain compatible with earlier callers.

Validation now contains 76 tests in the existing `test_pupil_detection.py`.
Fifteen new diagnostic tests cover inclusive boundaries, missing and malformed
values, nonfinite normal placeholders, native-versus-scaled values, continued
updates after a low-confidence output, and fresh skipped-frame/log/overlay data.
Seven new known-geometry tests independently project a 3D pupil circle, including
lens distortion, and exercise all 16 calibration/frame rotation combinations,
ROI offsets, principal-point shifts, ellipse-axis swaps, resolution scaling, and
reported length scaling. Two of these tests use the actual pye3d inverse geometry
and temporal fitter; they are skipped explicitly when pye3d is absent. The
temporal test processes 480 observations across four orientations and compares
the recovered centers, diameter, and direction with known answers.

The full real-video diagnostics run completed all 1,267 paired frames. It kept
exactly the preceding run's 2,306 updates, 228 skips, 2,301 ready outputs and
five warm-up outputs; every recorded 3D value matched exactly. Of the 2,306 model
calls, 2,305 failed a default range and one passed; there were no angular failures
or unavailable checks. After five seconds, eye-center y was the sole failed range.
The exact organizer also rendered a 50-pair smoke run with the new labels.

The separate synthetic experiment uses 2,100 observations and intentionally
matches the pinhole/no-refraction model. In the four matched-focal cases, the
maximum eye-center error during the last 50 frames was below 0.000016 native mm.
In the historical pre-step-7 adapter, using the actual calibration's focal-length ratio with a single
average focal length produced up to 0.190 native mm center error and 0.161 degrees
direction error in this artificial scenario, while confidence remained 1.0.
An 8% mismatch produced much larger error despite confidence 1.0. Repeating an
identical pupil orientation for all 300 frames produced no ready geometry. These
experiments check software conventions and model sensitivity, not real-eye
accuracy; they exclude segmentation error, refraction, anatomy, and sensor noise.

The camera audit found a 0.249% mismatch between `fx` and `fy`, internally
consistent rotation/point mappings, and one shared calibration file with no
camera identity. A field named `reprojection_error` is stored, but its definition
and acquisition provenance are absent. Its value alone cannot establish that
this calibration belongs to either eye camera.

The earlier interpretation of widespread confidence 0.1 as a convergence problem
was too strong. Almost all ready real-video outputs exceed the default y limit.
The x and y limits differ, so the flag also depends on camera-axis orientation:
merely expressing the same centers in the unrotated calibration-camera axes
removes 2,260 of 2,300 position-range failures. This is a diagnostic comparison;
it does not justify changing rotations to make confidence higher. The right-eye
fit also drifts later in the recording, so accuracy and physical stability still
require separate investigation with known targets and camera-specific calibration.

Primary references: the installed version's
[pye3d result/confidence implementation](https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/detector_3d.py),
[Pupil Labs' temporal model explanation and academic references](https://docs.pupil-labs.com/core/developer/pye3d/),
and [OpenCV's projection/distortion equations](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html).
Detailed real and synthetic results and reproducible scripts are in the workspace
directory `EyeTracker_v2_real_video_validation/model_diagnostics`.

### Calibration collection, stability monitoring and reference evaluation (step 7)

The software workflows are implemented. **Physical calibration and real gaze
accuracy remain unmeasured:** no per-camera checkerboard or known-target
recordings have been supplied. Synthetic fixtures test the math and code, not
whether the existing calibration belongs to either camera.

Most additions in this step remain in existing modules. Step 7 added
`gaze_validation.py` because it evaluates saved results independently of the
live loop. The later review workflow adds `blink_validation.py`, described near
the top of this guide. All automated tests remain in `test_pupil_detection.py`.

#### 1. Collect and fit each physical camera

Use each camera separately, with the same lens, focus, resolution and image
orientation as its raw eye recordings. Capture a flat checkerboard at multiple
positions, depths and tilts; cover the image center and edges, keep the entire
board visible, and avoid blur/glare. Do not resize or rotate these files during
calibration. Count **inner corners**, and measure the actual printed square
side length in millimeters. The example values below must match your board.

From the project directory, after collecting the images:

```sh
.venv/bin/python calibration.py calibrate \
  --images /path/to/left_checkerboards \
  --camera-id left-eye-camera --board-cols 9 --board-rows 6 \
  --square-size-mm 20 --output left_camera_calibration.npz

.venv/bin/python calibration.py calibrate \
  --images /path/to/right_checkerboards \
  --camera-id right-eye-camera --board-cols 9 --board-rows 6 \
  --square-size-mm 20 --output right_camera_calibration.npz
```

For video, replace `--images ...` with `--video ... --sample-every 30`.
Use camera serial numbers instead of the example labels if available, and set
matching `LEFT_CAMERA_ID`/`RIGHT_CAMERA_ID` presets or CLI options in the pipeline.
The file contains camera ID, source hashes, detected/used view counts, fitted
intrinsics/distortion, total RMS and per-view RMS reprojection errors.

The default requires at least ten distinct views, numerically independent
homography constraints after correction, and a fitted board-normal span of at
least ten degrees. The numerical/tilt guards are conservative collection
policies, not published accuracy thresholds. Noisy, poorly distributed data can
still pass; inspect captures and validate independently. A low training RMS
alone cannot certify correct intrinsics. Existing output files are protected
unless `--overwrite` is explicit.

`identity_status` means:

- `unverified`: no camera label, including the existing legacy NPZ.
- `declared`: a label exists, with no expected label requested.
- `matched`: the artifact label equals the configured expected label.

A match establishes bookkeeping only. The program cannot inspect which physical
camera made a file. A declared mismatch raises an error. The pipeline option
`--require-calibration-identity` also rejects unlabeled legacy artifacts.

Implementation entry points: `fit_checkerboard_calibration()` consumes detected
2D corner arrays; `calibrate_from_recordings()` detects those corners in images
or video; `save_checkerboard_calibration()` writes the validated artifact;
`CameraCalibration.load()` enforces the runtime coordinate/identity contract.

#### 2. Run and save every frame decision

The current recordings can be processed now with their explicitly unverified
legacy calibration:

```sh
.venv/bin/python eye_pipeline.py --headless \
  --left-video "cam_2026-08-15 17-10-39_lefteye.mp4" \
  --right-video "cam_2026-08-15 17-10-39_righteye.mp4" \
  --output run.jsonl
```

Review the interactive rotation setup first. Headless runs use the presets and
never open the rotation GUI. Add `--max-frames 100` for a short run. After making
separate camera artifacts, add `--require-calibration-identity` and, if needed,
explicit `--left-calibration`/`--right-calibration` paths. Export uses exclusive
file creation, so choose a new output filename for each run.

Every eye/frame gets a JSON record, including skips and startup. Records include
video/calibration/checkpoint hashes, rotations, intrinsics, gate settings,
frame index, video timestamp, gate decision, geometry, model ranges, temporal
consistency and the unit `gaze_direction_camera`. That direction is pye3d's
**pupil normal in the processed eye camera**, not a calibrated visual axis or a
scene/world vector. It is absent on skipped/invalid outputs. None of the checks
is a measured gaze-accuracy verdict; nonfinite optional values become JSON null.
An interrupted export is partial; timestamps are still frame index divided by
FPS, and cannot diagnose capture synchronization or variable-rate timing.

`_focal_scales()` and `_scaled_ellipse_shape()` now normalize unequal horizontal
and vertical focal lengths exactly under the pinhole model. The inverse
transform restores display ellipses and locations. Tests independently project
known 3D circles through a camera with a 30% focal difference and check recovered
centers, pupil normals and ellipse projections in all four rotations. This fixes
the prior single-focal approximation; it cannot fix an incorrect physical
calibration or the disabled corneal-refraction model.

#### 3. Read temporal consistency separately from model ranges

`ModelStabilityMonitor` lives in `eye_model_estimation.py`. The paired loop owns
one monitor per eye and supplies every timestamp, using no center on a skipped
frame. The monitor compares a recent median center with a fixed reference median
in **native pye3d millimeters**, before the assumed eye-radius scale.

Defaults exclude five startup seconds, require two observed seconds and twenty
samples, warn above two millimeters of reference displacement or one millimeter
of recent spread, and restart after an observation gap longer than half a second.
A brief missing interval earns no observation-time credit. Missing outputs
contain no stale center or drift measurement. A long gap deliberately clears the
reference, so stability across that gap is unknown. Thresholds are configurable
constructor arguments and are engineering starting points, not validated
physiological or convergence limits.

The overlay/export distinguishes `warming_up`, `insufficient_history`, `stable`,
`unstable`, `drift_detected`, `missing` and `data_gap`. **Stable means consistent
under these thresholds, not accurate.** These output diagnostics never block an
otherwise eligible observation from updating pye3d. The original upstream range
checks remain visible independently, including their rotation-sensitive bounds.

#### 4. Evaluate later against independently measured targets

Create a target JSON using the schema documented at the top of
`gaze_validation.py`. Supply reference **directions**, not screen pixel positions.
Converting a real target position into an eye-camera direction requires measured
camera/target geometry, coordinate transforms and an appropriate eye origin.
References must be obtained independently; copying the estimated normal or eye
center to fabricate a reference would invalidate the evaluation.

Bind the target file to the exact exported recording ID, camera ID, calibration
hash and both rotations. Label examples `data_kind: "synthetic"`; use
`"real_reference"` only for measured data and describe its measurement method.
Every reference must match an exported eye, frame index and timestamp exactly.
The evaluator refuses unknown camera identities, mismatches and missing frames.
Present-but-skipped frames reduce reported coverage; they are never filled in.

```sh
.venv/bin/python gaze_validation.py --records run.jsonl \
  --targets measured_targets.json --output evaluation.json --fit-rotation
```

The optional rotation uses only earlier `calibration` samples, requiring at
least three distinct noncollinear directions per eye. Later, disjoint
`validation` samples alone determine mean, median, 95th-percentile and maximum
angular error, reported together with usable coverage. Omit `--fit-rotation` to
compare raw normals. A fixed rotation is a limited optical-to-reference mapping;
it does not correct drifting centers, lens errors or parallax. The evaluator
does not silently exclude range-failing or drifting outputs to improve scores.
It reports errors for the supplied references and leaves `accuracy_verdict`
unset; target independence and physical identity remain the collector's duty.

Software tests cover exact and rendered checkerboards, degenerate captures,
identity mismatch, strong focal anisotropy, slow drift/spikes/gaps, headless
export with real gate enforcement, strict JSON, exact target alignment and
separate rotation fitting/validation. See the workspace
`EyeTracker_v2_real_video_validation/completion` for the actual run evidence.

The September 16 validation passed **121 tests**, including real installed
pye3d geometry tests. A full replay of the supplied videos completed **1,267
pairs / 2,534 eye records**: 2,304 accepted updates, 230 skips and 2,299 available
geometry outputs. The new focal normalization changed four input decisions
relative to the previous run. Eye-center-y range failures remain widespread;
they were diagnosed, not hidden or forcibly corrected. The temporal monitor
reported 14 right-eye drift frames, with maximum measured reference displacement
4.304 native mm, versus 0.877 mm on the left. These are consistency measurements
with reset gaps and engineering thresholds, not gaze errors. Real camera identity
and target accuracy remain unknown until new reference data is collected.

Research basis: [OpenCV calibration](https://docs.opencv.org/4.12.0/dc/dbb/tutorial_py_calibration.html),
[OpenCV projection equations](https://docs.opencv.org/4.12.0/d9/d0c/group__calib3d.html),
[Zhang's planar calibration method](https://www.microsoft.com/en-us/research/publication/a-flexible-new-technique-for-camera-calibration/),
[Pupil Labs calibration and validation](https://docs.pupil-labs.com/core/software/pupil-capture/#calibration),
and [NIST's proper-rotation derivation](https://doi.org/10.6028/jres.124.028).

### Extending the pipeline

Keep `PupilObservation` and `EyeModelEstimate` as the boundaries between detection,
geometry, and presentation. A replacement detector should return an ellipse in
the same full-frame coordinates and state its confidence meaning. The JSONL
exporter consumes both estimates and `timestamp_s` in `process_frame_loop()`;
retain status and missing values instead of substituting zeros in other formats.

Changing a camera, crop, resolution, or rotation requires checking the entire
coordinate path through calibration and overlays. Adding gaze mapping requires
a defined target coordinate system and additional calibration. Preserve separate
temporal state per eye and make recording alignment explicit when replacing the
current frame-index pairing.

When updating pye3d, verify the adapter against its upstream
[detector implementation](https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/detector_3d.py),
[camera model](https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/camera.py),
and [reference radius](https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/constants.py).
These links follow the upstream branch; check the version actually installed.


## Conservative boundary checks: September 17 development update

`pupil_detection.pupil_boundary_quality()` now checks whether the selected pupil
contour touches the ROI border and whether the fitted pupil's upper boundary
lacks contrast despite supported lower-boundary contrast. It attaches a reason
through `PupilObservation.boundary_rejection`; `pupil_quality.py` enforces it.
Baseline explicitly clears this new evidence, preserving the control condition.
Weights, pupil fits, ROIs, calibration and recovery settings are unchanged.

The complete comparison `20260918T001402_351917` uses minimum/recovery confidence
0.7/0.7 and recovery duration 0.05 seconds. Both modes processed 1,267 paired
frames. Scored against the SAME frozen 282-label snapshot (270 certain labels),
the old filter admitted 35/94 bad frames and discarded 6/176 good frames; the
new filter admitted 34/94 and discarded 7/176. The additional good rejection is
recovery waiting. This is a modest tradeoff, not established overall superiority.
The baseline remains 65/94 bad admissions and 1/176 good rejections. All 162
automated tests passed. The run folder contains the label snapshot and a
`development_comparison.md` report; the regular platform can score this run.

Clipping can still escape this check when segmentation retreats from the crop
boundary. Coverage can be missed when a fitted ellipse follows only the visible
pupil. These are conservative development heuristics, not a validated eyelid or
lash classifier. More development and independent held-out evaluation remain.
Restart the review launcher to reload Python modules after this update. Saved
older runs remain available for comparison; do not compare exports from different
source versions as a single baseline/filter pair.


## Upper-pupil coverage update and clickable error review

The latest full comparison is `20260918T023223_8c0e2e`. A broad upper-pupil
interior brightness check supplements the boundary checks; medians reduce
sensitivity to isolated LED reflections, but lighting gradients remain a possible
confound. The model weights, original pupil fits and recovery settings are unchanged.
At minimum/recovery confidence 0.7/0.7 and recovery hold 0.05 s, the current filter
admits 17/94 unusable frames and rejects 11/176 usable frames in the frozen
282-label development snapshot. The preceding boundary update was 34/94 and
7/176 on the same labels. This is a tradeoff, not held-out validation. Mean
measured recovery delay is 33.3 ms; one recovery interval is censored. See the
full report for denominators and per-eye results. All 164 tests pass.

Scored reports now include a `mistakes` list for each mode. The browser displays
the filtered list below the summary, with human notes and an Open frame button.
No labels are changed by opening a mistake. Changing the selected reviewer/run
or saving labels clears stale displayed results. Restart the launcher and rescore
to see the new UI. Existing reports remain readable; rescore to add the new list.

See BLINK_HANDOFF.md for the exact pending tasks. Collecting fresh paired videos,
agreeing on application-level acceptance criteria, and reproducing installation
on another computer still require team participation. Physical gaze validation
is a separate requirement; current outputs do not establish gaze accuracy.


# Blink timing, image evidence and robot-command gate

Development update, September 24, 2026. No physical robot is connected or enabled.

## Outcome

The default run is `20260924T184324_1ba59c`, using the existing image rejection policy, minimum/recovery confidence 0.7/0.7 and recovery hold 0.05 seconds. Both modes process 1,267 frame pairs. Scoring uses the same 282-label snapshot, revision 406: 94 unusable, 176 usable and 12 uncertain (excluded).

| Version | Bad frames admitted | Good frames discarded | Mean measured recovery delay |
|---|---:|---:|---:|
| Previous filter | 17/94 | 11/176 | 33.3 ms |
| Rejected image-rule prototype | 27/94 | 18/176 | 50.0 ms |
| Installed default with earlier gate | 17/94 | 11/176 | 33.3 ms |

The rejected image-rule prototype is saved as `20260924T183831_720e8a`. It catches some clipped pupils but worsens both error rates overall, so it is NOT the default. Its source fingerprint differs from the final version, which subsequently added policy selection. Labels guided development; these results do not establish independent accuracy.

The installed default preserves labeled counts, not every full-recording decision: left frame 368 and right frames 422 and 1071 change from recovery skips to accepted observations. They were not labeled. Image recovery now completes before geometry is evaluated, and each accepted current frame still passes geometry validation. Review those three frames before concluding these changes are beneficial. Do not change labels merely to match the pipeline.

## Where blink filtering happens

Read and rotate frames → detect pupil and image evidence → image-quality/recovery gate → coordinate correction and geometry check → eligible pye3d updates → model diagnostics/stability → robot eligibility → export/display.

`eye_pipeline.prepare_eye_update()` enforces the ordering. Missing/low-confidence/occluded/recovering observations skip lens correction and model updates. A geometry failure after image recovery invalidates the same frame and restarts recovery through `TemporalQualityTracker.reject_current()`, without submitting a duplicate timestamp. Both eyes are checked before either model is updated. One valid eye can continue learning while the other is rejected; binocular robot eligibility still requires both.

Detection necessarily follows frame acquisition and pupil localization; it precedes geometric eye-model updates and any robot authorization. Model initialization at startup is separate from learning from an individual frame.

## Image checks: default versus experiment

`pupil_detection.existing_boundary_quality()` retains the established default rejection rules. `pupil_boundary_evidence()` measures the experimental raw-border and corroborating contrast features. Both are kept in the existing detection module; no additional runtime files were introduced.

The experiment tests a connected dark corridor between the predicted contour and a nearby crop edge, masks invalid surrounding samples instead of zero-padding or immediately exiting, and treats brightness asymmetry alone as a warning. Brightness-driven rejection requires additional weak upper-boundary evidence. The numerical evidence is exported even under the existing policy, for analysis; experimental diagnostic status is not automatically the learning decision.

`PupilObservation.boundary_rejection` is the selected policy's decision. `boundary_evidence` describes the experimental measurements. `assess_pupil_quality()` enforces the selected rejection. The stricter robot gate independently requires clear evidence. Thus an observation may be useful for model learning without being eligible for a robot command.

The CLI supports `--boundary-policy existing` (default) and `--boundary-policy experimental`. Use explicit input/output paths and new filenames when evaluating the experiment. Both comparison modes must use the same policy, weights, recordings and settings. The normal browser comparison uses the current configured default; do not confuse a prior saved run with a new experiment. Policy is recorded in exported frames. The experimental policy regressed and needs further development before adoption.

## Robot gate contract

`pupil_quality.RobotCommandGate` is a software integration boundary, not a hardware safety system. Recorded-video runs always deny commands; there is no CLI option that turns them into live control, and no motor driver was added. All 5,068 eye records in the final paired run explicitly deny movement.

A future live integration must provide both eyes' current observation/quality/model results, genuine synchronized acquisition timestamps in the gate's monotonic clock domain, and an independently calibrated robot-space target. The caller must verify the camera-to-robot mapping. Passing camera-identity bookkeeping or returning a finite pupil normal is insufficient to establish accurate robot coordinates.

The gate blocks baseline mode, offline inputs, unknown/warning image evidence, rejected/recovering eyes, failed/absent model diagnostics, unstable estimates, unverified camera identities, stale/future/unsynchronized timestamps, invalid targets, and replayed/out-of-order captures. Default age and skew limits are 0.1 and 0.02 seconds; these are engineering starting points requiring hardware-specific validation. Video frame_index/fps timestamps are not live capture timestamps.

Use `dispatch()` immediately before an adapter sends a command, not a previously saved allowed flag. It rechecks eligibility at send time and supplies an expiry based on the older capture. On rejection it invokes `invalidate`; on sender failure it invalidates and re-raises. The adapter must implement what invalidation means for the actual robot. A receiver-side watchdog must expire commands autonomously if frames stop, communication fails, or this process crashes. Hardware stopping/holding behavior, workspace and velocity limits, collision checks and driver implementation remain separate unfinished integration work.

## Tests and inspection

180 tests pass. Added coverage checks exercise crop connectivity versus iris separation, incomplete sampling, brightness warnings, early image/recovery gating, geometry-induced recovery, model/robot ordering, offline/baseline blocking, missing model evidence, synchronization and freshness, replay prevention, send-time rechecks and sender failure. These tests establish software contracts, not physical robot safety.

Restart Blink Review.command, select `20260924T184324_1ba59c`, and score your labels. The mistake table now includes experimental image-evidence status/reason alongside the actual filter reason. Labels and model weights were not changed. Existing results remain available.

Remaining work: improve image evidence without the measured regression, inspect the three unlabeled timing changes, validate on fresh recordings, calibrate gaze/robot coordinates, and implement/test the live camera and robot adapter with its own watchdog. No claim of completed robot control or validated gaze accuracy is made.


# Blink evidence follow-up: September 28, 2026

The existing filter and early image/recovery gate remain the default. All 2,534
input decisions in the September 24 reference were reproduced before testing
separate border and brightness changes. The development ablation results were:

| Input-gate policy | Bad frames admitted | Good frames rejected |
|---|---:|---:|
| Current default | 17/94 | 11/176 |
| Earlier border rule only | 13/94 | 19/176 |
| Earlier brightness relaxation only | 31/94 | 10/176 |
| Earlier combined experiment | 27/94 | 18/176 |
| Border rule plus fitted-ellipse crossing | 16/94 | 16/176 |
| Locally referenced border candidate | 14/94 | 19/176 |

These are full-sequence input-decision replays, including recovery and corrected
geometry checks. They do not represent new pye3d output, runtime benchmarks, or
independent validation. None of these candidates was promoted. In particular,
a fitted ellipse crossing the crop is not a reliable ground-truth clipping label.
The local-border candidate stays outside the runtime code.

## New diagnostic exports

New runs retain `pupil_observation` even when a frame is rejected or recovering.
Its ellipse and eyelid points use `pupil_observation_coordinate_system` equal to
`rotated_raw_frame_pixels`: the image after rotation, before lens correction.
It also retains confidence, the selected `boundary_rejection`, and experimental
boundary evidence. The selected image rejection can explain why recovery began;
the separate `quality.reason` still explains why the current frame was skipped.

This is raw input evidence, not accepted 3D geometry or a robot target. Skipped
frames still have no current model estimate. Nonfinite measurements become JSON
null, and the export never triggers calibration or model updates. Existing
callers of `frame_record()` remain valid because `pupil` is an optional keyword.
Downloaded JSON mistake reports include these observations when available.
Old saved runs have no raw observations and are not reconstructed or backfilled.

Restart Blink Review.command after updating code. Use a newly completed paired
run and Download report for these fields. The 185-test suite covers the new
export contracts and the previous early gate, recovery and offline robot rules.

The remaining research work is unchanged: review left frame 368 and right
frames 422 and 1071, agree acceptance limits, obtain fresh paired videos and
independent labels, freeze settings, and reproduce the evaluation on a teammate's
machine. Stronger border/brightness decisions still need evidence that improves
the measured tradeoff. No adaptive ROI or physical robot integration was added.


## Full real-video validation

New run: `20260928T232232_ea5086`. Both modes completed all 1,267 frame pairs.
All **5,068 accept/skip decisions and quality records match** the corresponding
September 24 reference. All 5,068 records contain the original pupil observation
and deny robot commands. Skipped frames still have no current gaze estimate.
Label revision 406 and both error rates are unchanged. The new default filtered
result is **17/94 bad admissions and 11/176 good rejections**.

This was an export/behavior regression check, not a speed benchmark or proof of
gaze accuracy. The comparison subprocesses inherited single-thread OMP/MKL settings.
Restart the review platform and select this run, then score your current labels
and download the JSON report to inspect the additional input evidence.
