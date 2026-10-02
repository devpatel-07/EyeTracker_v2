# EyeTracker v2

Wearable eye-tracking research: pupil detection, blink/quality filtering, pye3d
eye-model estimation, and an experimental pupil/glint geometric alternative.

## Setup

Create an environment on your own machine; do not copy another person's `.venv`.

On macOS, the blink/glint development environment used Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos.txt
python -B -m unittest test_pupil_detection test_glints -q
```

pye3d may need Eigen 3 and compiler tools when no compatible wheel is available.
OpenCV's GUI setup also needs a Python installation with Tk support.

`requirements.txt` retains main's Python 3.11 dependency profile, including its
Windows-wheel rationale. That environment was not revalidated for these changes.
The review server currently uses Unix `fcntl` locking and the double-click
launcher is macOS-only; native Windows review support is not implemented.

## Run

```sh
python eye_pipeline.py
python blink_validation.py
```

The first command selects paired eye videos and runs the existing pye3d path.
The second opens the local labeling/comparison interface. On macOS you can also
double-click `Blink Review.command`. Keep that terminal open while reviewing.
Saved labels and results are in ignored `blink_validation_data/`; back them up
separately. Current examples use the bundled videos and model checkpoint.

Fixed pupil crops remain default. `--roi-mode tracked` enables an experimental
tracker that did not improve the current recorded comparison. The console
profiler from main remains available through `PROFILE` in `eye_pipeline.py`.

Glint detection and reconstruction are separate offline commands; see
[GLINT_WORKFLOW.md](GLINT_WORKFLOW.md). Detection exports candidate reflections,
not automatically identified LEDs. Reconstruction requires measured LED
positions, explicit correspondence, camera geometry and eye parameters. It
returns an optical axis, not calibrated visual gaze. No live robot driver is
connected, and recorded-video results cannot authorize robot movement.

## Read next

- [Project guide](PROJECT_GUIDE.md): code explanations and historical results.
- [Blink handoff](BLINK_HANDOFF.md): accepted development status and limitations.
- [ROI experiment](ROI_DEVELOPMENT.md): fixed/tracked measurements.
- [Glint workflow](GLINT_WORKFLOW.md): runnable steps and required measurements.

Development tests establish software behavior. Independent gaze accuracy and
generalization to new recordings remain unmeasured. The shared legacy camera
calibration is a fallback, not verified calibration for both physical cameras.
