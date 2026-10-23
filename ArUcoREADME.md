# Track a phone-displayed ArUco marker with your webcam

Runs locally on your computer with Python 3.10–3.12. The phone only displays an image;
the computer's webcam does all detection. No phone app, phone camera, or server is required.

## 1. Install (Mac / Linux)

```bash
python -m pip install -r requirements.txt
```



## 2. Put the marker on your phone

Generate **marker.png**, dictionary **DICT_5X5_100**, ID **0**:

```bash
python tracker.py marker --output marker.png
```

**Measure the width of the black outer square on the actual phone screen.**
Exclude the white margin. 
For example, a 50 mm black square means `--marker-mm 50`.

To generate another marker (IDs 0–99):

```bash
python tracker.py marker --id 0 --output marker.png
```

## 3. Quick preview

For a black square that measures 50 mm on your phone:

```bash
python tracker.py track --marker-mm 50
```

Press **q** to quit or **s** to save an annotated frame.
If the wrong camera opens, add `--camera 1` (or 2).

This preview assumes a 60-degree horizontal field of view and zero lens distortion.
**Its distances and angles are approximate.** You can supply a known horizontal FOV
with `--hfov 70`, but this is not a substitute for calibration.

## 4. Calibrate for useful position measurements

Generate **chessboard.png**: 10 × 7 squares, 9 × 6 internal corners:

```bash
python tracker.py board
```

Print it flat or display it on a rigid flat screen large enough to resolve all corners.
Measure the actual side of one square (not the whole board or its white margin).
For example, with 20 mm squares:

```bash
python tracker.py calibrate --square-mm 20
```

Hold the whole board in view. Press **Space** to capture when corners are found.
Capture at least 15 distinct, sharp views with different tilts, distances, and positions,
including near the image edges. Move between captures; repeated stationary captures
do not provide enough calibration information. Press **c** to save `camera.npz`.

Review the reported RMS error (pixels); smaller is better, but low error alone does not
guarantee a good calibration. Validate distance against a ruler at several positions.
Keep the same camera, lens/zoom/focus settings, and resolution afterward. Autofocus
changes can alter the calibration; lock focus if your webcam supports it.

Then run:

```bash
python tracker.py track --marker-mm 50 --calibration camera.npz --csv tracking.csv
```

The app requests 1280 × 720 and calibrates at the camera's actual delivered resolution.
It refuses to use a calibration at a different resolution. If you change `--width` or
`--height`, pass the same settings to both calibration and tracking. Recalibrate for a
different camera, crop, digital zoom, or field of view.

## What the numbers mean

- Position is the marker center relative to the webcam optical center, in mm.
- Camera X is right, Y is down, Z is forward into the scene. The preview is not mirrored.
- Distance is `sqrt(X² + Y² + Z²)`; Z alone is forward depth.
- The colored axes are marker-local X (red), Y (green), Z (blue).
- Orientation is the marker-to-camera rotation, decomposed as
  `R = Rz(yaw) Ry(pitch) Rx(roll)`, in degrees. A front-facing marker can have a
  roll near ±180° because the marker's Y axis points upward while camera Y points down.
- CSV also includes the Rodrigues rotation vector in radians, reprojection RMS,
  timestamps, calibration status, and explicit missing-detection rows. These rotation
  vector components are not Euler angles. Missing markers do not retain a stale pose.
- CSV files are overwritten when you reuse their filename.

This tracks the **displayed marker**, not the physical center or back of the phone.
The position is camera-relative; it is not world-coordinate tracking or phone IMU fusion.
One planar marker may have orientation ambiguity/jitter when nearly front-facing,
small, blurred, or partly obscured. The solver chooses a positive-depth solution with
the smallest corner reprojection error; that does not remove this geometric ambiguity.
Nothing is tracked while the marker is hidden. For better results use a larger marker,
sharp imagery, a modest tilt, and calibrated intrinsics. Avoid screen reflections,
rolling-shutter motion, and display banding.

## Sources and relationship to the videos

The second linked video is Nicolai Nielsen's “ArUco Marker Pose Estimation in OpenCV”:
https://www.youtube.com/watch?v=GEWoGDdjlSc

Its corresponding source is available here:
https://github.com/niconielsen32/ComputerVision/blob/master/ArUco/arucoPoseEstimation.py

That script uses older ArUco APIs, hard-coded camera intrinsics/distortion, and a fixed
0.02 m marker length. This package is an independently written implementation of the
same OpenCV workflow, rather than a verbatim copy. It uses modern ArucoDetector,
generateImageMarker, solvePnPGeneric with IPPE_SQUARE, and drawFrameAxes, and adds
PNG export, your measured marker size, your webcam calibration, and CSV recording.

The first supplied video (https://www.youtube.com/watch?v=sg1bVJBjbng) could not be
retrieved during preparation; this package does not claim to reproduce its code.

OpenCV references:
https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

## Validation

Both Python files passed compilation checks. OpenCV installation was blocked in the
build environment, so the included runtime tests could not be executed here.
After installing dependencies, run `python -m unittest -v test_tracker.py` for synthetic
marker-detection and known-pose tests. Physical webcam operation, camera permissions,
and accuracy on your phone must be checked on your own computer; no webcam is available
in the build environment.
