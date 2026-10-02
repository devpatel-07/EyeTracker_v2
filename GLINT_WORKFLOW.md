# Glint alternative: executable development workflow

October 1, 2026. This is an **offline alternative under development**, beside
the accepted blink/pye3d pipeline. Running `eye_pipeline.py` behaves as before.
Only two runtime modules are added; no model weights or baseline rules change.

## What is implemented

| File | Responsibility |
|---|---|
| `glint_detection.py` | Reuse pupil detection and image/recovery gating; extract bright-spot candidates; save annotated video, previews, measurements and provenance. |
| `glint_geometry.py` | Read measured geometry and explicit LED matches; estimate corneal center, refracted pupil center and optical direction. |
| `test_glints.py` | Synthetic image, geometry, rejection and file-workflow checks. |

The detector imports NumPy/OpenCV only. The video runner reuses the existing
project environment and modules. No new dependencies are required. Each output
directory must be new; existing runs are never overwritten.

## 1. Verify installation

In Terminal:

```sh
cd /path/to/EyeTracker_v2
.venv/bin/python -B -m unittest test_pupil_detection test_glints -q
```

The geometry tests generate light reflections from known sphere points and
recover the original center and pupil. Passing establishes software behavior
under the assumed model, not real gaze accuracy.

## 2. Run candidate detection now (no LED coordinates required)

Start with 150 consecutive frames:

```sh
.venv/bin/python -B glint_detection.py \
  --video "cam_2026-08-15 17-10-39_lefteye.mp4" --eye left \
  --output-dir glint_runs/left_preview --max-frames 150 \
  --preview-every 25 --write-video
```

For a full run, omit `--max-frames`. For the right eye use its video and
`--eye right`, with a different output directory. The CLI uses the corresponding
fixed ROI, rotation, weights, confidence and recovery settings in `eye_pipeline.py`.
These presets belong to the current recordings; verify them before new captures.

Open `glint_runs/left_preview/overlay.mp4` or its `frame_*.jpg` files in Finder.
Amber `C1`, `C2`, etc. are **candidate numbers in this frame**, not LED identities.
The JSONL file contains their full-frame pixel centers, area, contrast,
saturation warning and rejected-component reasons. `metadata.json` records
settings and input/source hashes. `summary.json` reports completion and timing.
Candidate-extraction timing excludes the much more expensive pupil inference.

### How the detector works

1. Locate the pupil using the existing TinyUNet and broad ROI.
2. Apply the existing image-quality and temporal recovery gate. On rejected or
   recovering frames, return no current glint candidates.
3. Form connected regions at brightness 200; require a seed at least 235.
   Both are adjustable 8-bit starting thresholds, not research-validated values.
4. Reject tiny/large, elongated, sparse or crop-clipped components. Compare
   brightness against nearby background and restrict candidates to a broad
   pupil neighborhood. Glints are allowed outside the pupil itself.
5. Compute a background-subtracted intensity centroid. Keep coordinates in
   rotated full-image pixels; never add the ROI offset twice.
6. Record all surviving candidates. Do not force six outputs, carry old
   measurements forward, or interpret a candidate count as blink/gaze accuracy.

This uses one native connected-component pass and small local array operations.
It does not erode the pupil mask. Saturation, close merged spots, skin/tear-film
reflections and thresholds remain detection limitations; visual review is needed.
Shape checks cannot identify every merged reflection.

The gate here is the shared image/recovery gate, **not pye3d-specific geometry**.
Therefore eligible counts need not equal final pye3d admissions. No 3D learning
or robot dispatch occurs in this runner.

## 3. Supply physical measurements when available

```sh
.venv/bin/python -B glint_geometry.py --write-template glint_left.json
```

Fill each null before reconstruction. Create a separate configuration for the
right camera. Required inputs:

- `calibration_path`, `camera_id`, `calibration_rotation`: correct calibration
  for this physical camera, with image dimensions matching the source video.
- `leds`: unique physical LED IDs and their measured `[x,y,z]` in **millimeters
  relative to the raw calibration camera**. Origin is camera center; X right,
  Y down, Z along the camera's viewing direction. CAD coordinates need a measured
  rigid transform into this frame. A distance from the eye alone is insufficient.
- `corneal_radius_mm`: radius of the cornea, NOT the existing 12 mm eyeball radius.
- `pupil_offset_mm`: distance K from corneal curvature center to physical pupil.
- `refractive_index`: effective refractive index for the eye model.
- `corneal_distance_bounds_mm`: physically plausible camera-to-cornea-center
  DISTANCE bounds, not simply Z-coordinate bounds. Require K < R < lower < upper.

Document whether eye parameters are measured, fitted from calibration, or
assumed. Placeholders intentionally fail instead of producing invented output.
The default requires a matching camera ID in the calibration file. The optional
`--allow-unverified-calibration` permits an explicitly exploratory legacy run;
it neither fixes an incorrect calibration nor verifies accuracy.

## 4. Establish LED correspondence

Coordinates alone do not tell the program which bright spot belongs to which
LED. Ideally collect a short controlled sequence illuminating one LED at a time
to establish identities; verify the hardware/exposure behavior with the team.
For the prototype, annotate selected frames explicitly:

```sh
.venv/bin/python -B glint_geometry.py \
  --detections glint_runs/left_preview/detections.jsonl \
  --make-matches glint_left_matches.json
```

Edit its `frames` entries, keeping the generated detection hash:

```json
"100": {"1": "LED2", "2": "LED4", "3": null}
```

This example means candidate C1 on frame 100 was identified as LED2, C2 as
LED4, and C3 remains unknown. **It is only a format example, not a mapping for
the supplied recording.** Frame numbers are zero-based. Leave unverified
candidates null. At least two correct, geometrically informative matches are
needed by this solver. More matched LEDs provide additional consistency checks.
Candidate indices may change every frame; copying a mapping across frames is
not a valid tracking strategy. Automatic correspondence is future work.

## 5. Reconstruct after steps 3 and 4

```sh
.venv/bin/python -B glint_geometry.py \
  --detections glint_runs/left_preview/detections.jsonl \
  --config glint_left.json --matches glint_left_matches.json \
  --output-dir glint_runs/left_geometry
```

The solver undoes image rotation, corrects lens distortion and creates camera
rays. LED/glint planes determine a center direction via SVD; a bounded scalar
search fits distance using reflection constraints. It then intersects the pupil
ray with the cornea, applies Snell refraction and uses K to recover pupil depth.
The normalized vector `pupil_center - corneal_center` is the outward optical axis.

`optical_axes.jsonl` contains results or explicit rejection reasons per frame.
Degenerate geometry, impossible intersections, distance-bound solutions and
large reflection residuals fail. Unmatched or blink-rejected frames produce no
axis. The 2-degree maximum reflection residual is an engineering starting gate,
not a gaze error bound. No robust automatic outlier removal is implemented.

For noisy inputs, the SVD direction followed by scalar fitting is an approximate
staged estimator, not a joint maximum-likelihood fit of every measurement.
Sensitivity to assumed radius, K, correspondence and centroid bias must be
measured. The fitted ellipse center is used as the pupil image-center estimate;
perspective/refraction can bias that approximation.

## 6. Validate and compare before demo integration

- Review known glint centers/identities and report detection misses and false
  candidates separately from gaze accuracy. Candidate count alone is not accuracy.
- Collect known targets separately for personal calibration and evaluation.
- Compare both methods using the same recordings and pupil observations. Include
  all failures in availability rates; repeat timing runs under matching conditions.
- This module returns **raw-calibration-camera** optical directions. Existing
  pye3d output uses the **processed-camera** frame. Rotate into a common frame
  before comparing vectors; use calibrated visual directions for target errors.
- Future pipeline integration should split the shared image/recovery gate from
  pye3d-specific checks, then select an estimator. Preserve pye3d as default.
- Personal optical-to-visual calibration, binocular/robot transforms, target
  depth/surface, method-specific reliability and live hardware dispatch remain
  separate work. Never feed these offline optical-axis files directly to servos.

## Research basis and implementation choices

[Guestrin & Eizenman (2006), Section II-B](https://vemlab.github.io/127.0.0.1/static/attachment/36.pdf)
provides the one-camera/multiple-light reflection and refraction constraints.
It requires camera geometry and eye parameters; this is not calibration-free
visual gaze. The paper's multiple-camera case observes the same eye.
The threshold/shape heuristics and bounded numerical implementation here are
our prototype choices, not a reproduction of a published detector benchmark.
[OpenCV connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html)
supplies component statistics; [OpenCV camera calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
supplies undistortion.

Commit the two runtime modules, tests and this guide with your code. Keep local
recordings, generated runs and personal calibration files out of a broad upload.
The provided runner does not modify `.gitignore`; add `glint_runs/` to your local
excludes or reviewed project ignore rules before staging outputs.
