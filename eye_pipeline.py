"""Run paired eye videos through 2D pupil detection and per-eye 3D estimation.

Start here when changing the workflow or its presets. ``video_preparation``
handles file selection/orientation; ``pupil_detection`` returns an ellipse in
the rotated full frame; ``calibration`` corrects that ellipse for lens
distortion; ``eye_model_estimation`` updates a temporal model for each eye;
``feature_output`` draws the observations and model projections.

The recordings are paired by frame index, so their first frames must already
represent the same instant. Each eye keeps its own camera/model coordinates;
this pipeline does not transform them into a shared binocular reference frame.
"""

from pathlib import Path
from dataclasses import replace, asdict
import argparse
import hashlib
import json
import warnings
import math
import time

import cv2
from debug_tools.profiler import Profiler


from calibration import CameraCalibration
from eye_model_estimation import EyeModelEstimator, ModelStabilityMonitor
from feature_output import create_output_frame, print_frame_features, frame_record
from pupil_roi import PupilRoiTracker
from pupil_quality import RobotCommandGate
from pupil_detection import (
    TemporalQualityTracker,
    FrameQualityDecision,
    assess_pupil_quality,
    load_pupil_detector,
)
from video_preparation import (
    matching_video_fps,
    open_video,
    read_first_frame,
    reset_video,
    rotate_frame,
    rotated_video_dimensions,
    select_eye_video,
    setup_rotation_gui,
    video_dimensions,
)


# Resolve bundled assets beside this script, independent of the working folder.
HERE = Path(__file__).resolve().parent

# Retain main's optional console profiler alongside the exported stage timings.
PROFILE = True
profiler = Profiler(PROFILE)


# Use a path to bypass that eye's file chooser; None opens the chooser at startup.

LEFT_VIDEO_PATH = None
RIGHT_VIDEO_PATH = None


# ROIs are (x, y, width, height) in pixels AFTER the full frame is rotated.
# FRAME_ROTATION turns decoded video frames. CALIBRATION_ROTATION describes
# the rotation from the calibration image orientation to the source video;
# CameraCalibration adds both rotations to map calibration into processed space.
# The setup window changes these rotations for this run only, not the file.

LEFT_ROI = (0, 100, 1080, 698)
LEFT_FRAME_ROTATION = "counterclockwise"
LEFT_CALIBRATION_ROTATION = "none"


# Right-eye presets follow the same coordinate convention as the left eye.

RIGHT_ROI = (0, 100, 1080, 698)
RIGHT_FRAME_ROTATION = "clockwise"
RIGHT_CALIBRATION_ROTATION = "none"


# Prefer separately calibrated cameras, retaining the legacy file for exploration.
# A declared/matched camera ID is bookkeeping, not proof of physical calibration.
# Calibration dimensions after CALIBRATION_ROTATION must match each source
# video; intrinsics are not rescaled.

LEFT_CALIBRATION_PATH = HERE / ("left_camera_calibration.npz"
    if (HERE / "left_camera_calibration.npz").exists() else "camera_calibration.npz")
RIGHT_CALIBRATION_PATH = HERE / ("right_camera_calibration.npz"
    if (HERE / "right_camera_calibration.npz").exists() else "camera_calibration.npz")
LEFT_CAMERA_ID = "left-eye-camera"
RIGHT_CAMERA_ID = "right-eye-camera"
REQUIRE_CALIBRATION_IDENTITY = False


# The checkpoint supplies the network's channel count and input resolution.
# Keep preprocessing in pupil_detection.py consistent with the model's training.

MODEL_PATH = HERE / "models" / "finetuned_2026-08-25" / "pupil_unet_best.pt"
DEVICE = "auto"  # Selects CUDA if available, otherwise CPU (not automatic MPS).
MASK_THRESHOLD = 0.5  # Per-pixel probability cutoff for the segmentation mask.
BOUNDARY_POLICY = "existing"  # New border/corroboration rules remain an opt-in experiment.
ROI_MODE = "fixed"  # Accepted mode; tracked remains an explicit development experiment.
MIN_CONFIDENCE = 0.70  # Observation-quality gate for updating the 3D model.
RECOVERY_CONFIDENCE = 0.70  # Stronger pupil evidence needed after a rejection.
RECOVERY_DURATION_S = 0.05  # Consecutive strong samples must span this video time.
EYE_RADIUS_MM = 12.0  # Assumed radius used to scale reported 3D lengths.


# Display/text controls; timing below does not change the video timestamps.

TEXT_OUTPUT = False  # Print one feature record per eye per processed frame pair.
# OpenCV key/event wait adds to processing time; it does not sync to source FPS.
WAIT_MS = 30
MAX_FRAMES = 0  # A positive value limits frame pairs; zero/negative runs to EOF.
HEADLESS = False  # Uses the configured rotations without opening setup/windows.
RESULTS_PATH = None  # Optional JSONL file; refuses to overwrite an existing run.
FILTER_MODE = "filtered"  # baseline keeps pupil/geometry checks, bypasses eyelids/recovery.
RUN_SUMMARY_PATH = None
PROGRESS_EVERY = 0


def _sha256(path):
    """Bind validation records to exact input bytes, without loading a video into RAM."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _source_fingerprint():
    """Identify the exact implementation used by a controlled comparison."""
    names = ("eye_pipeline.py", "pupil_detection.py", "pupil_quality.py", "pupil_roi.py", "calibration.py",
             "eye_model_estimation.py", "feature_output.py", "video_preparation.py")
    sources = {name: _sha256(HERE / name) for name in names}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def _validate_filter_settings():
    """Reject invalid experiments before opening videos or creating output files."""
    if ROI_MODE not in {"fixed", "tracked"}:
        raise ValueError("ROI mode must be fixed or tracked")
    if FILTER_MODE not in {"baseline", "filtered"}:
        raise ValueError("filter mode must be baseline or filtered")
    if not all(math.isfinite(value) for value in
               (MIN_CONFIDENCE, RECOVERY_CONFIDENCE, RECOVERY_DURATION_S)):
        raise ValueError("filter settings must be finite")
    if not 0 < MIN_CONFIDENCE <= RECOVERY_CONFIDENCE <= 1:
        raise ValueError("require 0 < minimum confidence <= recovery confidence <= 1")
    if RECOVERY_DURATION_S < 0:
        raise ValueError("recovery duration must be nonnegative")
    if MAX_FRAMES < 0 or PROGRESS_EVERY < 0:
        raise ValueError("frame limit and progress interval must be nonnegative")


def _filter_observation(pupil):
    """Baseline hides eyelid evidence only; original detection stays unchanged."""
    return replace(pupil, eyelid=None, boundary_rejection=None, boundary_evidence=None) if FILTER_MODE == "baseline" else pupil


def _quality_for_mode(pupil, timestamp_s, geometry_rejection, tracker):
    """Keep basic protections in both modes; give only filtered mode memory."""
    if FILTER_MODE == "filtered":
        return tracker.update(pupil, timestamp_s,
                              additional_rejection_reason=geometry_rejection)
    instant = assess_pupil_quality(_filter_observation(pupil), MIN_CONFIDENCE)
    if not instant.allow_model_update:
        return replace(instant, state="rejected")
    if geometry_rejection is not None:
        return FrameQualityDecision(False, geometry_rejection, "rejected")
    return FrameQualityDecision(True, "baseline pupil and geometry checks passed", "usable")


def run_pipeline():
    """Select inputs, validate camera geometry, and run until EOF or ``q``.

    One shared segmentation network serves both eyes, but separate estimators
    preserve each eye's temporal model. Captures and display windows are cleaned
    up on normal exit, chooser cancellation, and exceptions during setup/run.
    """
    _validate_filter_settings()
    started = time.perf_counter()
    if RUN_SUMMARY_PATH is not None and Path(RUN_SUMMARY_PATH).exists():
        raise FileExistsError(f"run summary already exists: {RUN_SUMMARY_PATH}")
    left_video = None
    right_video = None
    results = None
    try:
        if HEADLESS and (LEFT_VIDEO_PATH is None or RIGHT_VIDEO_PATH is None):
            raise ValueError("headless mode requires both video paths")
        left_path = select_eye_video(LEFT_VIDEO_PATH, "left")
        if left_path is None:
            return
        right_path = select_eye_video(RIGHT_VIDEO_PATH, "right")
        if right_path is None:
            return

        left_video = open_video(left_path, "left")
        right_video = open_video(right_path, "right")

        left_frame_rotation, right_frame_rotation = LEFT_FRAME_ROTATION, RIGHT_FRAME_ROTATION
        left_calibration_rotation = LEFT_CALIBRATION_ROTATION
        right_calibration_rotation = RIGHT_CALIBRATION_ROTATION
        if not HEADLESS:
            left_preview = read_first_frame(left_video, "left")
            right_preview = read_first_frame(right_video, "right")
            (left_frame_rotation, right_frame_rotation,
             left_calibration_rotation, right_calibration_rotation) = setup_rotation_gui(
                left_preview, right_preview, LEFT_ROI, RIGHT_ROI,
                LEFT_FRAME_ROTATION, RIGHT_FRAME_ROTATION,
                LEFT_CALIBRATION_ROTATION, RIGHT_CALIBRATION_ROTATION,
            )
            # Reading previews consumed a frame; rewind before timestamp zero.
            reset_video(left_video, "left")
            reset_video(right_video, "right")

        left_source_size = video_dimensions(left_video)
        right_source_size = video_dimensions(right_video)
        left_size = rotated_video_dimensions(
            left_source_size,
            left_frame_rotation,
        )
        right_size = rotated_video_dimensions(
            right_source_size,
            right_frame_rotation,
        )

        left_calibration = CameraCalibration.load(
            LEFT_CALIBRATION_PATH,
            left_calibration_rotation,
            left_frame_rotation,
            expected_camera_id=LEFT_CAMERA_ID,
            require_identity=REQUIRE_CALIBRATION_IDENTITY,
        )
        right_calibration = CameraCalibration.load(
            RIGHT_CALIBRATION_PATH,
            right_calibration_rotation,
            right_frame_rotation,
            expected_camera_id=RIGHT_CAMERA_ID,
            require_identity=REQUIRE_CALIBRATION_IDENTITY,
        )
        left_calibration.validate_video_dimensions(
            left_source_size,
            left_size,
            "left",
        )
        right_calibration.validate_video_dimensions(
            right_source_size,
            right_size,
            "right",
        )
        for side, calibration in (("left", left_calibration), ("right", right_calibration)):
            if calibration.identity_status == "unverified":
                warnings.warn(f"{side} calibration has no camera identity; geometry is exploratory",
                              RuntimeWarning, stacklevel=2)

        # Matching FPS supports index pairing but cannot detect start offsets.
        fps = matching_video_fps(left_video, right_video)
        expected_pairs = None
        if RUN_SUMMARY_PATH is not None:
            counts = [int(video.get(cv2.CAP_PROP_FRAME_COUNT)) for video in (left_video, right_video)]
            expected_pairs = min(counts) if all(count > 0 for count in counts) else None

        metadata = None
        if RESULTS_PATH is not None:
            video_hashes = {"left": _sha256(left_path), "right": _sha256(right_path)}
            model_hash = _sha256(MODEL_PATH)
            recording_id = hashlib.sha256(json.dumps(video_hashes, sort_keys=True).encode()).hexdigest()
            metadata = {}
            for side, calibration, path, frame_rotation, calibration_rotation in (
                ("left", left_calibration, LEFT_CALIBRATION_PATH,
                 left_frame_rotation, left_calibration_rotation),
                ("right", right_calibration, RIGHT_CALIBRATION_PATH,
                 right_frame_rotation, right_calibration_rotation),
            ):
                metadata[side] = dict(
                    schema_version=1, recording_id=recording_id, camera_id=calibration.camera_id,
                    coordinate_system="processed_eye_camera", video_sha256=video_hashes[side],
                    calibration_sha256=_sha256(path), frame_rotation=frame_rotation,
                    calibration_rotation=calibration_rotation,
                    camera_matrix=calibration.video_camera_matrix.tolist(),
                    calibration_identity_status=calibration.identity_status,
                    calibration_source=calibration.calibration_source,
                    direction_kind="pupil_normal_uncalibrated_visual_axis",
                    fps=fps, model_sha256=model_hash,
                    roi=list(LEFT_ROI if side == "left" else RIGHT_ROI),
                    eye_radius_mm=EYE_RADIUS_MM, focal_normalization="exact_affine",
                    min_confidence=MIN_CONFIDENCE, recovery_confidence=RECOVERY_CONFIDENCE,
                    recovery_duration_s=RECOVERY_DURATION_S,
                    filter_mode=FILTER_MODE, mask_threshold=MASK_THRESHOLD, device=DEVICE,
                    source_fingerprint=_source_fingerprint(),
                )
            # Exclusive creation protects earlier validation evidence. On an
            # interrupted run this remains an explicitly partial frame sequence.
            results = Path(RESULTS_PATH).open("x", encoding="utf-8", buffering=1)

        pupil_detector = load_pupil_detector(
            MODEL_PATH,
            device=DEVICE,
            mask_threshold=MASK_THRESHOLD,
            boundary_policy=BOUNDARY_POLICY,
        )
        left_eye_model = EyeModelEstimator(
            left_calibration,
            min_confidence=MIN_CONFIDENCE,
            eye_radius_mm=EYE_RADIUS_MM,
        )
        right_eye_model = EyeModelEstimator(
            right_calibration,
            min_confidence=MIN_CONFIDENCE,
            eye_radius_mm=EYE_RADIUS_MM,
        )

        processing_started = time.perf_counter()
        processed = process_frame_loop(
            left_video,
            right_video,
            fps,
            left_frame_rotation,
            right_frame_rotation,
            left_calibration,
            right_calibration,
            pupil_detector,
            left_eye_model,
            right_eye_model,
            headless=HEADLESS,
            record_stream=results,
            record_metadata=metadata,
        )
        if processed == 0:
            raise ValueError("no paired video frames could be decoded; check both recordings")
        if results is not None:
            print(f"Saved {2 * processed} eye/frame records to {RESULTS_PATH}", flush=True)
        if RUN_SUMMARY_PATH is not None:
            expected_stop = (min(MAX_FRAMES, expected_pairs) if MAX_FRAMES > 0
                             and expected_pairs is not None else
                             MAX_FRAMES if MAX_FRAMES > 0 else expected_pairs)
            status = ("complete" if expected_stop is not None and processed == expected_stop
                      else "length_unknown" if expected_stop is None else "early_stop")
            summary = dict(schema_version=1, filter_mode=FILTER_MODE, status=status,
                           completed_frame_pairs=processed, expected_frame_pairs=expected_pairs,
                           max_frames=MAX_FRAMES, elapsed_seconds=time.perf_counter()-started,
                           processing_seconds=time.perf_counter()-processing_started,
                           source_fingerprint=_source_fingerprint(), fps=fps)
            # Exclusive output and a complete JSON object; a failed run has no
            # success summary, so the comparison app cannot mistake it for one.
            with Path(RUN_SUMMARY_PATH).open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, allow_nan=False)
    finally:
        if results is not None:
            results.close()
        if left_video is not None:
            left_video.release()
        if right_video is not None:
            right_video.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


def _prepare_corrected_model_input(pupil, calibration, eye_model):
    """Correct and check geometry only after the image-level pupil checks pass.

    Returns ``(corrected_ellipse, later_rejection_reason)``. The temporal
    tracker consumes the optional reason so a post-calibration failure changes
    the real gate state and recovery history. A normal image-level rejection
    returns ``(None, None)`` without running lens correction.
    """
    instant = assess_pupil_quality(_filter_observation(pupil), MIN_CONFIDENCE)
    if not instant.allow_model_update:
        return None, None
    corrected_ellipse = calibration.undistort_ellipse(pupil.ellipse)
    geometry = eye_model.assess_input_geometry(corrected_ellipse)
    rejection_reason = None if geometry.allow_model_update else geometry.reason
    return corrected_ellipse, rejection_reason


def prepare_eye_update(pupil, timestamp_s, calibration, model, tracker):
    """Enforce image/recovery checks BEFORE correction, then validate geometry.

    A recovery frame does not run calibration or update pye3d. Geometry failures
    invalidate this same frame and restart recovery without a duplicate timestamp.
    The caller must still check the returned decision before model.update().
    """
    quality = _quality_for_mode(pupil, timestamp_s, None, tracker)
    if not quality.allow_model_update:
        return None, quality
    ellipse, rejection = _prepare_corrected_model_input(pupil, calibration, model)
    if rejection is not None:
        quality = (tracker.reject_current(rejection) if FILTER_MODE == "filtered"
                   else FrameQualityDecision(False, rejection, state="rejected"))
    return ellipse, quality


def process_frame_loop(
    left_video,
    right_video,
    fps,
    left_frame_rotation,
    right_frame_rotation,
    left_calibration,
    right_calibration,
    pupil_detector,
    left_eye_model,
    right_eye_model,
    *,
    headless=False,
    record_stream=None,
    record_metadata=None,
):
    """Process consecutive frame pairs, interleaving each stage for both eyes.

    Work is sequential, not parallel. Stop when either recording ends/fails to
    decode, MAX_FRAMES is reached, or ``q`` is pressed in an OpenCV window.
    Both eyes receive the same recording-time timestamp, regardless of latency.
    """
    # Track quality history independently, even though both eyes share TinyUNet.
    # Create new trackers for each run. A supplied timestamp gap over 1.5 frame
    # periods restarts recovery. Our frame_index / fps timestamps assume uniform
    # sampling; they cannot reveal frames already missing from the recording.
    left_quality_tracker = TemporalQualityTracker(
        MIN_CONFIDENCE, RECOVERY_CONFIDENCE, RECOVERY_DURATION_S, max_gap_s=1.5 / fps,
    )
    right_quality_tracker = TemporalQualityTracker(
        MIN_CONFIDENCE, RECOVERY_CONFIDENCE, RECOVERY_DURATION_S, max_gap_s=1.5 / fps,
    )
    # Output diagnostics are independent of the input gate: a drifting model
    # still receives eligible observations. Both monitors see skipped frames.
    left_stability, right_stability = ModelStabilityMonitor(), ModelStabilityMonitor()
    robot_gate = RobotCommandGate()  # Offline recordings can NEVER authorize movement.
    roi_trackers = {side: PupilRoiTracker(roi, MIN_CONFIDENCE, RECOVERY_DURATION_S)
                    for side, roi in (("left", LEFT_ROI), ("right", RIGHT_ROI))} if ROI_MODE == "tracked" else {}
    frame_index = 0
    while MAX_FRAMES <= 0 or frame_index < MAX_FRAMES:
        frame_started = time.perf_counter()
        left_ok, left_frame = left_video.read()
        right_ok, right_frame = right_video.read()
        if not left_ok or not right_ok:
            break

        profiler.start()

        left_frame = rotate_frame(left_frame, left_frame_rotation)
        right_frame = rotate_frame(right_frame, right_frame_rotation)

        # Estimate recording time at a constant FPS, independent of processing
        # speed. This does not read the video's individual presentation times.
        timestamp_s = frame_index / fps

        detection_started = time.perf_counter()
        profiler.clear()
        roi_info = {}
        if ROI_MODE == "tracked":
            left_pupil, roi_info["left"] = roi_trackers["left"].detect(pupil_detector, left_frame, timestamp_s)
            right_pupil, roi_info["right"] = roi_trackers["right"].detect(pupil_detector, right_frame, timestamp_s)
        else:
            left_pupil = pupil_detector.detect(left_frame, LEFT_ROI)
            right_pupil = pupil_detector.detect(right_frame, RIGHT_ROI)
            roi_info = {side: dict(mode="fixed", detection_roi=list(roi), search_roi=list(roi),
                                  requested_roi=list(roi), fallback_reason=None,
                                  network_calls=1, area_fraction=1.)
                        for side, roi in (("left", LEFT_ROI), ("right", RIGHT_ROI))}
        detection_done = time.perf_counter()
        profiler.checkpoint("Pupil Detection")

        # Image rejection and recovery are resolved before any lens correction.
        left_corrected_ellipse, left_quality = prepare_eye_update(
            left_pupil, timestamp_s, left_calibration, left_eye_model, left_quality_tracker)
        right_corrected_ellipse, right_quality = prepare_eye_update(
            right_pupil, timestamp_s, right_calibration, right_eye_model, right_quality_tracker)

        quality_done = time.perf_counter()
        profiler.checkpoint("Quality and Pupil Undistortion")
        # Enforce the final decision before pye3d. A rejected or recovering
        # frame returns an explicit empty estimate and preserves model history.
        if left_quality.allow_model_update:
            left_estimate = left_eye_model.update(
                left_corrected_ellipse,
                left_pupil.confidence,
                timestamp_s,
                left_frame,
            )
        else:
            left_estimate = left_eye_model.skip_update(
                left_pupil.confidence,
                left_quality.state,
                left_quality.reason,
            )

        if right_quality.allow_model_update:
            right_estimate = right_eye_model.update(
                right_corrected_ellipse,
                right_pupil.confidence,
                timestamp_s,
                right_frame,
            )
        else:
            right_estimate = right_eye_model.skip_update(
                right_pupil.confidence,
                right_quality.state,
                right_quality.reason,
            )

        for side, estimate, quality, monitor in (
            ("left", left_estimate, left_quality, left_stability),
            ("right", right_estimate, right_quality, right_stability),
        ):
            diagnostics = estimate.model_diagnostics
            center = (diagnostics.native_eye_center_mm
                      if estimate.ready and diagnostics is not None else None)
            estimate = replace(estimate, model_stability=monitor.update(center, timestamp_s))
            if side == "left":
                left_estimate = estimate
            else:
                right_estimate = estimate
        # This decision is downstream of BOTH image gates and model diagnostics.
        # There is intentionally no hardware sender or live-mode switch here.
        # A future live adapter must use RobotCommandGate.dispatch with actual
        # synchronized capture times, calibrated target coordinates and watchdog.
        robot_decision = robot_gate.evaluate(
            ((left_pupil, left_quality, left_estimate),
             (right_pupil, right_quality, right_estimate)), filter_mode=FILTER_MODE)
        # Pair timing includes broad retries and diagnostics but excludes the
        # following export/display work. Both eye records describe the SAME
        # pair duration; do not add them together in a latency report.
        model_done = time.perf_counter()
        profiler.checkpoint("Eye Model Update and Diagnostics")
        timing = dict(decode_rotate_ms=(detection_started-frame_started)*1000,
                      detection_ms=(detection_done-detection_started)*1000,
                      quality_geometry_ms=(quality_done-detection_done)*1000,
                      model_diagnostics_ms=(model_done-quality_done)*1000,
                      total_before_output_ms=(model_done-frame_started)*1000)
        if record_stream is not None:
            for side, pupil, estimate, quality in (
                ("left", left_pupil, left_estimate, left_quality),
                ("right", right_pupil, right_estimate, right_quality),
            ):
                record = frame_record(side, frame_index, timestamp_s, estimate,
                                      quality, (record_metadata or {}).get(side, {}), pupil=pupil)
                record["roi_mode"] = ROI_MODE
                record["roi_tracking"] = roi_info[side]
                record["frame_pair_timing_ms"] = timing
                record["boundary_policy"] = BOUNDARY_POLICY
                record["boundary_evidence"] = (asdict(pupil.boundary_evidence)
                                                if pupil.boundary_evidence is not None else None)
                record["robot_command"] = asdict(robot_decision)
                record_stream.write(json.dumps(record, allow_nan=False) + "\n")

        if TEXT_OUTPUT:
            print_frame_features("left", timestamp_s, left_estimate, left_quality)
            print_frame_features("right", timestamp_s, right_estimate, right_quality)
        frame_index += 1
        if PROGRESS_EVERY and frame_index % PROGRESS_EVERY == 0:
            print(f"Processed {frame_index} frame pairs ({FILTER_MODE})", flush=True)
        if headless:
            profiler.end()
            continue

        left_output = create_output_frame(
            left_frame,
            "Left eye",
            LEFT_ROI,
            left_pupil,
            left_estimate,
            quality_decision=left_quality,
        )
        right_output = create_output_frame(
            right_frame,
            "Right eye",
            RIGHT_ROI,
            right_pupil,
            right_estimate,
            quality_decision=right_quality,
        )
        profiler.checkpoint("Visual Output")

        # Allow window resizing without stretching the image's aspect ratio.
        cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # waitKey both services window events and checks the quit key.
        profiler.clear()
        cv2.imshow("Left Eye", left_output)
        cv2.imshow("Right Eye", right_output)
        quit_requested = cv2.waitKey(WAIT_MS) & 0xFF == ord("q")
        profiler.checkpoint("Window Display")
        profiler.end()
        if quit_requested:
            break
    return frame_index


def main():
    """Optional command-line controls; no arguments retain the interactive setup."""
    global BOUNDARY_POLICY, ROI_MODE
    global LEFT_VIDEO_PATH, RIGHT_VIDEO_PATH, HEADLESS, RESULTS_PATH, MAX_FRAMES, DEVICE
    global LEFT_CALIBRATION_PATH, RIGHT_CALIBRATION_PATH, LEFT_CAMERA_ID, RIGHT_CAMERA_ID
    global REQUIRE_CALIBRATION_IDENTITY
    global FILTER_MODE, MIN_CONFIDENCE, RECOVERY_CONFIDENCE, RECOVERY_DURATION_S
    global RUN_SUMMARY_PATH, PROGRESS_EVERY
    global LEFT_FRAME_ROTATION, RIGHT_FRAME_ROTATION, LEFT_CALIBRATION_ROTATION, RIGHT_CALIBRATION_ROTATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-video", type=Path, default=LEFT_VIDEO_PATH)
    parser.add_argument("--right-video", type=Path, default=RIGHT_VIDEO_PATH)
    parser.add_argument("--left-calibration", type=Path, default=LEFT_CALIBRATION_PATH)
    parser.add_argument("--right-calibration", type=Path, default=RIGHT_CALIBRATION_PATH)
    parser.add_argument("--left-camera-id", default=LEFT_CAMERA_ID)
    parser.add_argument("--right-camera-id", default=RIGHT_CAMERA_ID)
    parser.add_argument("--require-calibration-identity", action="store_true", default=REQUIRE_CALIBRATION_IDENTITY)
    parser.add_argument("--headless", action="store_true", default=HEADLESS)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH, help="new JSONL path; existing files are protected")
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--filter-mode", choices=("filtered", "baseline"), default=FILTER_MODE)
    parser.add_argument("--roi-mode", choices=("fixed", "tracked"), default=ROI_MODE)
    parser.add_argument("--boundary-policy", choices=("existing", "experimental"), default=BOUNDARY_POLICY)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--recovery-confidence", type=float, default=RECOVERY_CONFIDENCE)
    parser.add_argument("--recovery-duration", type=float, default=RECOVERY_DURATION_S)
    parser.add_argument("--run-summary", type=Path, default=RUN_SUMMARY_PATH)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    rotations = ("none", "clockwise", "180", "counterclockwise")
    parser.add_argument("--left-frame-rotation", choices=rotations, default=LEFT_FRAME_ROTATION)
    parser.add_argument("--right-frame-rotation", choices=rotations, default=RIGHT_FRAME_ROTATION)
    parser.add_argument("--left-calibration-rotation", choices=rotations, default=LEFT_CALIBRATION_ROTATION)
    parser.add_argument("--right-calibration-rotation", choices=rotations, default=RIGHT_CALIBRATION_ROTATION)
    args = parser.parse_args()
    BOUNDARY_POLICY, ROI_MODE = args.boundary_policy, args.roi_mode
    LEFT_VIDEO_PATH, RIGHT_VIDEO_PATH = args.left_video, args.right_video
    LEFT_CALIBRATION_PATH, RIGHT_CALIBRATION_PATH = args.left_calibration, args.right_calibration
    LEFT_CAMERA_ID, RIGHT_CAMERA_ID = args.left_camera_id, args.right_camera_id
    REQUIRE_CALIBRATION_IDENTITY = args.require_calibration_identity
    HEADLESS, RESULTS_PATH, MAX_FRAMES, DEVICE = args.headless, args.output, args.max_frames, args.device
    FILTER_MODE, MIN_CONFIDENCE = args.filter_mode, args.min_confidence
    RECOVERY_CONFIDENCE, RECOVERY_DURATION_S = args.recovery_confidence, args.recovery_duration
    RUN_SUMMARY_PATH, PROGRESS_EVERY = args.run_summary, args.progress_every
    LEFT_FRAME_ROTATION, RIGHT_FRAME_ROTATION = args.left_frame_rotation, args.right_frame_rotation
    LEFT_CALIBRATION_ROTATION, RIGHT_CALIBRATION_ROTATION = args.left_calibration_rotation, args.right_calibration_rotation
    try:
        _validate_filter_settings()
    except ValueError as error:
        parser.error(str(error))
    run_pipeline()


if __name__ == "__main__":
    main()
