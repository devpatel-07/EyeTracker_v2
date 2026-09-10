# Import dependencies

from pathlib import Path

import cv2


# Import Scripts

from calibration import CameraCalibration
from eye_model_estimation import EyeModelEstimator
from feature_output import create_output_frame, print_frame_features
from pupil_detection import load_pupil_detector
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


HERE = Path(__file__).resolve().parent


# Left and right video path preset variables

LEFT_VIDEO_PATH = None
RIGHT_VIDEO_PATH = None


# Left eye ROI region, ROI rotation, and calibration rotation preset variables

LEFT_ROI = (0, 100, 1080, 698)
LEFT_FRAME_ROTATION = "counterclockwise"
LEFT_CALIBRATION_ROTATION = "none"


# Right eye ROI region, ROI rotation, and calibration rotation preset variables

RIGHT_ROI = (0, 100, 1080, 698)
RIGHT_FRAME_ROTATION = "clockwise"
RIGHT_CALIBRATION_ROTATION = "none"


# Camera calibration file preset variables

LEFT_CALIBRATION_PATH = HERE / "camera_calibration.npz"
RIGHT_CALIBRATION_PATH = HERE / "camera_calibration.npz"


# Pupil detection and pye3d preset variables

MODEL_PATH = HERE / "models" / "finetuned_2026-08-25" / "pupil_unet_best.pt"
DEVICE = "auto"
MASK_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.60
EYE_RADIUS_MM = 12.0


# Output preset variables

TEXT_OUTPUT = False
WAIT_MS = 30
MAX_FRAMES = 0


# Organizes run order for one time actions before Frame Loop - function

def run_pipeline():
    left_video = None
    right_video = None
    try:
        left_path = select_eye_video(LEFT_VIDEO_PATH, "left")
        if left_path is None:
            return
        right_path = select_eye_video(RIGHT_VIDEO_PATH, "right")
        if right_path is None:
            return

        left_video = open_video(left_path, "left")
        right_video = open_video(right_path, "right")

        left_preview = read_first_frame(left_video, "left")
        right_preview = read_first_frame(right_video, "right")
        (
            left_frame_rotation,
            right_frame_rotation,
            left_calibration_rotation,
            right_calibration_rotation,
        ) = setup_rotation_gui(
            left_preview,
            right_preview,
            LEFT_ROI,
            RIGHT_ROI,
            LEFT_FRAME_ROTATION,
            RIGHT_FRAME_ROTATION,
            LEFT_CALIBRATION_ROTATION,
            RIGHT_CALIBRATION_ROTATION,
        )

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
        )
        right_calibration = CameraCalibration.load(
            RIGHT_CALIBRATION_PATH,
            right_calibration_rotation,
            right_frame_rotation,
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

        fps = matching_video_fps(left_video, right_video)
        pupil_detector = load_pupil_detector(
            MODEL_PATH,
            device=DEVICE,
            mask_threshold=MASK_THRESHOLD,
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

        process_frame_loop(
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
        )
    finally:
        if left_video is not None:
            left_video.release()
        if right_video is not None:
            right_video.release()
        cv2.destroyAllWindows()


# Organizes run order for all Frame Loop actions. Takes both eyes as inputs and interweaves function calls for both eyes to reduce latency between
# R and L eye outputs - function

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
):
    frame_index = 0
    while MAX_FRAMES <= 0 or frame_index < MAX_FRAMES:
        left_ok, left_frame = left_video.read()
        right_ok, right_frame = right_video.read()
        if not left_ok or not right_ok:
            break

        left_frame = rotate_frame(left_frame, left_frame_rotation)
        right_frame = rotate_frame(right_frame, right_frame_rotation)

        timestamp_s = frame_index / fps

        left_pupil = pupil_detector.detect(left_frame, LEFT_ROI)
        right_pupil = pupil_detector.detect(right_frame, RIGHT_ROI)

        left_corrected_ellipse = left_calibration.undistort_ellipse(
            left_pupil.ellipse
        )
        right_corrected_ellipse = right_calibration.undistort_ellipse(
            right_pupil.ellipse
        )

        left_estimate = left_eye_model.update(
            left_corrected_ellipse,
            left_pupil.confidence,
            timestamp_s,
            left_frame,
        )
        right_estimate = right_eye_model.update(
            right_corrected_ellipse,
            right_pupil.confidence,
            timestamp_s,
            right_frame,
        )

        left_output = create_output_frame(
            left_frame,
            "Left eye",
            LEFT_ROI,
            left_pupil,
            left_estimate,
        )
        right_output = create_output_frame(
            right_frame,
            "Right eye",
            RIGHT_ROI,
            right_pupil,
            right_estimate,
        )

        # Conditional to enable/disable text output depending on preset
        if TEXT_OUTPUT:
            print_frame_features("left", timestamp_s, left_estimate)
            print_frame_features("right", timestamp_s, right_estimate)

        # To make windows resizeable and lock aspect ratio
        cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # Show each output eye window
        cv2.imshow("Left Eye", left_output)
        cv2.imshow("Right Eye", right_output)
        if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
            break
        frame_index += 1


if __name__ == "__main__":
    run_pipeline()
