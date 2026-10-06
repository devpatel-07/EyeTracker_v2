# Import dependencies

from pathlib import Path

import cv2


# Import Scripts

from calibration import CameraCalibration
from data_logger import DataLogger
from eye_model_estimation import EyeModelEstimator, gaze_direction_from_eye_model
from feature_output import (
    create_output_frame,
    print_expected_eye_distance,
    print_frame_features,
    print_gaze_point,
)
from gaze_estimation import GazeEstimator
from pupil_detection import load_pupil_detector
from video_preparation import (
    frame_dimensions,
    rotate_frame,
    rotated_video_dimensions,
    select_eye_video,
    setup_rotation_gui,
)
from arm_communication import goToPoint
from video_sources import EyeStream, open_video_pair
from debug_tools.profiler import Profiler


HERE = Path(__file__).resolve().parent

# Debug tool presets
PROFILE = False
profiler = Profiler(PROFILE)

# Video source preset variable. "stream" uses live eye camera streams and "file" uses recorded eye videos

VIDEO_SOURCE = "file"


# Eye camera images are mirrored, so each frame is flipped horizontally before processing. ROIs are set on the flipped frame

FLIP_FRAMES = True


# Live stream preset variables. Each webcam_stream.py must already be running before eye_pipeline is run

LEFT_STREAM_HOST = "10.150.128.79"
LEFT_STREAM_PORT = 5555
RIGHT_STREAM_HOST = "172.17.90.232"
RIGHT_STREAM_PORT = 5556
STREAM_TIMEOUT_S = 5.0


# Left and right video path preset variables. Only used when VIDEO_SOURCE = "file"

LEFT_VIDEO_PATH = None #r"C:\Users\devpa\UTAustin\ECLAIR\EyeTracker_v2\cam_2026-08-15 17-10-39_lefteye.mp4"
RIGHT_VIDEO_PATH = None #r"C:\Users\devpa\UTAustin\ECLAIR\EyeTracker_v2\cam_2026-08-15 17-10-39_righteye.mp4"


# Left eye ROI region, ROI rotation, and calibration rotation preset variables

LEFT_ROI = (440, 100, 1080, 698) # Live
#LEFT_ROI = (0, 100, 1080, 698) 
LEFT_FRAME_ROTATION = "none"
LEFT_CALIBRATION_ROTATION = "none"


# Right eye ROI region, ROI rotation, and calibration rotation preset variables

RIGHT_ROI = (640, 200, 1080, 698) # Live
#RIGHT_ROI = (0, 100, 1080, 698)
RIGHT_FRAME_ROTATION = "none"
RIGHT_CALIBRATION_ROTATION = "none"

# Coordinate Conversion Variables
d = 27
h = 28
o = 2.75

# Camera calibration file preset variables

LEFT_CALIBRATION_PATH = HERE / "camera_calibration.npz"
RIGHT_CALIBRATION_PATH = HERE / "camera_calibration.npz"


# Pupil detection and pye3d preset variables

MODEL_PATH = HERE / "models" / "finetuned_2026-08-25" / "pupil_unet_best.pt"
DEVICE = "auto"
MASK_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.60
EYE_RADIUS_MM = 12.0


# Person preset variables. Eye height is a percentage of standing height unless set directly

PERSON_HEIGHT_MM = 45 * 25.4
EYE_HEIGHT_RATIO = 1
EYE_HEIGHT_MM = PERSON_HEIGHT_MM * EYE_HEIGHT_RATIO
IPD_MM = 70.0


# Glasses camera position preset variables. Yaw is the camera arm angle from straight ahead (35 degrees on the CAD),
# and pitch is how far each camera tilts up toward the eye

CAMERA_SEPARATION_MM = 113.21
CAMERA_YAW_DEG = 35.0
CAMERA_PITCH_DEG = 0.0


# Gaze point preset variables. Tolerance is the largest allowed gap between left and right gaze rays

GAZE_TOLERANCE_MM = 40.0
MIN_GAZE_DISTANCE_MM = 100.0
MAX_GAZE_DISTANCE_MM = 1000.0


# Output preset variables

TEXT_OUTPUT = True
SAVE_DATA = True
DATA_OUTPUT_PATH = HERE / "eye_data.csv"
WAIT_MS = 1
MAX_FRAMES = 0


# Opens live eye streams or recorded eye videos depending on video source preset - function

def open_eye_sources():
    if VIDEO_SOURCE == "stream":
        left_source = EyeStream(LEFT_STREAM_HOST, LEFT_STREAM_PORT, "left", STREAM_TIMEOUT_S, FLIP_FRAMES)
        right_source = EyeStream(RIGHT_STREAM_HOST, RIGHT_STREAM_PORT, "right", STREAM_TIMEOUT_S, FLIP_FRAMES)
        return left_source, right_source

    if VIDEO_SOURCE == "file":
        left_path = select_eye_video(LEFT_VIDEO_PATH, "left")
        if left_path is None:
            return None, None
        right_path = select_eye_video(RIGHT_VIDEO_PATH, "right")
        if right_path is None:
            return None, None
        return open_video_pair(left_path, right_path, FLIP_FRAMES)

    raise ValueError(f"unsupported video source: {VIDEO_SOURCE}")


# Organizes run order for one time actions before Frame Loop - function

def run_pipeline():
    left_source = None
    right_source = None
    data_logger = None
    try:
        left_source, right_source = open_eye_sources()
        if left_source is None:
            return

        left_preview = left_source.preview_frame()
        right_preview = right_source.preview_frame()
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

        left_source_size = frame_dimensions(left_preview)
        right_source_size = frame_dimensions(right_preview)
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
        gaze_estimator = GazeEstimator(
            ipd_mm=IPD_MM,
            eye_height_mm=EYE_HEIGHT_MM,
            camera_separation_mm=CAMERA_SEPARATION_MM,
            camera_yaw_deg=CAMERA_YAW_DEG,
            camera_pitch_deg=CAMERA_PITCH_DEG,
            tolerance_mm=GAZE_TOLERANCE_MM,
            min_distance_mm=MIN_GAZE_DISTANCE_MM,
            max_distance_mm=MAX_GAZE_DISTANCE_MM,
        )
        print_expected_eye_distance(gaze_estimator.expected_eye_distance_mm)

        if SAVE_DATA:
            data_logger = DataLogger(DATA_OUTPUT_PATH)

        process_frame_loop(
            left_source,
            right_source,
            left_frame_rotation,
            right_frame_rotation,
            left_calibration,
            right_calibration,
            pupil_detector,
            left_eye_model,
            right_eye_model,
            gaze_estimator,
            data_logger,
        )
    finally:
        if data_logger is not None:
            data_logger.close()
        if left_source is not None:
            left_source.release()
        if right_source is not None:
            right_source.release()
        cv2.destroyAllWindows()


# Organizes run order for all Frame Loop actions. Takes both eyes as inputs and interweaves function calls for both eyes to reduce latency between
# R and L eye outputs - function

def process_frame_loop(
    left_source,
    right_source,
    left_frame_rotation,
    right_frame_rotation,
    left_calibration,
    right_calibration,
    pupil_detector,
    left_eye_model,
    right_eye_model,
    gaze_estimator,
    data_logger=None,
):
    frame_index = 0
    while MAX_FRAMES <= 0 or frame_index < MAX_FRAMES:
        left_frame, left_timestamp_s = left_source.read()
        right_frame, right_timestamp_s = right_source.read()
        if left_frame is None or right_frame is None:
            break

        # Start frame loop profiler
        profiler.start()

        left_frame = rotate_frame(left_frame, left_frame_rotation)
        right_frame = rotate_frame(right_frame, right_frame_rotation)

        # Time of the newer frame in the left and right frame pair
        timestamp_s = max(left_timestamp_s, right_timestamp_s)

        profiler.clear()

        left_pupil = pupil_detector.detect(left_frame, LEFT_ROI)
        right_pupil = pupil_detector.detect(right_frame, RIGHT_ROI)

        profiler.checkpoint("Pupil Detection")

        left_corrected_ellipse = left_calibration.undistort_ellipse(
            left_pupil.ellipse
        )
        right_corrected_ellipse = right_calibration.undistort_ellipse(
            right_pupil.ellipse
        )

        profiler.checkpoint("Pupil Undistortion")

        left_estimate = left_eye_model.update(
            left_corrected_ellipse,
            left_pupil.confidence,
            left_timestamp_s,
            left_frame,
        )
        right_estimate = right_eye_model.update(
            right_corrected_ellipse,
            right_pupil.confidence,
            right_timestamp_s,
            right_frame,
        )

        profiler.checkpoint("Eye Model Update")

        gaze = gaze_estimator.estimate(left_estimate, right_estimate)

        profiler.checkpoint("Gaze Point")

        if data_logger is not None:
            data_logger.log_frame(
                frame_index,
                timestamp_s,
                left_eye_center=left_estimate.eye_center_mm,
                left_gaze=gaze_direction_from_eye_model(left_estimate),
                right_eye_center=right_estimate.eye_center_mm,
                right_gaze=gaze_direction_from_eye_model(right_estimate),
                gaze=gaze,
            )
            profiler.checkpoint("Data Logging")

        left_output = create_output_frame(
            left_frame,
            "Left eye",
            LEFT_ROI,
            left_pupil,
            left_estimate,
            gaze,
        )
        right_output = create_output_frame(
            right_frame,
            "Right eye",
            RIGHT_ROI,
            right_pupil,
            right_estimate,
            gaze,
        )

        profiler.checkpoint("Visual Output")

        # Conditional to enable/disable text output depending on preset
        if TEXT_OUTPUT:
            print_frame_features("left", timestamp_s, left_estimate)
            print_frame_features("right", timestamp_s, right_estimate)
            print_gaze_point(timestamp_s, gaze)

        # To make windows resizeable and lock aspect ratio
        cv2.namedWindow("Left Eye", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("Right Eye", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        profiler.clear()

        # Show each output eye window
        cv2.imshow("Left Eye", left_output)
        cv2.imshow("Right Eye", right_output)
        key = cv2.waitKey(WAIT_MS)
        if key & 0xFF == ord("q"):
            break
        if key & 0xFF == ord("g") and (gaze.point_mm is not None):
            goToPoint(*gaze.point_mm,d,h,o)
            print("Went to point")
        frame_index += 1

        profiler.checkpoint("Window Display")
        profiler.end()


if __name__ == "__main__":
    run_pipeline()
