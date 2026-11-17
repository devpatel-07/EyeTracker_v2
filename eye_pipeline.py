# Import dependencies

from pathlib import Path

import cv2
import numpy as np


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
from video_sources import DirectWebcamSource, DummyEyeSource, EyeStream, open_video_pair
from debug_tools.profiler import Profiler
from coord_convert import log_live_gaze_vectors
import tracker


HERE = Path(__file__).resolve().parent

# Debug tool presets
PROFILE = False
profiler = Profiler(PROFILE)

# Video source preset variable.
# "usb" uses direct local USB camera indices (no webcam_stream.py needed)
# "stream" uses network eye camera streams via webcam_stream.py
# "file" uses recorded eye videos
VIDEO_SOURCE = "usb"


# Eye camera images are mirrored, so each frame is flipped horizontally before processing. ROIs are set on the flipped frame
FLIP_FRAMES = True


# Camera enable presets: set to False (or set index to None) to ignore a camera
ENABLE_LEFT_EYE = True
ENABLE_RIGHT_EYE = False

# Direct USB camera index presets. Only used when VIDEO_SOURCE = "usb"
LEFT_CAMERA_INDEX = 0
RIGHT_CAMERA_INDEX = None


# Live stream preset variables. Each webcam_stream.py must already be running before eye_pipeline is run
# Only used when VIDEO_SOURCE = "stream"
LEFT_STREAM_HOST = "172.20.10.6"
LEFT_STREAM_PORT = 5555
RIGHT_STREAM_HOST = "172.20.10.2"
RIGHT_STREAM_PORT = 5555
STREAM_TIMEOUT_S = 5.0


# Left and right video path preset variables. Only used when VIDEO_SOURCE = "file"
LEFT_VIDEO_PATH = r"C:\Users\devpa\UTAustin\ECLAIR\EyeTracker_v2\cam_2026-08-15 17-10-39_lefteye.mp4"
RIGHT_VIDEO_PATH = r"C:\Users\devpa\UTAustin\ECLAIR\EyeTracker_v2\cam_2026-08-15 17-10-39_righteye.mp4"


# Left eye ROI region, ROI rotation, and calibration rotation preset variables
#LEFT_ROI = (440, 200, 1080, 698) # Live video
LEFT_ROI = (200, 100, 1080, 880)
LEFT_FRAME_ROTATION = "clockwise"
LEFT_CALIBRATION_ROTATION = "none"


# Right eye ROI region, ROI rotation, and calibration rotation preset variables
#RIGHT_ROI = (640, 200, 1080, 698) # Live video
RIGHT_ROI = (0, 100, 1080, 698)
RIGHT_FRAME_ROTATION = "counterclockwise"
RIGHT_CALIBRATION_ROTATION = "none"

# Coordinate Conversion Variables
d = 22 # distance from arm
h = 33.5 # height of table the arm is on
o = 2.75 # offset from table to shoulder servo base

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
PERSON_HEIGHT_MM = 48 * 25.4
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

# Base camera / ArUco tracking presets
ENABLE_BASE_CAM_TRACKER = True
BASE_CAM_INDEX = 1
BASE_CAM_WIDTH = 1920
BASE_CAM_HEIGHT = 1080
BASE_CALIBRATION_PATH = HERE / "base_camera_calibration.npz"
BASE_FOCUS_STEPS = (0.0, 28.0, 57.0, 85.0, 113.0, 142.0, 170.0, 198.0, 227.0, 255.0)
BASE_FOCUS_SETTLE_FRAMES = 0
BASE_FOCUS_HOLD_FRAMES = 5
ARUCO_MARKER_ID = 1
ARUCO_MARKER_MM = 20.0
ARUCO_THRESHOLD = False
ARUCO_DEBLUR = False
LIVE_GAZE_LOG_PATH = HERE / "live_gaze_vectors.txt"


# Opens live eye streams or recorded eye videos depending on video source preset - function

def open_eye_sources():
    if VIDEO_SOURCE == "usb":
        if ENABLE_LEFT_EYE and LEFT_CAMERA_INDEX is not None:
            left_source = DirectWebcamSource(LEFT_CAMERA_INDEX, "left", FLIP_FRAMES)
        else:
            left_source = DummyEyeSource("left")

        if ENABLE_RIGHT_EYE and RIGHT_CAMERA_INDEX is not None:
            right_source = DirectWebcamSource(RIGHT_CAMERA_INDEX, "right", FLIP_FRAMES)
        else:
            right_source = DummyEyeSource("right")
        return left_source, right_source

    if VIDEO_SOURCE == "stream":
        if ENABLE_LEFT_EYE:
            left_source = EyeStream(LEFT_STREAM_HOST, LEFT_STREAM_PORT, "left", STREAM_TIMEOUT_S, FLIP_FRAMES)
        else:
            left_source = DummyEyeSource("left")

        if ENABLE_RIGHT_EYE:
            right_source = EyeStream(RIGHT_STREAM_HOST, RIGHT_STREAM_PORT, "right", STREAM_TIMEOUT_S, FLIP_FRAMES)
        else:
            right_source = DummyEyeSource("right")
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

        # Base camera tracker initialization
        base_cap = None
        base_matrix = None
        base_distortion = None
        aruco_detector = None
        stepped_focus = None
        if ENABLE_BASE_CAM_TRACKER:
            try:
                base_cap = cv2.VideoCapture(BASE_CAM_INDEX)
                base_cap.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_CAM_WIDTH)
                base_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_CAM_HEIGHT)
                if base_cap.isOpened():
                    tracker.lock_camera_focus(base_cap)
                    stepped_focus = tracker.SteppedAutofocus(
                        base_cap,
                        BASE_FOCUS_STEPS,
                        settle_frames=BASE_FOCUS_SETTLE_FRAMES,
                        hold_frames=BASE_FOCUS_HOLD_FRAMES,
                    )
                if base_cap.isOpened() and BASE_CALIBRATION_PATH.exists():
                    base_matrix, base_distortion = tracker.camera_model(
                        BASE_CALIBRATION_PATH, (BASE_CAM_WIDTH, BASE_CAM_HEIGHT)
                    )
                    aruco_detector = tracker.detector()
                elif not base_cap.isOpened():
                    print(f"Warning: Base camera index {BASE_CAM_INDEX} could not be opened.")
                    base_cap = None
            except Exception as e:
                print(f"Base camera tracker setup failed: {e}")
                base_cap = None

        print("\nBase camera / ArUco settings")
        print(f"  enabled: {ENABLE_BASE_CAM_TRACKER}")
        print(f"  camera: {BASE_CAM_INDEX} ({BASE_CAM_WIDTH}x{BASE_CAM_HEIGHT})")
        print(f"  marker: ID {ARUCO_MARKER_ID}, {ARUCO_MARKER_MM:g} mm")
        print(f"  calibration: {BASE_CALIBRATION_PATH}")
        print(f"  grayscale: True, threshold: {ARUCO_THRESHOLD}")
        print(f"  deblur: {ARUCO_DEBLUR}")
        aruco_params = tracker.detector_parameters()
        print(
            "  relaxed detection: "
            f"min perimeter {aruco_params.minMarkerPerimeterRate}, "
            f"border error {aruco_params.maxErroneousBitsInBorderRate}, "
            f"error correction {aruco_params.errorCorrectionRate}"
        )
        print(
            f"  stepped focus: {BASE_FOCUS_STEPS}; settle "
            f"{BASE_FOCUS_SETTLE_FRAMES} frames; hold {BASE_FOCUS_HOLD_FRAMES} frames\n"
        )

        # Reset / initialize live log file
        with open(LIVE_GAZE_LOG_PATH, "w", encoding="utf-8") as f:
            f.write("# Live Gaze Vectors and ArUco Pose Log\n")

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
            base_cap=base_cap,
            base_matrix=base_matrix,
            base_distortion=base_distortion,
            aruco_detector=aruco_detector,
            stepped_focus=stepped_focus,
        )
    finally:
        if 'base_cap' in locals() and base_cap is not None:
            base_cap.release()
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
    base_cap=None,
    base_matrix=None,
    base_distortion=None,
    aruco_detector=None,
    stepped_focus=None,
):
    frame_index = 0
    aruco_deblurrer = None
    while True:
        left_frame, left_timestamp_s = left_source.read()
        right_frame, right_timestamp_s = right_source.read()
        if left_frame is None or right_frame is None:
            # Dropped/late frame: keep running, only 'q' stops the pipeline
            print(
                f"Missing frame at index {frame_index}: "
                f"left {'missing' if left_frame is None else 'ok'}, "
                f"right {'missing' if right_frame is None else 'ok'}"
            )
            if cv2.waitKey(WAIT_MS) & 0xFF == ord("q"):
                print(f"Pipeline stopped at frame {frame_index}: 'q' pressed")
                break
            continue

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

        left_direction = gaze_estimator.head_gaze_direction(left_estimate, gaze_estimator.left_rotation)
        right_direction = gaze_estimator.head_gaze_direction(right_estimate, gaze_estimator.right_rotation)

        gaze = gaze_estimator.estimate(left_estimate, right_estimate)

        profiler.checkpoint("Gaze Point")

        # Track ArUco marker from base camera and log vectors
        base_display_frame = None
        if base_cap is not None and aruco_detector is not None:
            ok, base_frame = base_cap.read()
            if ok and base_frame is not None:
                if stepped_focus is not None:
                    stepped_focus.update(base_frame)
                gray = cv2.cvtColor(base_frame, cv2.COLOR_BGR2GRAY)
                detection_image = tracker.otsu_threshold(gray) if ARUCO_THRESHOLD else gray
                base_display_frame = cv2.cvtColor(detection_image, cv2.COLOR_GRAY2BGR)
                if ARUCO_DEBLUR:
                    # Build the FFT filters once, and rebuild only if the
                    # camera unexpectedly changes resolution.
                    if aruco_deblurrer is None or aruco_deblurrer.shape != gray.shape:
                        aruco_deblurrer = tracker.Deblurrer(gray.shape)
                    corners, ids, preprocessing_stage = tracker.detect_deblurred(
                        aruco_detector,
                        aruco_deblurrer,
                        gray,
                        ARUCO_MARKER_ID,
                        threshold=ARUCO_THRESHOLD,
                    )
                else:
                    corners, ids, _ = aruco_detector.detectMarkers(detection_image)
                    preprocessing_stage = "grayscale"

                cv2.putText(
                    base_display_frame,
                    f"ArUco preprocessing: {preprocessing_stage}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
                if stepped_focus is not None:
                    cv2.putText(
                        base_display_frame,
                        stepped_focus.status,
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 255),
                        2,
                    )
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(base_display_frame, corners, ids)
                    matches = np.flatnonzero(ids.flatten() == ARUCO_MARKER_ID)
                    if matches.size:
                        pose = tracker.estimate_pose(
                            corners[int(matches[0])],
                            ARUCO_MARKER_MM,
                            base_matrix,
                            base_distortion,
                        )
                        if pose is not None:
                            error, rvec, tvec = pose
                            xyz = tvec.ravel()
                            angles = tracker.euler_degrees(cv2.Rodrigues(rvec)[0])
                            cv2.drawFrameAxes(
                                base_display_frame,
                                base_matrix,
                                base_distortion,
                                rvec,
                                tvec,
                                ARUCO_MARKER_MM * 0.5,
                                2,
                            )
                            cv2.putText(
                                base_display_frame,
                                f"ID {ARUCO_MARKER_ID}: X={xyz[0]:.1f} Y={xyz[1]:.1f} Z={xyz[2]:.1f} mm",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                            log_live_gaze_vectors(
                                LIVE_GAZE_LOG_PATH,
                                left_direction,
                                right_direction,
                                float(xyz[0]),
                                float(xyz[1]),
                                float(xyz[2]),
                                float(angles[0]),
                                float(angles[1]),
                                float(angles[2]),
                            )
                else:
                    cv2.putText(
                        base_display_frame,
                        f"ID {ARUCO_MARKER_ID}: NOT DETECTED",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    # Marker not currently visible in this frame
                    log_live_gaze_vectors(
                        LIVE_GAZE_LOG_PATH,
                        left_direction,
                        right_direction,
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                    )

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
        if base_display_frame is not None:
            cv2.namedWindow("Base Camera ArUco Tracker", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        profiler.clear()

        # Show each output eye window
        cv2.imshow("Left Eye", left_output)
        cv2.imshow("Right Eye", right_output)
        if base_display_frame is not None:
            cv2.imshow("Base Camera ArUco Tracker", base_display_frame)
        key = cv2.waitKey(WAIT_MS)
        if key & 0xFF == ord("q"):
            print(f"Pipeline stopped at frame {frame_index}: 'q' pressed")
            break
        frame_index += 1

        profiler.checkpoint("Window Display")

        # Move Robotic Arm if user presses 'g'
        if key & 0xFF == ord("g") and (gaze.point_mm is not None):
            goToPoint(*gaze.point_mm,d,h,o) 
            print("Went to point")
            
        profiler.end()


if __name__ == "__main__":
    run_pipeline()
