# Import dependencies

from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np


VALID_ROTATIONS = ("none", "clockwise", "180", "counterclockwise")

_ROTATION_QUARTER_TURNS = {
    "none": 0,
    "clockwise": 1,
    "180": 2,
    "counterclockwise": 3,
}
_QUARTER_TURN_ROTATIONS = {
    turns: name for name, turns in _ROTATION_QUARTER_TURNS.items()
}


# Determines calibration orientation depending on ROI + calibration rotation preset - function

def combined_rotation(calibration_rotation, frame_rotation):
    _check_rotation(calibration_rotation)
    _check_rotation(frame_rotation)
    turns = (
        _ROTATION_QUARTER_TURNS[calibration_rotation]
        + _ROTATION_QUARTER_TURNS[frame_rotation]
    ) % 4
    return _QUARTER_TURN_ROTATIONS[turns]


def _check_rotation(rotation):
    if rotation not in VALID_ROTATIONS:
        choices = ", ".join(VALID_ROTATIONS)
        raise ValueError(f"unsupported rotation {rotation!r}; use one of {choices}")


def _rotated_size(size, rotation):
    _check_rotation(rotation)
    width, height = (int(value) for value in size)
    if rotation in {"clockwise", "counterclockwise"}:
        return height, width
    return width, height


# Converts points between raw calibration and rotated video coordinates - function

def _raw_to_video_points(points, raw_size, rotation):
    width, height = raw_size
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    if rotation == "clockwise":
        return np.column_stack((height - 1 - y, x))
    if rotation == "counterclockwise":
        return np.column_stack((y, width - 1 - x))
    if rotation == "180":
        return np.column_stack((width - 1 - x, height - 1 - y))
    return points.copy()


def _video_to_raw_points(points, raw_size, rotation):
    width, height = raw_size
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    if rotation == "clockwise":
        return np.column_stack((y, height - 1 - x))
    if rotation == "counterclockwise":
        return np.column_stack((width - 1 - y, x))
    if rotation == "180":
        return np.column_stack((width - 1 - x, height - 1 - y))
    return points.copy()


# Generates camera matrix for rotated video coordinates - function

def _rotated_camera_matrix(camera_matrix, raw_size, rotation):
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    width, height = raw_size
    fx = matrix[0, 0]
    fy = matrix[1, 1]
    cx = matrix[0, 2]
    cy = matrix[1, 2]
    if rotation == "clockwise":
        return np.array(
            [[fy, 0.0, height - 1 - cy], [0.0, fx, cx], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if rotation == "counterclockwise":
        return np.array(
            [[fy, 0.0, cy], [0.0, fx, width - 1 - cx], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if rotation == "180":
        return np.array(
            [
                [fx, 0.0, width - 1 - cx],
                [0.0, fy, height - 1 - cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    return matrix.copy()


def _ellipse_points(ellipse, samples):
    try:
        (center_x, center_y), (axis_width, axis_height), angle_degrees = ellipse
        values = (center_x, center_y, axis_width, axis_height, angle_degrees)
        center_x, center_y, axis_width, axis_height, angle_degrees = map(
            float, values
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ellipse must use OpenCV fitEllipse format") from error
    if (
        samples < 5
        or axis_width <= 0
        or axis_height <= 0
        or not all(math.isfinite(value) for value in values)
    ):
        raise ValueError("ellipse must contain finite positive axes")

    angle = np.deg2rad(angle_degrees)
    parameters = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    local_x = axis_width * 0.5 * np.cos(parameters)
    local_y = axis_height * 0.5 * np.sin(parameters)
    return np.column_stack(
        (
            center_x + local_x * np.cos(angle) - local_y * np.sin(angle),
            center_y + local_x * np.sin(angle) + local_y * np.cos(angle),
        )
    )


@dataclass(frozen=True)
class CameraCalibration:
    raw_camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    raw_image_size: tuple[int, int]
    calibration_rotation: str
    frame_rotation: str

    # Loads camera calibration file - function

    @classmethod
    def load(cls, path, calibration_rotation="none", frame_rotation="none"):
        _check_rotation(calibration_rotation)
        _check_rotation(frame_rotation)
        calibration_path = Path(path).expanduser()
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"camera calibration file not found: {calibration_path}"
            )
        try:
            with np.load(calibration_path) as data:
                matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
                distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
                raw_size = tuple(int(value) for value in data["image_size"])
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"could not load camera calibration {calibration_path}: {error}"
            ) from error

        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera matrix must be a finite 3x3 matrix")
        if distortion.size == 0 or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion coefficients must be finite")
        if len(raw_size) != 2 or any(value <= 0 for value in raw_size):
            raise ValueError("calibration image dimensions must be positive")
        return cls(
            matrix,
            distortion,
            raw_size,
            calibration_rotation,
            frame_rotation,
        )

    @property
    def processed_rotation(self):
        return combined_rotation(self.calibration_rotation, self.frame_rotation)

    @property
    def source_video_size(self):
        return _rotated_size(self.raw_image_size, self.calibration_rotation)

    @property
    def video_size(self):
        return _rotated_size(self.raw_image_size, self.processed_rotation)

    @property
    def video_camera_matrix(self):
        return _rotated_camera_matrix(
            self.raw_camera_matrix,
            self.raw_image_size,
            self.processed_rotation,
        )

    # Validates video dimensions against camera calibration - function

    def validate_video_dimensions(self, source_size, processed_size, side):
        source_size = tuple(int(value) for value in source_size)
        processed_size = tuple(int(value) for value in processed_size)
        if source_size != self.source_video_size:
            raise ValueError(
                f"{side} source video size {source_size} does not match calibrated "
                f"size {self.source_video_size} for calibration rotation "
                f"{self.calibration_rotation!r}"
            )
        if processed_size != self.video_size:
            raise ValueError(
                f"{side} processed video size {processed_size} does not match "
                f"calibrated size {self.video_size} for combined rotation "
                f"{self.processed_rotation!r}"
            )

    def raw_to_processed_points(self, points):
        return _raw_to_video_points(
            points,
            self.raw_image_size,
            self.processed_rotation,
        )

    def processed_to_raw_points(self, points):
        return _video_to_raw_points(
            points,
            self.raw_image_size,
            self.processed_rotation,
        )

    def undistort_points(self, points):
        raw_points = self.processed_to_raw_points(points)
        corrected_raw = cv2.undistortPoints(
            raw_points.reshape(-1, 1, 2),
            self.raw_camera_matrix,
            self.distortion_coefficients,
            P=self.raw_camera_matrix,
        ).reshape(-1, 2)
        return self.raw_to_processed_points(corrected_raw)

    # Converts projected undistorted points back to distorted display coordinates when needed - function

    def distort_points(self, points):
        corrected_raw = self.processed_to_raw_points(points)
        fx = self.raw_camera_matrix[0, 0]
        fy = self.raw_camera_matrix[1, 1]
        cx = self.raw_camera_matrix[0, 2]
        cy = self.raw_camera_matrix[1, 2]
        object_points = np.column_stack(
            (
                (corrected_raw[:, 0] - cx) / fx,
                (corrected_raw[:, 1] - cy) / fy,
                np.ones(len(corrected_raw), dtype=np.float64),
            )
        )
        distorted_raw, _ = cv2.projectPoints(
            object_points,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            self.raw_camera_matrix,
            self.distortion_coefficients,
        )
        return self.raw_to_processed_points(distorted_raw.reshape(-1, 2))

    # Corrects pupil ellipse for camera distortion - function

    def undistort_ellipse(self, ellipse, samples=72):
        if ellipse is None:
            return None
        corrected = self.undistort_points(_ellipse_points(ellipse, samples))
        return cv2.fitEllipse(corrected.astype(np.float32).reshape(-1, 1, 2))

    def distort_ellipse(self, ellipse, samples=72):
        if ellipse is None:
            return None
        distorted = self.distort_points(_ellipse_points(ellipse, samples))
        return cv2.fitEllipse(distorted.astype(np.float32).reshape(-1, 1, 2))
