from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


VALID_FRAME_ROTATIONS = {
    "none",
    "clockwise",
    "counterclockwise",
    "180",
}


@dataclass(frozen=True)
class FrameTransform:
    rotation: str = "none"
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self):
        if not isinstance(self.rotation, str) or self.rotation not in VALID_FRAME_ROTATIONS:
            raise ValueError(
                "rotation must be none, clockwise, counterclockwise, or 180"
            )
        for name in ("flip_horizontal", "flip_vertical"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")

    def output_size(self, input_size):
        width, height = _validate_size(input_size)
        if self.rotation in {"clockwise", "counterclockwise"}:
            return (height, width)
        return (width, height)

    def apply_frame(self, frame):
        if self.rotation == "clockwise":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == "counterclockwise":
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.rotation == "180":
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.flip_vertical:
            frame = cv2.flip(frame, 0)
        return frame

    def forward_points(self, points, input_size):
        width, height = _validate_size(input_size)
        points = _validate_points(points)
        points = _forward_rotation_points(points, width, height, self.rotation)
        output_width, output_height = self.output_size((width, height))
        if self.flip_horizontal:
            points[:, 0] = output_width - 1 - points[:, 0]
        if self.flip_vertical:
            points[:, 1] = output_height - 1 - points[:, 1]
        return points

    def inverse_points(self, points, input_size):
        width, height = _validate_size(input_size)
        points = _validate_points(points)
        output_width, output_height = self.output_size((width, height))
        if self.flip_vertical:
            points[:, 1] = output_height - 1 - points[:, 1]
        if self.flip_horizontal:
            points[:, 0] = output_width - 1 - points[:, 0]
        return _inverse_rotation_points(points, width, height, self.rotation)

    def to_runtime_camera_axes(self, coordinate):
        x, y, z = _validate_coordinate(coordinate)
        if self.rotation == "clockwise":
            x, y = -y, x
        elif self.rotation == "counterclockwise":
            x, y = y, -x
        elif self.rotation == "180":
            x, y = -x, -y
        if self.flip_horizontal:
            x = -x
        if self.flip_vertical:
            y = -y
        return (x, y, z)

    def to_source_camera_axes(self, coordinate):
        x, y, z = _validate_coordinate(coordinate)
        if self.flip_vertical:
            y = -y
        if self.flip_horizontal:
            x = -x
        if self.rotation == "clockwise":
            x, y = y, -x
        elif self.rotation == "counterclockwise":
            x, y = -y, x
        elif self.rotation == "180":
            x, y = -x, -y
        return (x, y, z)

    def transform_camera_matrix(self, matrix, input_size):
        matrix = _validate_camera_matrix(matrix)
        width, height = _validate_size(input_size)
        principal_point = self.forward_points(
            matrix[np.newaxis, 0:2, 2],
            (width, height),
        )[0]
        fx, fy = matrix[0, 0], matrix[1, 1]
        if self.rotation in {"clockwise", "counterclockwise"}:
            fx, fy = fy, fx
        return np.array(
            [
                [abs(fx), 0.0, principal_point[0]],
                [0.0, abs(fy), principal_point[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def _validate_size(size):
    try:
        width, height = (int(value) for value in size)
    except (TypeError, ValueError) as error:
        raise ValueError("image size must contain two positive integers") from error
    if width <= 0 or height <= 0:
        raise ValueError("image size must contain two positive integers")
    return width, height


def _validate_points(points):
    try:
        points = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("points must have shape (N, 2)") from error
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    return points.copy()


def _validate_coordinate(coordinate):
    try:
        values = tuple(float(value) for value in coordinate)
    except (TypeError, ValueError) as error:
        raise ValueError("coordinate must be a finite 3D coordinate") from error
    if len(values) != 3 or not all(np.isfinite(value) for value in values):
        raise ValueError("coordinate must be a finite 3D coordinate")
    return values


def _validate_camera_matrix(matrix):
    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("camera matrix must be a finite 3x3 pinhole matrix") from error
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera matrix must be a finite 3x3 pinhole matrix")
    if (
        matrix[0, 1] != 0.0
        or matrix[1, 0] != 0.0
        or matrix[2, 0] != 0.0
        or matrix[2, 1] != 0.0
        or matrix[2, 2] != 1.0
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise ValueError("camera matrix must be a finite 3x3 pinhole matrix")
    return matrix


def _forward_rotation_points(points, width, height, rotation):
    x = points[:, 0]
    y = points[:, 1]
    if rotation == "clockwise":
        return np.column_stack((height - 1 - y, x))
    if rotation == "counterclockwise":
        return np.column_stack((y, width - 1 - x))
    if rotation == "180":
        return np.column_stack((width - 1 - x, height - 1 - y))
    return points


def _inverse_rotation_points(points, width, height, rotation):
    x = points[:, 0]
    y = points[:, 1]
    if rotation == "clockwise":
        return np.column_stack((y, height - 1 - x))
    if rotation == "counterclockwise":
        return np.column_stack((width - 1 - y, x))
    if rotation == "180":
        return np.column_stack((width - 1 - x, height - 1 - y))
    return points


class TransformedVideoCapture:
    def __init__(self, capture, frame_transform):
        self._capture = capture
        self._frame_transform = frame_transform
        source_size = (
            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self._source_size = _validate_size(source_size)
        self._output_size = frame_transform.output_size(self._source_size)
        self._released = False

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._output_size[0])
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._output_size[1])
        return self._capture.get(property_id)

    def read(self):
        successful, frame = self._capture.read()
        if successful and frame is not None:
            frame = self._frame_transform.apply_frame(frame)
        return successful, frame

    def isOpened(self):
        return self._capture.isOpened()

    def release(self):
        if not self._released:
            self._capture.release()
            self._released = True
