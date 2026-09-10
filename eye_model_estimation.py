# Import dependencies

from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


PYE3D_REFERENCE_EYE_RADIUS_MM = 10.392304845413264


# Stores pye3d eye model outputs - data class

@dataclass(frozen=True)
class EyeModelEstimate:
    ready: bool
    eye_center_mm: tuple[float, float, float] | None
    pupil_center_mm: tuple[float, float, float] | None
    pupil_diameter_mm: float | None
    pupil_confidence: float
    model_confidence: float | None
    update_time_ms: float
    projected_eye_sphere: tuple | None
    projected_eye_center: tuple[float, float] | None
    projected_pupil_center: tuple[float, float] | None
    status: str


# Checks pye3d coordinate output contains valid XYZ values - function

def _finite_triplet(value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        return None
    return result


def _finite_point(value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 2 or not all(math.isfinite(component) for component in result):
        return None
    return result


def _opencv_ellipse(value):
    if not isinstance(value, dict):
        return None
    center = _finite_point(value.get("center"))
    axes = _finite_point(value.get("axes"))
    try:
        angle = float(value.get("angle"))
    except (TypeError, ValueError):
        return None
    if (
        center is None
        or axes is None
        or axes[0] <= 0
        or axes[1] <= 0
        or not math.isfinite(angle)
    ):
        return None
    return center, axes, angle


# Converts corrected pupil ellipse into pye3d input format - function

def _pye3d_ellipse(ellipse, width, height, principal_x, principal_y):
    try:
        (center_x, center_y), (axis_0, axis_1), angle = ellipse
        center_x, center_y, axis_0, axis_1, angle = map(
            float,
            (center_x, center_y, axis_0, axis_1, angle),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("ellipse must use OpenCV fitEllipse format") from error
    if not all(
        math.isfinite(value)
        for value in (center_x, center_y, axis_0, axis_1, angle)
    ) or axis_0 <= 0 or axis_1 <= 0:
        raise ValueError("ellipse must contain finite positive axes")
    if axis_0 > axis_1:
        axis_0, axis_1 = axis_1, axis_0
        angle += 90.0

    model_center = (
        center_x + width / 2.0 - principal_x,
        center_y + height / 2.0 - principal_y,
    )
    return {
        "center": model_center,
        "axes": (axis_0, axis_1),
        "angle": angle % 180.0,
    }


# Converts pye3d projected image coordinates back to processed frame coordinates - function

def _projected_point_to_processed(point, width, height, principal_x, principal_y):
    point = _finite_point(point)
    if point is None:
        return None
    return (
        point[0] - width / 2.0 + principal_x,
        point[1] - height / 2.0 + principal_y,
    )


def _projected_ellipse_to_processed(
    ellipse,
    width,
    height,
    principal_x,
    principal_y,
):
    if ellipse is None:
        return None
    center, axes, angle = ellipse
    center = _projected_point_to_processed(
        center,
        width,
        height,
        principal_x,
        principal_y,
    )
    return (center, axes, angle) if center is not None else None


# Creates one pye3d eye model estimator - class

class EyeModelEstimator:
    def __init__(self, calibration, min_confidence, eye_radius_mm):
        try:
            from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
        except ImportError as error:
            raise RuntimeError("pye3d is not installed") from error

        matrix = np.asarray(calibration.video_camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera matrix must be finite and 3x3")
        self.width, self.height = (
            int(value) for value in calibration.video_size
        )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("processed video dimensions must be positive")
        self.calibration = calibration
        self.fx = float(matrix[0, 0])
        self.fy = float(matrix[1, 1])
        self.cx = float(matrix[0, 2])
        self.cy = float(matrix[1, 2])
        self.min_confidence = float(min_confidence)
        self.scale = float(eye_radius_mm) / PYE3D_REFERENCE_EYE_RADIUS_MM
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("minimum confidence must be between zero and one")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("eye radius must be finite and positive")

        camera = CameraModel(
            focal_length=(self.fx + self.fy) / 2.0,
            resolution=(self.width, self.height),
        )
        self.detector = Detector3D(
            camera=camera,
            threshold_swirski=max(0.0, self.min_confidence - 1e-6),
            threshold_short_term=self.min_confidence,
            threshold_long_term=self.min_confidence,
            long_term_mode=DetectorMode.blocking,
        )
        self.last_timestamp = None

    def _empty(
        self,
        pupil_confidence,
        status,
        elapsed_ms=0.0,
        model_confidence=None,
    ):
        return EyeModelEstimate(
            ready=False,
            eye_center_mm=None,
            pupil_center_mm=None,
            pupil_diameter_mm=None,
            pupil_confidence=float(pupil_confidence),
            model_confidence=model_confidence,
            update_time_ms=float(elapsed_ms),
            projected_eye_sphere=None,
            projected_eye_center=None,
            projected_pupil_center=None,
            status=status,
        )

    # Updates pye3d temporal eye model - function

    def update(self, corrected_ellipse, pupil_confidence, timestamp_s, frame):
        pupil_confidence = float(pupil_confidence)
        timestamp_s = float(timestamp_s)
        if not math.isfinite(pupil_confidence) or not 0.0 <= pupil_confidence <= 1.0:
            raise ValueError("pupil confidence must be between zero and one")
        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("pye3d timestamp must be finite and nonnegative")
        if corrected_ellipse is None:
            return self._empty(pupil_confidence, "no pupil")
        if pupil_confidence < self.min_confidence:
            return self._empty(pupil_confidence, "low pupil confidence")
        if self.last_timestamp is not None and timestamp_s <= self.last_timestamp:
            raise ValueError("pye3d timestamps must strictly increase")

        if frame.ndim == 3:
            grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            grayscale = frame
        else:
            raise ValueError("processed frame must be a grayscale or BGR image")
        if grayscale.shape != (self.height, self.width):
            raise ValueError("full processed frame has the wrong dimensions for pye3d")
        if grayscale.dtype != np.uint8:
            raise ValueError("full processed grayscale frame must use uint8 pixels")

        ellipse = _pye3d_ellipse(
            corrected_ellipse,
            self.width,
            self.height,
            self.cx,
            self.cy,
        )
        datum = {
            "ellipse": ellipse,
            "diameter": ellipse["axes"][1],
            "location": ellipse["center"],
            "confidence": pupil_confidence,
            "timestamp": timestamp_s,
            "norm_pos": (
                ellipse["center"][0] / self.width,
                1.0 - ellipse["center"][1] / self.height,
            ),
            "method": "custom-ml",
        }

        started = time.perf_counter()
        raw = self.detector.update_and_detect(
            datum,
            grayscale,
            apply_refraction_correction=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_timestamp = timestamp_s
        if not isinstance(raw, dict):
            return self._empty(pupil_confidence, "pye3d warming", elapsed_ms)

        return self._result(raw, pupil_confidence, elapsed_ms)

    # Returns eye center, pupil center, pupil diameter, pupil confidence, model confidence, and update time - function

    def _result(self, raw, pupil_confidence, elapsed_ms):
        eye_center = _finite_triplet(raw.get("sphere", {}).get("center"))
        pupil_center = _finite_triplet(raw.get("circle_3d", {}).get("center"))
        try:
            pupil_diameter = float(raw["diameter_3d"])
        except (KeyError, TypeError, ValueError):
            pupil_diameter = math.nan
        try:
            model_confidence = float(raw["model_confidence"])
        except (KeyError, TypeError, ValueError):
            model_confidence = None
        if model_confidence is not None and not math.isfinite(model_confidence):
            model_confidence = None

        valid_geometry = (
            eye_center is not None
            and pupil_center is not None
            and eye_center[2] > 0
            and pupil_center[2] > 0
            and math.isfinite(pupil_diameter)
            and pupil_diameter > 0
        )
        if not valid_geometry:
            return self._empty(
                pupil_confidence,
                "pye3d warming",
                elapsed_ms,
                model_confidence,
            )

        projected_sphere = _opencv_ellipse(raw.get("projected_sphere"))
        projected_sphere = _projected_ellipse_to_processed(
            projected_sphere,
            self.width,
            self.height,
            self.cx,
            self.cy,
        )
        projected_eye = projected_sphere[0] if projected_sphere is not None else None
        projected_pupil = _projected_point_to_processed(
            raw.get("location"),
            self.width,
            self.height,
            self.cx,
            self.cy,
        )

        display_sphere = (
            self.calibration.distort_ellipse(projected_sphere)
            if projected_sphere is not None
            else None
        )
        if projected_eye is not None:
            display_eye = tuple(
                float(value)
                for value in self.calibration.distort_points([projected_eye])[0]
            )
        else:
            display_eye = None
        if projected_pupil is not None:
            display_pupil = tuple(
                float(value)
                for value in self.calibration.distort_points([projected_pupil])[0]
            )
        else:
            display_pupil = None

        return EyeModelEstimate(
            ready=True,
            eye_center_mm=tuple(value * self.scale for value in eye_center),
            pupil_center_mm=tuple(value * self.scale for value in pupil_center),
            pupil_diameter_mm=pupil_diameter * self.scale,
            pupil_confidence=float(pupil_confidence),
            model_confidence=model_confidence,
            update_time_ms=float(elapsed_ms),
            projected_eye_sphere=display_sphere,
            projected_eye_center=display_eye,
            projected_pupil_center=display_pupil,
            status="ready",
        )
