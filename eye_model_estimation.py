"""Adapt corrected pupil ellipses to a separate temporal pye3d model per eye.

Input ellipses use full, rotated-frame pixels after lens undistortion. pye3d
receives focal normalization and a principal-point shift; projections are restored and distorted
again for display on the original rotated image. Reported 3D lengths are scaled
to the configured eye radius and remain in each eye camera's coordinate system.
"""

from collections import deque
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


# Matches pye3d/constants.py::_EYE_RADIUS_DEFAULT; recheck when changing pye3d.
# https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/constants.py
PYE3D_REFERENCE_EYE_RADIUS_MM = 10.392304845413264


@dataclass(frozen=True)
class ModelDiagnostics:
    """Explain pye3d 0.3.2's output checks without changing its model updates.

    Native values are BEFORE our eye-radius scale. ``passed`` only means the
    upstream ranges passed; it is not a convergence or gaze-accuracy claim.
    Missing/nonfinite values become None, so dataclasses.asdict() can be logged
    as strict JSON. A skipped frame has no diagnostics rather than stale data.
    """

    range_status: str  # passed, failed, or unavailable
    failed_checks: tuple[str, ...]
    unavailable_checks: tuple[str, ...]
    native_eye_center_mm: tuple[float, float, float] | None
    native_pupil_diameter_mm: float | None
    phi_offset_deg: float | None
    theta_offset_deg: float | None


@dataclass(frozen=True)
class EyeModelEstimate:
    """One frame's model output, shared by console and overlay rendering.

    ``ready`` means finite, positive-depth geometry passed local checks; it does
    not certify convergence or require a minimum ``model_confidence``. Lengths
    ending in ``_mm`` depend on the assumed eye radius. Camera XYZ points right,
    down, and into the scene relative to the processed image. ``projected_*`` fields
    are distorted display pixels, with ellipse axes expressed as diameters.
    ``projected_pupil_center`` is pye3d's projected ellipse center, which can
    differ from projecting the physical 3D pupil center under perspective.
    Missing geometry is ``None``; ``status`` explains why it is unavailable.
    ``update_time_ms`` measures only the pye3d call, not total frame processing.
    """

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
    model_diagnostics: ModelDiagnostics | None = None
    # Optical/pupil normal in the processed eye camera, NOT a calibrated visual
    # axis or a direction in a world/scene camera. No eye-radius scaling applies.
    gaze_direction_camera: tuple[float, float, float] | None = None
    model_stability: "ModelStability | None" = None
    calibration_identity_status: str = "unverified"


@dataclass(frozen=True)
class Pye3DInputDecision:
    """State whether one corrected ellipse fits pye3d's input image.

    This check happens after lens correction and the principal-point shift.
    It is deliberately separate from pupil confidence and eyelid evidence:
    those are evaluated before calibration by ``FrameQualityDecision``.
    """

    allow_model_update: bool
    reason: str


def _finite_triplet(value):
    """Normalize a finite XYZ result, or return ``None`` for invalid geometry."""
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        return None
    return result


def diagnose_model_output(raw):
    """Explain the native output ranges used by Detector3D._prepare_result.

    ``raw`` is the dictionary returned by pye3d, not an EyeModelEstimate.
    Ranges are inclusive and version-specific; recheck them on a pye3d upgrade:
    https://github.com/pupil-labs/pye3d-detector/blob/master/pye3d/detector_3d.py
    These diagnostics never reject an input or stop an eligible model update.
    """
    def finite(value):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return value if math.isfinite(value) else None

    sphere = raw.get("sphere")
    circle = raw.get("circle_3d")
    center = _finite_triplet(sphere.get("center")) if isinstance(sphere, dict) else None
    normal = _finite_triplet(circle.get("normal")) if isinstance(circle, dict) else None
    diameter = finite(raw.get("diameter_3d"))
    failed, unavailable = [], []

    def check(name, value, lower, upper):
        if value is None:
            unavailable.append(name)
        elif not lower <= value <= upper:
            failed.append(name)

    for index, (name, lower, upper) in enumerate((
        ("eye_center_x", -15.0, 15.0),
        ("eye_center_y", -10.0, 10.0),
        ("eye_center_z", 15.0, 75.0),
    )):
        check(name, None if center is None else center[index], lower, upper)
    check("pupil_diameter", diameter, 1.0, 9.0)

    # pye3d writes phi=theta=0 when its normal yields NaN angles. Inspect the
    # normal as well, so those placeholders cannot masquerade as real angles.
    normal_length = None if normal is None else math.hypot(*normal)
    phi_offset = theta_offset = None
    if normal_length is None or not math.isfinite(normal_length) or normal_length == 0:
        unavailable.append("pupil_normal")
    else:
        phi, theta = finite(raw.get("phi")), finite(raw.get("theta"))
        phi_offset = None if phi is None else finite(math.degrees(phi) + 90.0)
        theta_offset = None if theta is None else finite(math.degrees(theta) - 90.0)
    check("gaze_phi", phi_offset, -90.0, 90.0)
    check("gaze_theta", theta_offset, -80.0, 80.0)
    status = "failed" if failed else "unavailable" if unavailable else "passed"
    return ModelDiagnostics(status, tuple(failed), tuple(unavailable), center,
                            diameter, phi_offset, theta_offset)


def _finite_point(value):
    """Normalize a finite pixel coordinate pair without rounding for display."""
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if len(result) != 2 or not all(math.isfinite(component) for component in result):
        return None
    return result


def _opencv_ellipse(value):
    """Validate a pye3d ellipse dictionary and unpack it for OpenCV drawing."""
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


def _focal_scales(fx=None, fy=None):
    """Map calibrated pixels to pye3d's single-focal-length virtual camera.

    In a pinhole camera x = fx*X/Z + cx. Multiplying (x-cx) by f/fx
    therefore preserves the camera ray while replacing fx with f (and y
    likewise). Omitting both focal lengths preserves older centered-pixel calls.
    """
    if fx is None and fy is None:
        return 1.0, 1.0
    try:
        fx, fy = float(fx), float(fy)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("both focal lengths must be finite and positive") from error
    if not all(math.isfinite(value) and value > 0 for value in (fx, fy)):
        raise ValueError("both focal lengths must be finite and positive")
    focal = fx / 2 + fy / 2
    return focal / fx, focal / fy


def _scaled_ellipse_shape(axes, angle, scale_x, scale_y):
    """Transform ellipse axes exactly under an x/y scale, including its angle.

    Scaling the two diameters alone is wrong for a tilted ellipse. Its shape
    matrix Q = R diag(radius**2) R.T describes every rotated boundary point;
    A Q A.T applies the pixel transform. Eigenvectors give the new axis
    directions and square roots of eigenvalues give the new radii.
    """
    if scale_x == scale_y:
        return tuple(float(value * scale_x) for value in axes), angle % 180.0
    radians = math.radians(angle)
    rotation = np.array([[math.cos(radians), -math.sin(radians)],
                         [math.sin(radians), math.cos(radians)]])
    transform = np.diag([scale_x, scale_y]) @ rotation
    shape = transform @ np.diag((np.asarray(axes) / 2.0) ** 2) @ transform.T
    values, vectors = np.linalg.eigh(shape)  # ascending: minor axis first
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("scaled ellipse is degenerate")
    diameters = tuple(float(value) for value in 2 * np.sqrt(values))
    orientation = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0])) % 180.0
    return diameters, orientation


def _pye3d_ellipse(ellipse, width, height, principal_x, principal_y, fx=None, fy=None):
    """Convert an undistorted OpenCV ellipse into pye3d's input dictionary.

    Axes are full diameters in pixels, ordered minor then major. Focal scaling
    preserves camera rays even when fx != fy; the center shift compensates for
    pye3d treating the image midpoint as principal point.
    This matches Detector3D._extract_observation's input convention.
    """
    try:
        (center_x, center_y), (axis_0, axis_1), angle = ellipse
        center_x, center_y, axis_0, axis_1, angle = map(
            float,
            (center_x, center_y, axis_0, axis_1, angle),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("ellipse must use OpenCV fitEllipse format") from error
    if not all(
        math.isfinite(value)
        for value in (center_x, center_y, axis_0, axis_1, angle)
    ) or axis_0 <= 0 or axis_1 <= 0:
        raise ValueError("ellipse must contain finite positive axes")
    if axis_0 > axis_1:
        # Swapping axis lengths requires rotating the axis direction as well.
        axis_0, axis_1 = axis_1, axis_0
        angle += 90.0

    scale_x, scale_y = _focal_scales(fx, fy)
    axes, angle = _scaled_ellipse_shape((axis_0, axis_1), angle, scale_x, scale_y)
    # pye3d subtracts the midpoint internally, leaving f*(X/Z, Y/Z).
    model_center = (
        (center_x - principal_x) * scale_x + width / 2.0,
        (center_y - principal_y) * scale_y + height / 2.0,
    )
    return {
        "center": model_center,
        "axes": axes,
        "angle": angle % 180.0,
    }


def assess_pye3d_input_geometry(
    corrected_ellipse,
    width,
    height,
    principal_x,
    principal_y,
    fx=None,
    fy=None,
):
    """Check a corrected ellipse in the exact image coordinates pye3d receives.

    pye3d subtracts half the image size from the supplied ellipse center. This
    adapter scales coordinates to one focal length and shifts the center so
    pye3d's midpoint acts like the calibrated principal point. That center must lie
    in the half-open image rectangle ``[0, width) x [0, height)``. Otherwise
    pye3d's binned long-term stores calculate an invalid bin and ignore the
    observation after it has already entered the detector.

    This check intentionally does not require the complete ellipse outline to
    fit inside the image. A partly occluded pupil can still have a meaningful
    inferred ellipse, and the current evidence supports only a center-domain
    requirement. Confidence, eyelid evidence, and temporal recovery are handled
    by ``pupil_detection.py``.
    """
    try:
        width = int(width)
        height = int(height)
        principal_x = float(principal_x)
        principal_y = float(principal_y)
    except (TypeError, ValueError, OverflowError):
        return Pye3DInputDecision(False, "invalid pye3d camera geometry")
    if (
        width <= 0
        or height <= 0
        or not math.isfinite(principal_x)
        or not math.isfinite(principal_y)
    ):
        return Pye3DInputDecision(False, "invalid pye3d camera geometry")

    try:
        converted = _pye3d_ellipse(
            corrected_ellipse,
            width,
            height,
            principal_x,
            principal_y,
            fx,
            fy,
        )
    except ValueError:
        return Pye3DInputDecision(False, "invalid corrected pupil geometry")

    center_x, center_y = converted["center"]
    if not (0.0 <= center_x < width and 0.0 <= center_y < height):
        return Pye3DInputDecision(
            False,
            "corrected pupil center outside pye3d image",
        )
    return Pye3DInputDecision(True, "corrected pupil geometry passed")


def _projected_point_to_processed(point, width, height, principal_x, principal_y,
                                  fx=None, fy=None):
    """Undo focal normalization and the principal-point shift before redistortion."""
    point = _finite_point(point)
    if point is None:
        return None
    scale_x, scale_y = _focal_scales(fx, fy)
    return (
        (point[0] - width / 2.0) / scale_x + principal_x,
        (point[1] - height / 2.0) / scale_y + principal_y,
    )


def _projected_ellipse_to_processed(
    ellipse,
    width,
    height,
    principal_x,
    principal_y,
    fx=None,
    fy=None,
):
    """Restore a projected ellipse to calibrated undistorted image pixels."""
    if ellipse is None:
        return None
    center, axes, angle = ellipse
    center = _projected_point_to_processed(
        center,
        width,
        height,
        principal_x,
        principal_y,
        fx,
        fy,
    )
    scale_x, scale_y = _focal_scales(fx, fy)
    axes, angle = _scaled_ellipse_shape(axes, angle, 1 / scale_x, 1 / scale_y)
    return (center, axes, angle) if center is not None else None


class EyeModelEstimator:
    """Own pye3d history for one eye and one continuous video sequence.

    Create separate instances for left/right videos. Recreate the estimator
    when restarting a sequence or changing its camera calibration.
    """

    def __init__(self, calibration, min_confidence, eye_radius_mm):
        """Configure intrinsics, accepted pupil confidence, and assumed mm scale."""
        try:
            from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
        except ImportError as error:
            raise RuntimeError("pye3d is not installed") from error

        # pye3d 0.3.2 passes one-dimensional vectors to cv2.KalmanFilter.
        # OpenCV 5 changed those arrays from implicit column matrices to true
        # one-dimensional matrices, so the second model update fails inside
        # cv2.gemm. requirements.txt pins the last locally validated 4.x build.
        try:
            opencv_major = int(cv2.__version__.split(".", 1)[0])
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError("could not determine the installed OpenCV version") from error
        if opencv_major >= 5:
            raise RuntimeError(
                "pye3d 0.3.2 is incompatible with OpenCV 5; "
                "install the versions pinned in requirements.txt"
            )

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
        # This changes reported lengths only; pye3d still fits its fixed radius.
        self.scale = float(eye_radius_mm) / PYE3D_REFERENCE_EYE_RADIUS_MM
        if not 0.0 < self.min_confidence <= 1.0:
            # Confidence zero enters pye3d's image-search fallback. Our image
            # pixels are not transformed to the virtual camera, so disallow
            # configurations that could send an observation down that path.
            raise ValueError("minimum confidence must be positive and at most one")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("eye radius must be finite and positive")

        # CameraModel takes one focal length. The ellipse adapter maps BOTH
        # coordinates and its full shape to this virtual camera, preserving rays.
        _focal_scales(self.fx, self.fy)  # fail early for invalid intrinsics
        camera = CameraModel(
            focal_length=(self.fx + self.fy) / 2.0,
            resolution=(self.width, self.height),
        )
        self.detector = Detector3D(
            camera=camera,
            # pye3d uses a strict > comparison for its ellipse-based prediction.
            # The small offset normally keeps accepted samples on that path.
            threshold_swirski=max(0.0, self.min_confidence - 1e-6),
            threshold_short_term=self.min_confidence,
            threshold_long_term=self.min_confidence,
            # Complete model fitting within this frame's call before returning.
            long_term_mode=DetectorMode.blocking,
        )
        self.last_timestamp = None

    def assess_input_geometry(self, corrected_ellipse):
        """Return the post-calibration decision used before temporal gating."""
        return assess_pye3d_input_geometry(
            corrected_ellipse,
            self.width,
            self.height,
            self.cx,
            self.cy,
            self.fx,
            self.fy,
        )

    def _empty(
        self,
        pupil_confidence,
        status,
        elapsed_ms=0.0,
        model_confidence=None,
        model_diagnostics=None,
    ):
        """Represent a skipped or unusable update without reusing old geometry."""
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
            model_diagnostics=model_diagnostics,
            calibration_identity_status=getattr(self.calibration, "identity_status", "unverified"),
        )

    def skip_update(self, pupil_confidence, quality_state, reason):
        """Return an explicit skipped result without touching pye3d history.

        The temporal quality gate calls this for rejected/recovering frames.
        No ellipse is submitted, ``update_and_detect`` is not called, and
        ``last_timestamp`` is intentionally unchanged. The next accepted frame
        therefore enters pye3d with its real later video timestamp.
        """
        try:
            confidence = float(pupil_confidence)
        except (TypeError, ValueError, OverflowError):
            confidence = math.nan
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            confidence = math.nan
        state = str(quality_state).strip() or "rejected"
        explanation = str(reason).strip() or "quality gate rejected frame"
        return self._empty(
            confidence,
            f"quality {state}: {explanation}",
        )

    def update(self, corrected_ellipse, pupil_confidence, timestamp_s, frame):
        """Consume a pupil observation and return this frame's estimate.

        ``timestamp_s`` is video time in seconds. Accepted updates must advance
        beyond the last submitted timestamp. Missing/low-confidence pupils do
        not update pye3d or advance that timestamp, preserving its prior history.
        ``frame`` is the full rotated uint8 image, not the detection ROI.
        """
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

        # The pipeline normally converts this failure into a temporal quality
        # rejection before calling update(). Keep the adapter defensive so a
        # different caller cannot reintroduce pye3d's out-of-bounds bin warning.
        input_decision = self.assess_input_geometry(corrected_ellipse)
        if not input_decision.allow_model_update:
            raise ValueError(f"pye3d input rejected: {input_decision.reason}")

        # Only ellipse geometry is undistorted and principal-point shifted;
        # these image pixels remain in the original processed-frame coordinates.
        # Revisit image alignment if enabling pye3d's image-based fallback.
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
            self.fx,
            self.fy,
        )
        datum = {
            "ellipse": ellipse,
            "diameter": ellipse["axes"][1],
            "location": ellipse["center"],
            "confidence": pupil_confidence,
            "timestamp": timestamp_s,
            # Pupil-style normalized image coordinates have an upward y axis.
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
            # Lens distortion was handled earlier; this separately disables
            # pye3d's correction for refraction at the eye's cornea.
            apply_refraction_correction=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_timestamp = timestamp_s
        if not isinstance(raw, dict):
            return self._empty(pupil_confidence, "pye3d warming", elapsed_ms)

        return self._result(raw, pupil_confidence, elapsed_ms)

    def _result(self, raw, pupil_confidence, elapsed_ms):
        """Validate pye3d geometry, restore display coordinates, and scale lengths."""
        diagnostics = diagnose_model_output(raw)
        eye_center = diagnostics.native_eye_center_mm
        circle = raw.get("circle_3d")
        pupil_center = _finite_triplet(circle.get("center")) if isinstance(circle, dict) else None
        pupil_diameter = diagnostics.native_pupil_diameter_mm
        normal = _finite_triplet(circle.get("normal")) if isinstance(circle, dict) else None
        normal_length = None if normal is None else math.hypot(*normal)
        direction = (tuple(value / normal_length for value in normal)
                     if normal_length is not None and math.isfinite(normal_length)
                     and normal_length > 0 else None)
        try:
            model_confidence = float(raw["model_confidence"])
        except (KeyError, TypeError, ValueError, OverflowError):
            model_confidence = None
        if model_confidence is not None and not math.isfinite(model_confidence):
            model_confidence = None

        # This is a geometry availability check, not a model-quality threshold.
        # In particular, a low or absent model_confidence can still be ready.
        valid_geometry = (
            eye_center is not None
            and pupil_center is not None
            and eye_center[2] > 0
            and pupil_center[2] > 0
            and pupil_diameter is not None
            and pupil_diameter > 0
        )
        if not valid_geometry:
            return self._empty(
                pupil_confidence,
                "pye3d warming",
                elapsed_ms,
                model_confidence,
                diagnostics,
            )

        projected_sphere = _opencv_ellipse(raw.get("projected_sphere"))
        projected_sphere = _projected_ellipse_to_processed(
            projected_sphere,
            self.width,
            self.height,
            self.cx,
            self.cy,
            self.fx,
            self.fy,
        )
        projected_eye = projected_sphere[0] if projected_sphere is not None else None
        projected_pupil = _projected_point_to_processed(
            raw.get("location"),
            self.width,
            self.height,
            self.cx,
            self.cy,
            self.fx,
            self.fy,
        )

        # The displayed frame retains lens distortion, so overlays need the
        # forward distortion mapping after undoing pye3d's center shift.
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

        # A uniform scale of sphere position and radius leaves its projection
        # unchanged, so only 3D lengths (not display pixels) are multiplied.
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
            model_diagnostics=diagnostics,
            gaze_direction_camera=direction,
            calibration_identity_status=getattr(self.calibration, "identity_status", "unverified"),
        )


@dataclass(frozen=True)
class ModelStability:
    """A causal consistency diagnostic, not measured gaze accuracy.

    Centers and distances use native pye3d millimeters BEFORE the configured
    eye-radius scale. ``reference_center_mm`` is a fixed, early stable-window
    median; keeping it fixed makes gradual drift visible. ``recent_spread_mm``
    is the 90th percentile distance of recent centers from their median, which
    reduces the influence of a brief spike. A stable but systematically wrong
    model can pass these checks. A skipped frame reports no cached measurements.

    ``window_duration_s`` is the elapsed time between first and last samples.
    ``observed_duration_s`` excludes intervals adjacent to skipped frames, so a
    period without observations cannot satisfy the required observation time.
    """

    status: str  # warming_up, insufficient_history, stable, drift_detected,
    # unstable, missing, or data_gap
    reference_center_mm: tuple[float, float, float] | None = None
    recent_center_mm: tuple[float, float, float] | None = None
    displacement_mm: float | None = None
    recent_spread_mm: float | None = None
    window_duration_s: float = 0.0
    observed_duration_s: float = 0.0
    sample_count: int = 0
    reference_timestamp_s: float | None = None


class ModelStabilityMonitor:
    """Track per-eye native eye-center consistency with bounded memory.

    Defaults (5 s warmup, 2 s observation window, 20 samples, 2 mm reference
    displacement, 1 mm recent spread, 0.5 s maximum gap) are configurable
    ENGINEERING starting points, not published or validated convergence/gaze
    thresholds. They must be evaluated against camera-specific target data.

    The first five seconds of each uninterrupted segment are excluded from
    baseline collection. Only a sufficiently sampled, low-spread window can
    establish the reference. It remains fixed until reset() or a gap exceeding
    max_gap_seconds. After a gap the new segment must warm up again. A stable
    result means these consistency thresholds passed, not pye3d's anatomical
    range checks, calibrated gaze accuracy, or proof of model convergence.

    The time window retains one sample at/before its start to measure coverage;
    its extent can exceed window_seconds by at most max_gap_seconds. The hard
    max_samples bound protects memory at unusually high frame rates. If that
    limit prevents sufficient coverage, status remains insufficient_history.
    """

    def __init__(self, startup_seconds=5.0, window_seconds=2.0,
                 min_samples=20, max_drift_mm=2.0, max_spread_mm=1.0,
                 max_gap_seconds=0.5, max_samples=1000):
        values = {
            "startup_seconds": startup_seconds,
            "window_seconds": window_seconds,
            "max_drift_mm": max_drift_mm,
            "max_spread_mm": max_spread_mm,
            "max_gap_seconds": max_gap_seconds,
        }
        for name, value in values.items():
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite number") from exc
            if not math.isfinite(number) or number < 0 or (
                name != "startup_seconds" and number == 0
            ):
                raise ValueError(f"{name} must be finite and positive"
                                 " (startup_seconds may be zero)")
            setattr(self, name, number)
        if (isinstance(min_samples, bool) or not isinstance(min_samples, int)
                or min_samples < 2):
            raise ValueError("min_samples must be an integer of at least two")
        if (isinstance(max_samples, bool) or not isinstance(max_samples, int)
                or max_samples < min_samples):
            raise ValueError("max_samples must be an integer >= min_samples")
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.reset()

    def reset(self):
        """Forget timestamps, history and reference for a new recording."""
        self._last_timestamp = None
        self._gap_pending = False
        self._clear_segment()

    def _clear_segment(self):
        """Restart observed history without accepting a stale baseline."""
        self._samples = deque(maxlen=self.max_samples)
        self._segment_start = None
        self._last_center_timestamp = None
        self._previous_frame_had_center = False
        self._observed_time = 0.0
        self._reference = None
        self._reference_timestamp = None

    def update(self, native_eye_center_mm, timestamp_s):
        """Assess one frame; use None when its geometry is unavailable.

        ``timestamp_s`` is video time, finite/nonnegative/strictly increasing
        even for skipped frames. Invalid timestamps raise before any state
        changes. Centers must contain three finite values with positive depth;
        malformed centers follow the same missing-data path as None. A warning
        from this method must never suppress a later eligible pye3d update.
        """
        try:
            timestamp = float(timestamp_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("stability timestamp must be finite/nonnegative") from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("stability timestamp must be finite/nonnegative")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("stability timestamps must strictly increase")

        try:
            center = tuple(float(value) for value in native_eye_center_mm)
        except (TypeError, ValueError, OverflowError):
            center = None
        if center is not None and (len(center) != 3
                or not all(math.isfinite(value) for value in center)
                or center[2] <= 0):
            center = None

        previous_timestamp = self._last_timestamp
        self._last_timestamp = timestamp
        if (self._last_center_timestamp is not None
                and timestamp - self._last_center_timestamp > self.max_gap_seconds):
            self._clear_segment()
            self._gap_pending = True

        if center is None:
            self._previous_frame_had_center = False
            # Do not return an old reference, recent center, or displacement as
            # if it were measured on this frame; internal history may survive.
            return ModelStability("data_gap" if self._gap_pending else "missing")

        if self._segment_start is None:
            self._segment_start = timestamp
        if self._previous_frame_had_center:
            self._observed_time += timestamp - previous_timestamp
        self._previous_frame_had_center = True
        self._last_center_timestamp = timestamp
        restarted = self._gap_pending
        self._gap_pending = False

        if timestamp - self._segment_start < self.startup_seconds:
            return ModelStability("data_gap" if restarted else "warming_up")

        self._samples.append((timestamp, center, self._observed_time))
        cutoff = timestamp - self.window_seconds
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        duration = timestamp - self._samples[0][0]
        observed = self._observed_time - self._samples[0][2]
        count = len(self._samples)
        common = dict(window_duration_s=duration,
                      observed_duration_s=observed, sample_count=count)
        if restarted:
            return ModelStability("data_gap", **common)
        if count < self.min_samples or observed + 1e-9 < self.window_seconds:
            return ModelStability("insufficient_history", **common)

        centers = np.asarray([sample[1] for sample in self._samples], dtype=float)
        median = np.median(centers, axis=0)
        recent = tuple(float(value) for value in median)
        distances = np.linalg.norm(centers - median, axis=1)
        spread = float(np.percentile(distances, 90))
        # Do not establish the fixed reference from a still-moving fit.
        if self._reference is None and spread <= self.max_spread_mm:
            self._reference = recent
            self._reference_timestamp = timestamp
        displacement = (None if self._reference is None
                        else math.dist(self._reference, recent))
        status = ("drift_detected" if displacement is not None
                  and displacement > self.max_drift_mm else
                  "unstable" if spread > self.max_spread_mm else "stable")
        return ModelStability(
            status, reference_center_mm=self._reference,
            recent_center_mm=recent, displacement_mm=displacement,
            recent_spread_mm=spread,
            reference_timestamp_s=self._reference_timestamp, **common)
