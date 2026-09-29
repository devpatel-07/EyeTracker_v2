# Import dependencies

from dataclasses import dataclass
import math
import time

import numpy as np


# Stores normal-vector eye model outputs - data class

@dataclass(frozen=True)
class EyeModelEstimate:
    ready: bool

    # Kept for compatibility with the existing feature output code
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

    # Normal-vector outputs
    normal_vector: tuple[float, float, float] | None
    alternate_normal_vector: tuple[float, float, float] | None
    alternate_pupil_center_mm: tuple[float, float, float] | None
    camera_matrix: np.ndarray | None


# Checks that a value contains a valid XYZ triplet - function

def _finite_triplet(value):
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None

    if len(result) != 3 or not all(math.isfinite(component) for component in result):
        return None

    return result


# Converts a detected ellipse into its conic matrix - function

def _ellipse_conic_matrix(ellipse):
    try:
        (xc, yc), (width, height), angle = ellipse
        xc = float(xc)
        yc = float(yc)
        width = float(width)
        height = float(height)
        angle = float(angle)
    except (TypeError, ValueError) as error:
        raise ValueError("ellipse must use OpenCV fitEllipse format") from error

    values = (xc, yc, width, height, angle)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("ellipse values must be finite")

    if width <= 0 or height <= 0:
        raise ValueError("ellipse axes must be positive")

    phi = np.deg2rad(angle)

    a = width / 2.0
    b = height / 2.0

    cos_p = np.cos(phi)
    sin_p = np.sin(phi)

    A = (cos_p / a) ** 2 + (sin_p / b) ** 2
    B = 2.0 * cos_p * sin_p * (1.0 / (a ** 2) - 1.0 / (b ** 2))
    C = (sin_p / a) ** 2 + (cos_p / b) ** 2
    D = -2.0 * A * xc - B * yc
    E = -B * xc - 2.0 * C * yc
    F = A * xc ** 2 + B * xc * yc + C * yc ** 2 - 1.0

    return np.array(
        [
            [A, B / 2.0, D / 2.0],
            [B / 2.0, C, E / 2.0],
            [D / 2.0, E / 2.0, F],
        ],
        dtype=np.float64,
    )


# Calculates the two possible 3D pupil-plane normals from one pupil ellipse - function

def calculate_normal_vectors(ellipse, camera_matrix, pupil_radius_mm):
    Q = _ellipse_conic_matrix(ellipse)

    K = np.asarray(camera_matrix, dtype=np.float64)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError("camera matrix must be finite and 3x3")

    M = K.T @ Q @ K

    eigenvalues, V = np.linalg.eigh(M)

    # Match the ordering used in eye_pipeline_normal.py
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    l1, l2, l3 = eigenvalues

    denominator = l3 - l1
    if not math.isfinite(float(denominator)) or abs(float(denominator)) < 1e-12:
        return None

    c1_term = (l3 - l2) / denominator
    c3_term = (l2 - l1) / denominator

    c1 = np.sqrt(np.clip(c1_term, 0.0, 1.0))
    c3 = np.sqrt(np.clip(c3_term, 0.0, 1.0))

    n_prime_1 = np.array([c3, 0.0, c1], dtype=np.float64)
    n_prime_2 = np.array([-c3, 0.0, c1], dtype=np.float64)

    scale_term_denominator = -l1 * l3
    if (
        not math.isfinite(float(scale_term_denominator))
        or abs(float(scale_term_denominator)) < 1e-12
    ):
        return None

    scale_term = l2 / scale_term_denominator
    if not math.isfinite(float(scale_term)) or scale_term <= 0:
        return None

    scale = float(pupil_radius_mm) * np.sqrt(scale_term)

    solutions = []

    for n_prime in (n_prime_1, n_prime_2):
        n_cam = V @ n_prime

        normal_length = np.linalg.norm(n_cam)
        if not math.isfinite(float(normal_length)) or normal_length <= 0:
            return None

        n_cam = n_cam / normal_length

        # Same center calculation used by calculate_gaze_vector()
        c_prime = scale * np.array(
            [
                n_prime[0] / l1,
                0.0,
                n_prime[2] / l3,
            ],
            dtype=np.float64,
        )

        c_cam = V @ c_prime

        if c_cam[2] <= 0:
            c_cam = -c_cam

        # Point the normal toward the camera, matching the original function
        if n_cam[2] > 0:
            n_cam = -n_cam

        center = _finite_triplet(c_cam)
        normal = _finite_triplet(n_cam)

        if center is None or normal is None:
            return None

        solutions.append(
            {
                "center_3d": center,
                "normal_3d": normal,
            }
        )

    return tuple(solutions)


# Creates one normal-vector eye model estimator - class

class EyeNormalEstimator:
    def __init__(self, calibration, min_confidence, pupil_radius_mm):
        matrix = np.asarray(calibration.video_camera_matrix, dtype=np.float64)

        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera matrix must be finite and 3x3")

        self.calibration = calibration
        self.camera_matrix = matrix.copy()
        self.min_confidence = float(min_confidence)
        self.pupil_radius_mm = float(pupil_radius_mm)

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("minimum confidence must be between zero and one")

        if not math.isfinite(self.pupil_radius_mm) or self.pupil_radius_mm <= 0:
            raise ValueError("pupil radius must be finite and positive")

        self.last_timestamp = None

    def _empty(self, pupil_confidence, status, elapsed_ms=0.0):
        return EyeModelEstimate(
            ready=False,
            eye_center_mm=None,
            pupil_center_mm=None,
            pupil_diameter_mm=None,
            pupil_confidence=float(pupil_confidence),
            model_confidence=None,
            update_time_ms=float(elapsed_ms),
            projected_eye_sphere=None,
            projected_eye_center=None,
            projected_pupil_center=None,
            status=status,
            normal_vector=None,
            alternate_normal_vector=None,
            alternate_pupil_center_mm=None,
            camera_matrix=self.camera_matrix.copy(),
        )

    # Updates the normal-vector eye estimate - function

    def update(self, corrected_ellipse, pupil_confidence, timestamp_s, frame):
        # timestamp_s and frame are accepted so eye_pipeline.py can keep
        # the same update(...) call used by the pye3d estimator.
        pupil_confidence = float(pupil_confidence)
        timestamp_s = float(timestamp_s)

        if not math.isfinite(pupil_confidence) or not 0.0 <= pupil_confidence <= 1.0:
            raise ValueError("pupil confidence must be between zero and one")

        if not math.isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp must be finite and nonnegative")

        if corrected_ellipse is None:
            return self._empty(pupil_confidence, "no pupil")

        if pupil_confidence < self.min_confidence:
            return self._empty(pupil_confidence, "low pupil confidence")

        if self.last_timestamp is not None and timestamp_s <= self.last_timestamp:
            raise ValueError("timestamps must strictly increase")

        started = time.perf_counter()

        solutions = calculate_normal_vectors(
            corrected_ellipse,
            self.camera_matrix,
            self.pupil_radius_mm,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.last_timestamp = timestamp_s

        if solutions is None or len(solutions) != 2:
            return self._empty(
                pupil_confidence,
                "normal calculation failed",
                elapsed_ms,
            )

        first_solution = solutions[0]
        second_solution = solutions[1]

        return EyeModelEstimate(
            ready=True,
            eye_center_mm=None,
            pupil_center_mm=first_solution["center_3d"],
            pupil_diameter_mm=2.0 * self.pupil_radius_mm,
            pupil_confidence=pupil_confidence,
            model_confidence=None,
            update_time_ms=elapsed_ms,
            projected_eye_sphere=None,
            projected_eye_center=None,
            projected_pupil_center=None,
            status="normal ready",
            normal_vector=first_solution["normal_3d"],
            alternate_normal_vector=second_solution["normal_3d"],
            alternate_pupil_center_mm=second_solution["center_3d"],
            camera_matrix=self.camera_matrix.copy(),
        )


# Alias keeps the existing eye_pipeline.py class name compatible.
EyeModelEstimator = EyeNormalEstimator
