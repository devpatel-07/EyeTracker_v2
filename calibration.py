"""Keep camera calibration and pupil geometry in compatible image coordinates.

There are three full-image coordinate frames: ``raw`` is the orientation used
to calibrate the camera, ``source`` is the decoded video, and ``processed`` is
the source after the pipeline's frame rotation. ``calibration_rotation`` maps
raw to source; ``frame_rotation`` maps source to processed. Neither rotation
includes an ROI crop or image resize.

Sizes are (width, height); pixel points are (x, y), with x right and y down.
Ellipses use OpenCV's ((cx, cy), (axis_width, axis_height), angle_degrees)
format: the two axis lengths are full diameters in pixels, not semiaxes.
Distortion is always evaluated in the original calibration frame, then the
result is rotated back for the model or display. Images themselves are not
undistorted here.
"""

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone

import cv2
import numpy as np


VALID_ROTATIONS = ("none", "clockwise", "180", "counterclockwise")
SUPPORTED_DISTORTION_LENGTHS = (4, 5, 8, 12, 14)


def _validated_image_size(size):
    """Require an exact positive (width, height), never silently round a size."""
    values = np.asarray(size, dtype=np.float64)
    if (values.shape != (2,) or not np.all(np.isfinite(values))
            or np.any(values <= 0) or np.any(values != np.floor(values))):
        raise ValueError("calibration image size must be two positive integers")
    return tuple(int(value) for value in values)


def _validate_intrinsics(matrix, distortion):
    """Validate the zero-skew OpenCV pinhole model used by all point transforms."""
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera matrix must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths fx and fy must be positive")
    if (not np.allclose(matrix[2], (0, 0, 1), rtol=0, atol=1e-12)
            or abs(matrix[0, 1]) > 1e-12 or abs(matrix[1, 0]) > 1e-12):
        raise ValueError("camera matrix must use the standard zero-skew pinhole form")
    vector_shape = distortion.ndim == 1 or (
        distortion.ndim == 2 and 1 in distortion.shape
    )
    if (not vector_shape or distortion.size not in SUPPORTED_DISTORTION_LENGTHS
            or not np.all(np.isfinite(distortion))):
        raise ValueError("distortion must be a finite vector with 4, 5, 8, 12, or 14 values")


def _camera_identifier(value):
    """A device label is required when creating new calibration artifacts."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("camera_id must be a nonempty camera label")
    return value.strip()

_ROTATION_QUARTER_TURNS = {
    "none": 0,
    "clockwise": 1,
    "180": 2,
    "counterclockwise": 3,
}
_QUARTER_TURN_ROTATIONS = {
    turns: name for name, turns in _ROTATION_QUARTER_TURNS.items()
}


def combined_rotation(calibration_rotation, frame_rotation):
    """Compose raw-to-source and source-to-processed quarter-turn rotations."""
    _check_rotation(calibration_rotation)
    _check_rotation(frame_rotation)
    # Rotations are clockwise quarter-turn counts; four turns return to identity.
    turns = (
        _ROTATION_QUARTER_TURNS[calibration_rotation]
        + _ROTATION_QUARTER_TURNS[frame_rotation]
    ) % 4
    return _QUARTER_TURN_ROTATIONS[turns]


def _check_rotation(rotation):
    """Reject names that cannot be represented by an exact image quarter turn."""
    if rotation not in VALID_ROTATIONS:
        choices = ", ".join(VALID_ROTATIONS)
        raise ValueError(f"unsupported rotation {rotation!r}; use one of {choices}")


def _rotated_size(size, rotation):
    """Return (width, height) after rotation, swapping dimensions for 90 degrees."""
    _check_rotation(rotation)
    width, height = (int(value) for value in size)
    if rotation in {"clockwise", "counterclockwise"}:
        return height, width
    return width, height


def _raw_to_video_points(points, raw_size, rotation):
    """Map raw pixels to a rotated full image as an N-by-2 float64 array.

    Callers supply a validated rotation. The width/height minus one terms rotate
    pixel-center indices (0 through size - 1), matching the frame rotation.
    """
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
    """Undo ``_raw_to_video_points`` using the original, unrotated dimensions."""
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


def _rotated_camera_matrix(camera_matrix, raw_size, rotation):
    """Express focal lengths and principal point in the rotated image frame.

    A quarter turn swaps fx/fy and rotates (cx, cy). This uses the standard
    zero-skew pinhole matrix; nonzero skew is not preserved by rotated cases.
    Distortion coefficients stay in the raw frame and are handled separately.
    """
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
    """Sample an OpenCV ellipse boundary uniformly in its angular parameter.

    At least five points are required for the later ellipse fit. The input axes
    are diameters, so halve them when forming local boundary coordinates.
    """
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
    # Exclude the endpoint so the boundary does not repeat its first sample.
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
    """Intrinsics plus orientation metadata shared by model and display code.

    Focal lengths, principal points, and image dimensions use pixels; lens
    distortion coefficients are dimensionless. ``load`` checks file contents;
    constructing the dataclass directly bypasses those checks. Frozen fields
    prevent reassignment, but callers must also treat the NumPy arrays as
    read-only to avoid changing calibration during a run.
    """

    raw_camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    raw_image_size: tuple[int, int]
    calibration_rotation: str
    frame_rotation: str
    camera_id: str | None = None
    calibration_source: str = "unverified legacy file"
    identity_status: str = "unverified"
    metadata: dict = field(default_factory=dict, compare=False)

    @classmethod
    def load(cls, path, calibration_rotation="none", frame_rotation="none",
             expected_camera_id=None, require_identity=False):
        """Read camera_matrix, dist_coeffs, and image_size from an NPZ file.

        ``image_size`` must describe the raw calibration image as (width,
        height). ``expected_camera_id`` binds a labeled artifact to the configured
        device; an explicit mismatch is always an error. Legacy files remain
        usable but unverified unless ``require_identity`` is enabled. A matching
        label verifies bookkeeping, not physical accuracy or data ownership.
        """
        _check_rotation(calibration_rotation)
        _check_rotation(frame_rotation)
        if expected_camera_id is not None:
            expected_camera_id = _camera_identifier(expected_camera_id)
        calibration_path = Path(path).expanduser()
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"camera calibration file not found: {calibration_path}"
            )
        try:
            with np.load(calibration_path, allow_pickle=False) as data:
                matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
                distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
                raw_size = _validated_image_size(data["image_size"])
                camera_id = (
                    _camera_identifier(data["camera_id"].item())
                    if "camera_id" in data else None
                )
                metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
                if not isinstance(metadata, dict):
                    raise ValueError("calibration metadata must be a JSON object")
                if "camera_id" in metadata and metadata["camera_id"] != camera_id:
                    raise ValueError("calibration metadata camera_id disagrees with artifact camera_id")
                source = str(metadata.get("source", "unverified legacy file"))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"could not load camera calibration {calibration_path}: {error}"
            ) from error

        _validate_intrinsics(matrix, distortion)
        if camera_id is not None and expected_camera_id is not None and camera_id != expected_camera_id:
            raise ValueError(f"calibration camera_id {camera_id!r} does not match expected {expected_camera_id!r}")
        if require_identity and (camera_id is None or expected_camera_id is None):
            raise ValueError("verified calibration identity requires file camera_id and expected_camera_id")
        identity_status = "unverified" if camera_id is None else (
            "matched" if expected_camera_id is not None else "declared"
        )
        return cls(
            matrix,
            distortion,
            raw_size,
            calibration_rotation,
            frame_rotation,
            camera_id,
            source,
            identity_status,
            metadata,
        )

    @property
    def processed_rotation(self):
        """The total rotation from the raw calibration image to processed video."""
        return combined_rotation(self.calibration_rotation, self.frame_rotation)

    @property
    def source_video_size(self):
        """Expected decoded-video (width, height), before pipeline rotation."""
        return _rotated_size(self.raw_image_size, self.calibration_rotation)

    @property
    def video_size(self):
        """Expected full processed-frame (width, height), not the pupil ROI size."""
        return _rotated_size(self.raw_image_size, self.processed_rotation)

    @property
    def video_camera_matrix(self):
        """Pinhole intrinsics for undistorted points in processed pixel coordinates."""
        return _rotated_camera_matrix(
            self.raw_camera_matrix,
            self.raw_image_size,
            self.processed_rotation,
        )

    def validate_video_dimensions(self, source_size, processed_size, side):
        """Require source and processed dimensions to match the rotation metadata.

        No intrinsic rescaling or crop offset is inferred. Size checks alone
        cannot distinguish opposite rotations with the same dimensions; the
        rotation preview must be used to choose the intended orientation.
        ``side`` labels errors for the pipeline's left and right video streams.
        """
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
        """Rotate full-image pixel points; preserve their current distortion state."""
        return _raw_to_video_points(
            points,
            self.raw_image_size,
            self.processed_rotation,
        )

    def processed_to_raw_points(self, points):
        """Undo rotation only; input points must already include any ROI offset."""
        return _video_to_raw_points(
            points,
            self.raw_image_size,
            self.processed_rotation,
        )

    def undistort_points(self, points):
        """Correct distorted processed pixels and return processed pixel positions."""
        # Lens coefficients belong to the raw orientation, including tangential
        # terms, so rotate points back before applying the calibration.
        raw_points = self.processed_to_raw_points(points)
        corrected_raw = cv2.undistortPoints(
            raw_points.reshape(-1, 1, 2),
            self.raw_camera_matrix,
            self.distortion_coefficients,
            # P retains pixel units instead of returning normalized camera rays.
            P=self.raw_camera_matrix,
        ).reshape(-1, 2)
        return self.raw_to_processed_points(corrected_raw)

    def distort_points(self, points):
        """Map ideal processed pixels onto the original, distorted display image."""
        corrected_raw = self.processed_to_raw_points(points)
        fx = self.raw_camera_matrix[0, 0]
        fy = self.raw_camera_matrix[1, 1]
        cx = self.raw_camera_matrix[0, 2]
        cy = self.raw_camera_matrix[1, 2]
        # Construct rays at arbitrary unit depth. projectPoints reapplies the
        # lens distortion; these temporary XYZ values are not eye geometry in mm.
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

    def undistort_ellipse(self, ellipse, samples=72):
        """Approximate an undistorted pupil boundary with a refitted ellipse.

        Distortion is nonlinear, so correcting only the center and axes is
        insufficient. Sample the boundary and fit again in processed pixels;
        the corrected boundary need not be an exact ellipse. Preserve None to
        let a missing pupil pass through the pipeline without special handling.
        """
        if ellipse is None:
            return None
        corrected = self.undistort_points(_ellipse_points(ellipse, samples))
        return cv2.fitEllipse(corrected.astype(np.float32).reshape(-1, 1, 2))

    def distort_ellipse(self, ellipse, samples=72):
        """Refit ideal projected geometry for overlay on the distorted frame.

        As with undistort_ellipse, the boundary fit is an approximation, so the
        two ellipse methods are not exact inverses. None remains None.
        """
        if ellipse is None:
            return None
        distorted = self.distort_points(_ellipse_points(ellipse, samples))
        return cv2.fitEllipse(distorted.astype(np.float32).reshape(-1, 1, 2))


def checkerboard_object_points(board_shape, square_size_mm):
    """Return the inner-corner grid in board coordinates, in millimeters.

    ``board_shape=(columns, rows)`` counts interior intersections, not squares:
    a board with 10 by 7 squares has 9 by 6 inner corners. Coordinates use the
    top-left inner corner as origin and a flat board at Z=0.
    """
    columns, rows = _validated_image_size(board_shape)
    if columns < 3 or rows < 3:
        raise ValueError("checkerboard needs at least 3 by 3 inner corners")
    square_size_mm = float(square_size_mm)
    if not math.isfinite(square_size_mm) or square_size_mm <= 0:
        raise ValueError("square_size_mm must be finite and positive")
    points = np.zeros((rows * columns, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points *= square_size_mm
    return points


def _checkerboard_constraint_singular_values(corners, object_points, image_size):
    """Check that the views constrain intrinsics, using planar homographies.

    A checkerboard only translated while facing the camera can yield many
    distinct images but still fail to constrain focal length. Each tilted view
    contributes two independent constraints on the symmetric intrinsic matrix
    (Zhang's planar calibration method). Five independent constraints are needed
    up to overall scale. Normalize pixels here solely for numerical conditioning;
    calibration itself still uses the original, full-resolution pixels. This
    catches degenerate pose sets; noisy near-degenerate data can still pass, so
    it does not certify adequate capture coverage or accurate lens correction.
    """
    width, height = image_size
    object_xy = object_points[:, :2].astype(np.float64)
    object_xy /= max(np.ptp(object_xy[:, 0]), np.ptp(object_xy[:, 1]))

    def constraint(h, i, j):
        return np.array((h[0, i] * h[0, j],
                         h[0, i] * h[1, j] + h[1, i] * h[0, j],
                         h[1, i] * h[1, j],
                         h[2, i] * h[0, j] + h[0, i] * h[2, j],
                         h[2, i] * h[1, j] + h[1, i] * h[2, j],
                         h[2, i] * h[2, j]))

    constraints = []
    for points in corners:
        normalized = points.astype(np.float64) / np.array((width, height))
        homography, _ = cv2.findHomography(object_xy, normalized, method=0)
        if homography is None or not np.all(np.isfinite(homography)):
            raise ValueError("checkerboard view has degenerate corner geometry")
        for row in (constraint(homography, 0, 1),
                    constraint(homography, 0, 0) - constraint(homography, 1, 1)):
            norm = np.linalg.norm(row)
            if norm <= 1e-12:
                raise ValueError("checkerboard view has degenerate corner geometry")
            constraints.append(row / norm)
    return np.linalg.svd(np.asarray(constraints), compute_uv=False)


def fit_checkerboard_calibration(image_points, image_size, board_shape,
                                square_size_mm, camera_id, source_ids=None,
                                minimum_views=10, minimum_tilt_span_degrees=10.):
    """Fit one camera from ordered corners; return arrays plus audit metadata.

    ``image_points`` is a sequence of N-by-2 (or N-by-1-by-2) arrays in the raw
    calibration-image coordinates. Every view must show the whole same board at
    the same image resolution. The default requires ten distinct views, following
    OpenCV's practical tutorial; this count is a collection policy, not proof of
    accuracy. Repeated views and rank-deficient pose sets are rejected.

    A second pose-rank check runs after fitted distortion is removed: distorted
    front-facing boards can otherwise appear tilted and pass the first check.
    Its 1e-5 singular-value ratio is a numerical conditioning guard. Fitted board
    normals must also differ by at least ``minimum_tilt_span_degrees`` (default
    10 degrees), preventing noise from making almost untilted views pass rank
    checks. These are conservative collection policies, not accuracy standards;
    a noisy or poorly distributed dataset can still pass them.
    This pure fitting entry point also accepts synthetic projected corners for
    tests. Metadata describes the input, and never certifies a physical camera.
    RMS and per-view errors measure fit to those corners, not held-out accuracy.
    """
    image_size = _validated_image_size(image_size)
    camera_id = _camera_identifier(camera_id)
    object_points = checkerboard_object_points(board_shape, square_size_mm)
    if not isinstance(minimum_views, int) or minimum_views < 3:
        raise ValueError("minimum_views must be an integer of at least 3")
    minimum_tilt_span_degrees = float(minimum_tilt_span_degrees)
    if not math.isfinite(minimum_tilt_span_degrees) or not 0 < minimum_tilt_span_degrees < 180:
        raise ValueError("minimum_tilt_span_degrees must be finite and between 0 and 180")
    views = []
    for index, values in enumerate(image_points):
        points = np.asarray(values, dtype=np.float32)
        if points.shape == (len(object_points), 1, 2):
            points = points.reshape(-1, 2)
        if points.shape != (len(object_points), 2) or not np.all(np.isfinite(points)):
            raise ValueError(f"view {index} must contain every finite checkerboard corner")
        if np.any(points < 0) or np.any(points >= np.asarray(image_size)):
            raise ValueError(f"view {index} has corners outside the image")
        if cv2.contourArea(cv2.convexHull(points)) <= 1:
            raise ValueError(f"view {index} has degenerate checkerboard geometry")
        # Reversed detector ordering represents the same physical image. Check
        # both sequences so a stationary board cannot satisfy the view count.
        if any(min(np.sqrt(np.mean(np.sum((points - previous) ** 2, axis=1))),
                   np.sqrt(np.mean(np.sum((points[::-1] - previous) ** 2, axis=1)))) < 0.25
               for previous in views):
            raise ValueError(f"view {index} repeats an earlier checkerboard pose")
        views.append(points.copy())
    if len(views) < minimum_views:
        raise ValueError(f"need at least {minimum_views} distinct checkerboard views; got {len(views)}")
    if source_ids is None:
        source_ids = [f"provided-corners:{index}" for index in range(len(views))]
    if len(source_ids) != len(views) or any(not isinstance(item, str) for item in source_ids):
        raise ValueError("source_ids must provide one text identifier per view")
    singular_values = _checkerboard_constraint_singular_values(views, object_points, image_size)
    if singular_values[4] <= singular_values[0] * 1e-6:
        raise ValueError("checkerboard poses do not constrain intrinsics; vary board tilt and position")
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        [object_points.copy() for _ in views], views, image_size, None, None,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10),
    )
    _validate_intrinsics(matrix, distortion)
    if not math.isfinite(rms):
        raise ValueError("calibration returned a nonfinite fitting error")
    # A lens bends a planar grid away from a homography. Those bends can inflate
    # the raw constraint rank, making an underconstrained capture appear useful.
    # Recheck with corrected corners; low fitting RMS alone can conceal a wildly
    # wrong focal length. This remains a guard, not held-out lens validation.
    corrected_views = [
        cv2.undistortPoints(points.reshape(-1, 1, 2), matrix, distortion, P=matrix).reshape(-1, 2)
        for points in views
    ]
    corrected_singular_values = _checkerboard_constraint_singular_values(
        corrected_views, object_points, image_size
    )
    if corrected_singular_values[4] <= corrected_singular_values[0] * 1e-5:
        raise ValueError("corrected checkerboard poses do not constrain intrinsics; vary board tilt and position")
    # Corner noise can restore full numerical rank even on a frontal board.
    # Requiring an intentional change in board normal addresses this failure;
    # translating or rotating a flat board within its own plane is insufficient.
    board_normals = np.asarray([cv2.Rodrigues(rotation)[0][:, 2] for rotation in rotations])
    normal_cosines = np.clip(board_normals @ board_normals.T, -1., 1.)
    fitted_tilt_span = float(np.degrees(np.arccos(normal_cosines)).max())
    if fitted_tilt_span < minimum_tilt_span_degrees:
        raise ValueError(
            f"checkerboard tilt span {fitted_tilt_span:.2f} degrees is below required "
            f"{minimum_tilt_span_degrees:g}; capture deliberately different board tilts"
        )
    errors = []
    for observed, rotation, translation in zip(views, rotations, translations):
        projected, _ = cv2.projectPoints(object_points, rotation, translation, matrix, distortion)
        residual = projected.reshape(-1, 2) - observed
        errors.append(float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))))
    metadata = {
        "schema_version": 1,
        "source": "provided checkerboard corners",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv2.__version__,
        "camera_id": camera_id,
        "board_inner_corners": list(map(int, board_shape)),
        "square_size_mm": float(square_size_mm),
        "view_count": len(views),
        "minimum_views": minimum_views,
        "rms_reprojection_error_px": float(rms),
        "per_view_rms_error_px": errors,
        "source_ids": list(source_ids),
        "pose_constraint_singular_values": singular_values.tolist(),
        "corrected_pose_constraint_singular_values": corrected_singular_values.tolist(),
        "minimum_corrected_pose_constraint_ratio": 1e-5,
        "fitted_tilt_span_degrees": fitted_tilt_span,
        "minimum_tilt_span_degrees": minimum_tilt_span_degrees,
        "accuracy_status": "unvalidated; training reprojection error only",
    }
    return {"camera_matrix": matrix, "dist_coeffs": distortion,
            "image_size": image_size, "camera_id": camera_id, "metadata": metadata,
            "rotation_vectors": np.asarray(rotations),
            "translation_vectors": np.asarray(translations)}


def save_checkerboard_calibration(result, path, overwrite=False):
    """Save a non-pickled NPZ, refusing to replace an existing artifact by default."""
    path = Path(path).expanduser()
    matrix = np.asarray(result["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(result["dist_coeffs"], dtype=np.float64)
    _validate_intrinsics(matrix, distortion)
    size = _validated_image_size(result["image_size"])
    camera_id = _camera_identifier(result["camera_id"])
    if path.suffix.lower() != ".npz":
        raise ValueError("calibration output must have an .npz extension")
    # Exclusive mode makes the non-overwrite promise hold even if another
    # process creates the destination between checking and writing.
    metadata = result["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("calibration metadata must be a JSON object")
    if "camera_id" in metadata and metadata["camera_id"] != camera_id:
        raise ValueError("calibration metadata camera_id disagrees with artifact camera_id")
    metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
    with path.open("wb" if overwrite else "xb") as output:
        np.savez_compressed(output, camera_matrix=matrix, dist_coeffs=distortion,
                            image_size=np.asarray(size), camera_id=np.asarray(camera_id),
                            metadata_json=np.asarray(metadata_json))
    return path


def _file_digest(path):
    """Content hash records which input file was used, without embedding images."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate_from_recordings(*, images=None, video=None, camera_id, board_shape,
                              square_size_mm, sample_every=30, minimum_views=10,
                              minimum_tilt_span_degrees=10.):
    """Detect a board in raw still images or sampled video frames, then fit it.

    Mixed image sizes fail immediately. Detection misses and near-duplicate
    frames are reported in metadata instead of increasing the usable-view count.
    No rotations, crops, or resizes are applied during collection.
    """
    if (images is None) == (video is None):
        raise ValueError("provide exactly one of images or video")
    checkerboard_object_points(board_shape, square_size_mm)
    if not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive frame interval")
    corners, identifiers, provenance = [], [], []
    image_size = None
    counts = {"examined": 0, "detected": 0, "duplicate": 0, "not_detected": 0}

    def observe(frame, identifier):
        nonlocal image_size
        if frame is None:
            raise ValueError(f"could not decode calibration image: {identifier}")
        size = (frame.shape[1], frame.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            raise ValueError(f"mixed calibration image sizes: {size} versus {image_size}")
        counts["examined"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        found, detected = cv2.findChessboardCornersSB(
            gray, tuple(map(int, board_shape)),
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
        )
        if not found:
            counts["not_detected"] += 1
            return
        counts["detected"] += 1
        detected = detected.reshape(-1, 2)
        if any(min(np.sqrt(np.mean(np.sum((detected - previous) ** 2, axis=1))),
                   np.sqrt(np.mean(np.sum((detected[::-1] - previous) ** 2, axis=1)))) < 0.25
               for previous in corners):
            counts["duplicate"] += 1
            return
        corners.append(detected)
        identifiers.append(identifier)

    if images is not None:
        directory = Path(images).expanduser()
        if not directory.is_dir():
            raise ValueError(f"calibration image directory does not exist: {directory}")
        files = sorted(path for path in directory.iterdir() if path.is_file()
                       and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not files:
            raise ValueError("calibration image directory contains no supported images")
        for path in files:
            identifier = str(path.resolve())
            observe(cv2.imread(str(path), cv2.IMREAD_COLOR), identifier)
            provenance.append({"path": identifier, "sha256": _file_digest(path)})
        source = "checkerboard images"
    else:
        path = Path(video).expanduser()
        if not path.is_file():
            raise ValueError(f"calibration video does not exist: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"could not open calibration video: {path}")
        try:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % sample_every == 0:
                    observe(frame, f"{path.resolve()}#frame={index}")
                index += 1
        finally:
            capture.release()
        provenance.append({"path": str(path.resolve()), "sha256": _file_digest(path)})
        source = "checkerboard video frames"
    if image_size is None:
        raise ValueError("no calibration images could be read")
    result = fit_checkerboard_calibration(
        corners, image_size, board_shape, square_size_mm, camera_id,
        source_ids=identifiers, minimum_views=minimum_views,
        minimum_tilt_span_degrees=minimum_tilt_span_degrees,
    )
    result["metadata"].update(source=source, input_files=provenance, collection_counts=counts)
    if video is not None:
        result["metadata"]["sample_every_frames"] = sample_every
    return result


def main(argv=None):
    """Command-line entry point; run ``python calibration.py calibrate --help``."""
    parser = argparse.ArgumentParser(description="Create a separate calibration for each eye camera.")
    commands = parser.add_subparsers(dest="command", required=True)
    calibrate = commands.add_parser("calibrate", help="fit a camera from checkerboard images or video")
    inputs = calibrate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path, help="directory of raw checkerboard images")
    inputs.add_argument("--video", type=Path, help="raw checkerboard video")
    calibrate.add_argument("--camera-id", required=True, help="same device label configured in eye_pipeline.py")
    calibrate.add_argument("--board-cols", required=True, type=int, help="number of inner corners across")
    calibrate.add_argument("--board-rows", required=True, type=int, help="number of inner corners down")
    calibrate.add_argument("--square-size-mm", required=True, type=float, help="measured side length of one printed square")
    calibrate.add_argument("--sample-every", type=int, default=30, help="video frame interval (default: 30)")
    calibrate.add_argument("--minimum-views", type=int, default=10, help="required distinct views (default: 10)")
    calibrate.add_argument("--minimum-tilt-span-deg", type=float, default=10., help="required fitted board-normal span (default: 10 degrees)")
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--overwrite", action="store_true", help="explicitly replace an existing calibration")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}; use --overwrite to replace it")
        result = calibrate_from_recordings(
            images=args.images, video=args.video, camera_id=args.camera_id,
            board_shape=(args.board_cols, args.board_rows), square_size_mm=args.square_size_mm,
            sample_every=args.sample_every, minimum_views=args.minimum_views,
            minimum_tilt_span_degrees=args.minimum_tilt_span_deg,
        )
        path = save_checkerboard_calibration(result, args.output, overwrite=args.overwrite)
    except (OSError, ValueError, cv2.error) as error:
        parser.error(str(error))
    report = dict(result["metadata"], output=str(path))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
