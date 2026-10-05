# Import dependencies

from dataclasses import dataclass
import math

import numpy as np

from eye_model_estimation import gaze_direction_from_eye_model


# Head frame used for gaze points (all values in mm). The origin is on the ground directly below the midpoint between
# the eyes. x points forward, y points to the person's left, and z points up. The person is assumed to stand upright
# without rotating their head, so the head frame stays fixed to the room

# pye3d returns eye and pupil centers in each eye camera's own frame. x points right in the processed frame, y points
# down, and z points out of the camera lens. Frame rotation presets must make each eye image upright for this to hold


# Stores gaze point and the reason it was accepted or rejected - data class

@dataclass(frozen=True)
class GazePoint:
    valid: bool
    point_mm: tuple[float, float, float] | None
    miss_distance_mm: float | None
    distance_mm: float | None
    status: str


# Builds rotation from one eye camera frame into the head frame. Each camera arm leaves the glasses at yaw_deg from
# straight ahead and the camera faces back at its eye turned inward by the same angle. side_sign is +1 for the right
# camera (turns toward the person's left) and -1 for the left camera - function

def camera_to_head_rotation(yaw_deg, pitch_deg, side_sign):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Camera z axis looks back at the eye, turned inward by yaw and tilted up by pitch
    camera_z = np.array([
        -math.cos(yaw) * math.cos(pitch),
        side_sign * math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
    ])

    # Camera y axis is image down, so it is the head frame down direction with any camera z part removed
    down = np.array([0.0, 0.0, -1.0])
    camera_y = down - np.dot(down, camera_z) * camera_z
    camera_y /= np.linalg.norm(camera_y)

    camera_x = np.cross(camera_y, camera_z)
    return np.column_stack((camera_x, camera_y, camera_z))


# Estimates camera lens to eye center distance from glasses geometry. Used to sanity check pye3d eye depth - function

def expected_eye_distance(ipd_mm, camera_separation_mm, yaw_deg):
    side_offset_mm = (camera_separation_mm - ipd_mm) / 2.0
    if side_offset_mm <= 0 or yaw_deg <= 0:
        return None
    return side_offset_mm / math.sin(math.radians(yaw_deg))


# Finds how far along each gaze ray the two rays come closest to each other. Returns None for parallel rays - function

def closest_ray_steps(left_origin, left_direction, right_origin, right_direction):
    offset = left_origin - right_origin
    alignment = np.dot(left_direction, right_direction)
    left_offset = np.dot(left_direction, offset)
    right_offset = np.dot(right_direction, offset)

    denominator = 1.0 - alignment ** 2
    if denominator < 1e-9:
        return None
    left_step = (alignment * right_offset - left_offset) / denominator
    right_step = (right_offset - alignment * left_offset) / denominator
    return left_step, right_step


# Combines left and right eye gaze rays into one gaze point - class

class GazeEstimator:
    def __init__(
        self,
        ipd_mm,
        eye_height_mm,
        camera_separation_mm,
        camera_yaw_deg,
        camera_pitch_deg,
        tolerance_mm,
        min_distance_mm,
        max_distance_mm,
    ):
        self.left_eye_mm = np.array([0.0, ipd_mm / 2.0, eye_height_mm])
        self.right_eye_mm = np.array([0.0, -ipd_mm / 2.0, eye_height_mm])
        self.eye_midpoint_mm = np.array([0.0, 0.0, eye_height_mm])

        self.left_rotation = camera_to_head_rotation(camera_yaw_deg, camera_pitch_deg, -1)
        self.right_rotation = camera_to_head_rotation(camera_yaw_deg, camera_pitch_deg, 1)

        self.tolerance_mm = float(tolerance_mm)
        self.min_distance_mm = float(min_distance_mm)
        self.max_distance_mm = float(max_distance_mm)
        self.expected_eye_distance_mm = expected_eye_distance(
            ipd_mm,
            camera_separation_mm,
            camera_yaw_deg,
        )

    # Rotates one eye's pye3d gaze direction into the head frame - function

    def head_gaze_direction(self, estimate, rotation):
        direction = gaze_direction_from_eye_model(estimate)
        if direction is None:
            return None
        return rotation @ np.asarray(direction)

    # Finds the gaze point where both gaze rays pass within the set tolerance of each other - function

    def estimate(self, left_estimate, right_estimate):
        left_direction = self.head_gaze_direction(left_estimate, self.left_rotation)
        right_direction = self.head_gaze_direction(right_estimate, self.right_rotation)
        if left_direction is None or right_direction is None:
            return GazePoint(False, None, None, None, "waiting for both eyes")

        # A gaze ray pointing backward means camera yaw/pitch presets or frame rotation presets are wrong
        if left_direction[0] <= 0 or right_direction[0] <= 0:
            return GazePoint(False, None, None, None, "gaze points backward")

        steps = closest_ray_steps(
            self.left_eye_mm,
            left_direction,
            self.right_eye_mm,
            right_direction,
        )

        # Parallel or diverging rays mean the person is looking far away, so place the point at max distance
        if steps is None or steps[0] <= 0 or steps[1] <= 0:
            return self._max_distance_point(left_direction, right_direction, None)

        left_point = self.left_eye_mm + steps[0] * left_direction
        right_point = self.right_eye_mm + steps[1] * right_direction
        miss_distance = float(np.linalg.norm(left_point - right_point))
        if miss_distance > self.tolerance_mm:
            return GazePoint(False, None, miss_distance, None, "rays too far apart")

        point = (left_point + right_point) / 2.0
        distance = float(np.linalg.norm(point - self.eye_midpoint_mm))
        if distance < self.min_distance_mm:
            return GazePoint(False, None, miss_distance, distance, "too close")
        if distance > self.max_distance_mm:
            return self._max_distance_point(left_direction, right_direction, miss_distance)
        return self._checked_point(point, miss_distance, distance, "ready")

    # Places gaze point at max distance along the average of both gaze directions - function

    def _max_distance_point(self, left_direction, right_direction, miss_distance):
        direction = left_direction + right_direction
        direction /= np.linalg.norm(direction)
        point = self.eye_midpoint_mm + self.max_distance_mm * direction
        return self._checked_point(point, miss_distance, self.max_distance_mm, "max distance")

    # Rejects gaze points below the ground - function

    def _checked_point(self, point, miss_distance, distance, status):
        if point[2] < 0:
            return GazePoint(False, None, miss_distance, distance, "below ground")
        return GazePoint(
            True,
            tuple(float(value) for value in point),
            miss_distance,
            distance,
            status,
        )
