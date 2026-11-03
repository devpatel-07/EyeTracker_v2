import math
import numpy as np


# Convert mm to inches
def mm2in(mm):
    return mm / 25.4


# Convert gaze point to reach point for arm (old)
# Gaze point (mm): origin on the ground below the person's eyes. x forward, y to the person's left, z up
# Arm point (in): origin at the arm reference point, d ahead of the person and h + o above the ground.
# x points back toward the person, y to the person's left, z up
def convert_coords(gx, gy, gz, d, h, o):
    rx = abs(d - mm2in(gx))
    ry = mm2in(gy)
    rz = mm2in(gz) - (h + o)

    return (rx, ry, rz)


def gaze_to_eye_midpoint_origin(point_mm, eye_height_mm):
    """
    Shifts the GazePoint origin from (0, 0, 0) on the ground
    to the midpoint between the person's eyes (0, 0, eye_height_mm).

    Input:
        point_mm: (x, y, z) in mm where z=0 is ground level.
        eye_height_mm: height of the eye midpoint above the ground in mm.
    Output:
        (x, y, z) in mm with origin at the eye midpoint.
    """
    gx, gy, gz = point_mm
    return (gx, gy, gz - eye_height_mm)


def aruco_to_camera_matrix(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg):
    """
    Constructs the 4x4 homogeneous transformation matrix T_cam_marker that transforms
    a point from the ArUco marker basis to the base camera basis:

        P_cam = T_cam_marker @ P_marker

    Variables from tracker.py:
        x_mm, y_mm, z_mm: ArUco marker translation relative to the camera optical center.
        roll_deg, pitch_deg, yaw_deg: Euler angles representing marker-to-camera rotation,
            where R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    # Rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])
    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    r_cam_marker = rz @ ry @ rx

    t_cam_marker = np.eye(4, dtype=float)
    t_cam_marker[:3, :3] = r_cam_marker
    t_cam_marker[:3, 3] = [x_mm, y_mm, z_mm]

    return t_cam_marker


# Mapping from Head frame (x forward, y wearer's left, z up)
# to ArUco Marker frame (x marker-right / wearer's left, y marker-up, z marker-normal / forward)
HEAD_TO_ARUCO_ROTATION = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])


def gaze_to_base_camera(
    gaze_point_mm,
    x_mm,
    y_mm,
    z_mm,
    roll_deg,
    pitch_deg,
    yaw_deg,
    eye_height_mm=None,
    is_marker_basis=False,
):
    """
    Transforms a gaze point into coordinates relative to the base camera.

    Parameters:
        gaze_point_mm: (gx, gy, gz) in mm.
        x_mm, y_mm, z_mm: Position of ArUco marker in base camera frame (from tracker.py).
        roll_deg, pitch_deg, yaw_deg: Orientation of ArUco marker in degrees (from tracker.py).
        eye_height_mm: If gaze_point_mm has origin on the ground, provide eye_height_mm
                       to shift origin to eye midpoint. If already centered at eye midpoint, leave as None.
        is_marker_basis: If True, gaze_point_mm is already expressed in the ArUco marker basis.
                         If False (default), gaze_point_mm is in the standard head frame
                         (x forward, y left, z up) and is mapped into the ArUco marker basis.

    Returns:
        (x_cam, y_cam, z_cam): Gaze point coordinates in mm relative to the base camera optical frame.
    """
    # 1. Shift origin from ground to eye midpoint if eye_height_mm is specified
    if eye_height_mm is not None:
        p_head = np.array(gaze_to_eye_midpoint_origin(gaze_point_mm, eye_height_mm), dtype=float)
    else:
        p_head = np.array(gaze_point_mm, dtype=float)

    # 2. Convert to ArUco marker basis if starting in head frame
    if not is_marker_basis:
        p_marker = HEAD_TO_ARUCO_ROTATION @ p_head
    else:
        p_marker = p_head

    # 3. Apply the ArUco-to-camera transformation matrix template
    t_cam_marker = aruco_to_camera_matrix(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)

    p_marker_homo = np.array([p_marker[0], p_marker[1], p_marker[2], 1.0])
    p_cam_homo = t_cam_marker @ p_marker_homo

    return tuple(float(v) for v in p_cam_homo[:3])
