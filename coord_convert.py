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
        (x, y, z) in mm with origin at the eye midpoint (center of glasses).
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


def vector_to_base_camera(
    vector,
    roll_deg,
    pitch_deg,
    yaw_deg,
    is_marker_basis=False,
):
    """
    Transforms a direction vector into the base camera frame.
    Because this is a vector (direction), translation does NOT apply;
    only the rotation is applied.

    Parameters:
        vector: (vx, vy, vz) direction vector.
        roll_deg, pitch_deg, yaw_deg: Orientation of ArUco marker from tracker.py.
        is_marker_basis: True if vector is already in marker coordinates,
                         False if in head coordinates (x forward, y left, z up).

    Returns:
        (vx_cam, vy_cam, vz_cam): unit direction vector relative to base camera.
    """
    v = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(v)
    if norm > 1e-12:
        v = v / norm

    if not is_marker_basis:
        v_marker = HEAD_TO_ARUCO_ROTATION @ v
    else:
        v_marker = v

    r_cam_marker = aruco_to_camera_matrix(0, 0, 0, roll_deg, pitch_deg, yaw_deg)[:3, :3]
    v_cam = r_cam_marker @ v_marker
    v_cam_norm = np.linalg.norm(v_cam)
    if v_cam_norm > 1e-12:
        v_cam = v_cam / v_cam_norm

    return tuple(float(x) for x in v_cam)


def log_live_gaze_vectors(
    filepath,
    left_vector_head,
    right_vector_head,
    x_mm,
    y_mm,
    z_mm,
    roll_deg,
    pitch_deg,
    yaw_deg,
    mode="w",
):
    """
    Writes the latest gaze vectors and ArUco pose to a .txt file.

    The default write mode replaces the previous snapshot instead of growing
    the file once per frame. Pass a different mode only when history is wanted.

    Logged data:
      1. Left eye gaze vector in head frame (x forward, y left, z up)
      2. Right eye gaze vector in head frame (x forward, y left, z up)
      3. Left eye gaze vector transformed to base camera frame
      4. Right eye gaze vector transformed to base camera frame
      5. ArUco marker position relative to base camera (x_mm, y_mm, z_mm)
    """
    if left_vector_head is not None:
        left_cam = vector_to_base_camera(left_vector_head, roll_deg, pitch_deg, yaw_deg)
        lh_str = f"{left_vector_head[0]:.4f}, {left_vector_head[1]:.4f}, {left_vector_head[2]:.4f}"
        lc_str = f"{left_cam[0]:.4f}, {left_cam[1]:.4f}, {left_cam[2]:.4f}"
    else:
        lh_str = "None"
        lc_str = "None"

    if right_vector_head is not None:
        right_cam = vector_to_base_camera(right_vector_head, roll_deg, pitch_deg, yaw_deg)
        rh_str = f"{right_vector_head[0]:.4f}, {right_vector_head[1]:.4f}, {right_vector_head[2]:.4f}"
        rc_str = f"{right_cam[0]:.4f}, {right_cam[1]:.4f}, {right_cam[2]:.4f}"
    else:
        rh_str = "None"
        rc_str = "None"

    aruco_pos_str = f"{x_mm:.2f}, {y_mm:.2f}, {z_mm:.2f}"

    line = (
        f"left_vector_head=[{lh_str}] | "
        f"right_vector_head=[{rh_str}] | "
        f"left_vector_cam=[{lc_str}] | "
        f"right_vector_cam=[{rc_str}] | "
        f"aruco_pos_mm=[{aruco_pos_str}]\n"
    )

    with open(filepath, mode, encoding="utf-8") as f:
        if mode == "w":
            f.write("# Live Gaze Vectors and ArUco Pose Log\n")
        f.write(line)
        f.flush()

    return {
        "left_vector_head": left_vector_head,
        "right_vector_head": right_vector_head,
        "left_vector_cam": left_cam if left_vector_head is not None else None,
        "right_vector_cam": right_cam if right_vector_head is not None else None,
        "aruco_pos_mm": (x_mm, y_mm, z_mm),
    }
