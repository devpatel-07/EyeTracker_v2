# Import dependencies

import argparse
import math

import numpy as np
import matplotlib.pyplot as plt


# 3D reconstruction of the base camera, the head (ArUco marker on the glasses) and the gaze point. All lengths are mm.
#
# Base camera frame (OpenCV, same as tracker.py): origin at the lens, x right, y down, z out of the lens. This is the
# frame the marker position and orientation are reported in. For plotting it is redrawn as x right, y forward (depth)
# and z up so the scene looks upright.
#
# Head frame (gaze_estimation.py): x forward, y to the person's left, z up, origin at the eye midpoint (the gaze point
# is reported with the origin on the ground, so the eye height is subtracted first).
#
# Marker frame (tracker.py object points): x right and y up as seen by a viewer facing the marker, z out of the marker
# toward the viewer. With the marker on the front of the glasses, that is wearer's left, up, and forward.

# Columns are the head x (forward), y (left), z (up) axes written in marker coordinates
HEAD_IN_MARKER = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
])

# Camera (x right, y down, z forward) to plot (x right, y forward, z up)
CAMERA_TO_PLOT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])

AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")
AXIS_NAMES = ("forward", "left", "up")


# Builds the marker to camera rotation from roll, pitch, yaw in degrees. Same convention as tracker.euler_degrees:
# R = Rz(yaw) Ry(pitch) Rx(roll) - function

def rotation_from_euler(roll_deg, pitch_deg, yaw_deg):
    roll, pitch, yaw = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


# Places the head and gaze point in the base camera frame. marker_rotation is the marker to camera rotation. Returns
# the head axes (columns forward, left, up) and the gaze point in the camera frame, or None for the point if the gaze
# estimate is not valid - function

def head_in_camera(position_mm, marker_rotation, gaze=None, eye_midpoint_mm=None):
    position = np.asarray(position_mm, dtype=float)
    head_axes = marker_rotation @ HEAD_IN_MARKER

    gaze_point = None
    if gaze is not None and gaze.valid and gaze.point_mm is not None:
        offset = np.asarray(gaze.point_mm, dtype=float) - np.asarray(eye_midpoint_mm, dtype=float)
        gaze_point = position + head_axes @ offset
    return head_axes, gaze_point


# Live 3D view. Call update() with each new marker pose and gaze estimate - class

class Scene3D:
    def __init__(self, eye_midpoint_mm, axis_length_mm=100.0, interactive=True):
        self.eye_midpoint_mm = np.asarray(eye_midpoint_mm, dtype=float)
        self.axis_length_mm = float(axis_length_mm)

        if interactive:
            plt.ion()
        self.fig = plt.figure("Head and gaze reconstruction", figsize=(8, 7))
        self.ax = self.fig.add_subplot(projection="3d")
        self.view = None

    # Redraws the scene. position_mm is the marker (eye midpoint) in the base camera frame, rpy_deg is
    # (roll, pitch, yaw) of the marker in degrees, gaze is a gaze_estimation.GazePoint or None - function

    def update(self, position_mm, rpy_deg, gaze=None):
        rotation = rotation_from_euler(*rpy_deg)
        head_axes, gaze_cam = head_in_camera(position_mm, rotation, gaze, self.eye_midpoint_mm)

        origin = CAMERA_TO_PLOT @ np.asarray(position_mm, dtype=float)
        axes_plot = CAMERA_TO_PLOT @ head_axes
        points = [np.zeros(3), origin]

        ax = self.ax
        if self.view is not None:
            self.view = (ax.elev, ax.azim)
        ax.clear()

        # Base camera: origin marker plus its own axes
        ax.scatter(*np.zeros(3), color="black", s=80, marker="s", label="base camera")
        for i, color in enumerate(("tab:red", "tab:green", "tab:blue")):
            direction = np.eye(3)[i] * self.axis_length_mm * 0.5
            ax.quiver(0, 0, 0, *direction, color=color, alpha=0.4, linewidth=1)

        # Head: dot at the eye midpoint with the head frame axes
        ax.scatter(*origin, color="tab:orange", s=60, label="eye midpoint (marker)")
        for i, (color, name) in enumerate(zip(AXIS_COLORS, AXIS_NAMES)):
            ax.quiver(*origin, *(axes_plot[:, i] * self.axis_length_mm), color=color, linewidth=2, label=name)
        ax.plot(*np.column_stack((np.zeros(3), origin)), color="gray", linestyle=":", linewidth=1)

        # Gaze: vergence point and the line of sight to it
        if gaze_cam is not None:
            target = CAMERA_TO_PLOT @ gaze_cam
            points.append(target)
            ax.scatter(*target, color="tab:purple", s=80, marker="*", label="eye vergence point")
            ax.plot(*np.column_stack((origin, target)), color="tab:purple", linewidth=1.5)

        status = "no gaze" if gaze is None else gaze.status
        distance = np.linalg.norm(position_mm)
        ax.set_title(f"marker {distance:.0f} mm from base camera | gaze: {status}")
        self._frame(points)
        ax.legend(loc="upper left")

        if self.view is not None:
            ax.view_init(*self.view)
        else:
            ax.view_init(elev=20, azim=-60)
            self.view = (ax.elev, ax.azim)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # Equal-scale cubic limits around everything drawn - function

    def _frame(self, points):
        pts = np.array(points)
        center = (pts.max(axis=0) + pts.min(axis=0)) / 2
        half = max((pts.max(axis=0) - pts.min(axis=0)).max() / 2, self.axis_length_mm) * 1.1
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)
        self.ax.set_box_aspect((1, 1, 1))
        self.ax.set_xlabel("x right (mm)")
        self.ax.set_ylabel("y forward from camera (mm)")
        self.ax.set_zlabel("z up (mm)")

    def show(self):
        plt.ioff()
        plt.show()


# Demo with made-up numbers - function

def main():
    parser = argparse.ArgumentParser(description="Plot base camera, head marker and gaze point")
    parser.add_argument("--pos", type=float, nargs=3, default=[0.0, -200.0, 1200.0], metavar=("X", "Y", "Z"),
                        help="marker position in base camera frame, mm (x right, y down, z forward)")
    parser.add_argument("--rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("ROLL", "PITCH", "YAW"),
                        help="marker roll pitch yaw in degrees as printed by tracker.py")
    parser.add_argument("--gaze", type=float, nargs=3, default=[600.0, 0.0, 1200.0], metavar=("X", "Y", "Z"),
                        help="gaze point in the head frame, mm (origin on ground below eyes, x forward, y left, z up)")
    parser.add_argument("--eye-height", type=float, default=1600.0, help="eye midpoint height above ground, mm")
    args = parser.parse_args()

    from types import SimpleNamespace
    gaze = SimpleNamespace(valid=True, point_mm=tuple(args.gaze), status="demo")
    scene = Scene3D(eye_midpoint_mm=[0.0, 0.0, args.eye_height], interactive=False)
    scene.update(args.pos, args.rpy, gaze)
    scene.show()


if __name__ == "__main__":
    main()
