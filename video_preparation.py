# Import dependencies

import base64
import math
from pathlib import Path
from tkinter import Tk, filedialog
import tkinter as tk

import cv2

from calibration import VALID_ROTATIONS


# Selects eye video when video path is not already provided - function

def select_eye_video(video_path, side):
    if video_path is not None:
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{side} eye video not found: {path}")
        return path

    root = Tk()
    root.withdraw()
    try:
        root.update()
        selected = filedialog.askopenfilename(
            title=f"Select {side} eye video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()
    return Path(selected).resolve() if selected else None


# Opens video and confirms video can be read - function

def open_video(video_path, side):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {side} eye video: {video_path}")
    return capture


def read_first_frame(capture, side):
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"could not read first frame from {side} eye video")
    return frame


def reset_video(capture, side):
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise RuntimeError(f"could not reset {side} eye video to frame 0")



# Creates primary GUI used to set up video rotation, dimensions, etc. Functionality includes being able to see two first frames of each eye video
# Each frame is either to show ROI rotation or calibration rotation preset. Buttons allow for 90 degree frame rotations that show and determine preset
# for both ROI and calibration. Confirm button locks presets and closes window - function

def setup_rotation_gui(
    left_frame,
    right_frame,
    left_roi,
    right_roi,
    left_frame_rotation,
    right_frame_rotation,
    left_calibration_rotation,
    right_calibration_rotation,
):
    # Uses stored presets passed from eye_pipeline.py as initial frame rotations in window
    rotations = {
        "left_frame": left_frame_rotation,
        "right_frame": right_frame_rotation,
        "left_calibration": left_calibration_rotation,
        "right_calibration": right_calibration_rotation,
    }
    for rotation in rotations.values():
        if rotation not in VALID_ROTATIONS:
            raise ValueError(f"unsupported rotation: {rotation}")

    root = tk.Tk()
    root.title("Eye video rotation setup")
    root.resizable(True, True)
    confirmed = {"value": False}
    image_labels = {}
    rotation_labels = {}

    panels = (
        ("left_frame", "Left ROI/frame orientation", left_frame, left_roi, True),
        (
            "left_calibration",
            "Left calibration orientation",
            left_frame,
            left_roi,
            False,
        ),
        (
            "right_frame",
            "Right ROI/frame orientation",
            right_frame,
            right_roi,
            True,
        ),
        (
            "right_calibration",
            "Right calibration orientation",
            right_frame,
            right_roi,
            False,
        ),
    )

    def update_panel(key, title, frame, roi, show_roi):
        preview = rotate_frame(frame, rotations[key]).copy()
        if show_roi:
            x, y, width, height = (int(value) for value in roi)
            cv2.rectangle(
                preview,
                (x, y),
                (x + width, y + height),
                (255, 0, 0),
                2,
            )
        photo = _preview_photo(preview)
        image_labels[key].configure(image=photo)
        image_labels[key].image = photo
        rotation_labels[key].configure(text=f"{title}: {rotations[key]}")

    def turn(key, direction):
        index = VALID_ROTATIONS.index(rotations[key])
        rotations[key] = VALID_ROTATIONS[(index + direction) % 4]
        for panel in panels:
            if panel[0] == key:
                update_panel(*panel)
                break

    for panel_index, panel in enumerate(panels):
        key, title, _, _, _ = panel
        row = panel_index // 2
        column = panel_index % 2
        container = tk.Frame(root, padx=8, pady=8)
        container.grid(row=row, column=column, sticky="nsew")
        rotation_labels[key] = tk.Label(container, text=title)
        rotation_labels[key].pack()
        image_labels[key] = tk.Label(container)
        image_labels[key].pack(pady=5)
        controls = tk.Frame(container)
        controls.pack()
        tk.Button(
            controls,
            text="Rotate left",
            command=lambda selected=key: turn(selected, -1),
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            controls,
            text="Rotate right",
            command=lambda selected=key: turn(selected, 1),
        ).pack(side=tk.LEFT, padx=4)
        update_panel(*panel)

    def confirm():
        confirmed["value"] = True
        root.destroy()

    tk.Button(root, text="Confirm rotations", command=confirm, padx=20).grid(
        row=2,
        column=0,
        columnspan=2,
        pady=10,
    )
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    if not confirmed["value"]:
        raise RuntimeError("rotation setup was closed without confirmation")
    return (
        rotations["left_frame"],
        rotations["right_frame"],
        rotations["left_calibration"],
        rotations["right_calibration"],
    )


def _preview_photo(frame, maximum_width=320, maximum_height=200):
    height, width = frame.shape[:2]
    scale = min(maximum_width / width, maximum_height / height, 1.0)
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise RuntimeError("could not create setup preview")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data, format="png")



# Rotates frame according to ROI rotation preset - function

def rotate_frame(frame, rotation):
    if rotation == "clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == "none":
        return frame
    raise ValueError(f"unsupported frame rotation: {rotation}")


# Calculates video dimensions after ROI rotation - function

def video_dimensions(capture):
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    return width, height


def rotated_video_dimensions(size, rotation):
    width, height = (int(value) for value in size)
    if rotation in {"clockwise", "counterclockwise"}:
        return height, width
    if rotation in {"none", "180"}:
        return width, height
    raise ValueError(f"unsupported frame rotation: {rotation}")


# Checks that left and right videos have valid matching FPS - function

def matching_video_fps(left_capture, right_capture):
    left_fps = float(left_capture.get(cv2.CAP_PROP_FPS))
    right_fps = float(right_capture.get(cv2.CAP_PROP_FPS))
    if not all(
        math.isfinite(value) and value > 0 for value in (left_fps, right_fps)
    ):
        raise ValueError("both videos must report a positive finite frame rate")
    if not math.isclose(left_fps, right_fps, abs_tol=0.01, rel_tol=0.0):
        raise ValueError(
            f"video frame rates differ: left={left_fps}, right={right_fps}"
        )
    return left_fps
