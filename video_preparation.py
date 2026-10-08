"""Select paired recordings and establish their processing orientations.

OpenCV supplies BGR frames; image sizes use (width, height), while array shapes
use (height, width). ROIs are (x, y, width, height) in the rotated full frame.
The setup GUI previews each rotation independently. It returns choices to the
pipeline without modifying videos or saved presets.
"""

import base64
import math
from pathlib import Path
from tkinter import Tk, filedialog
import tkinter as tk

import cv2

from calibration import VALID_ROTATIONS


def select_eye_video(video_path, side):
    """Resolve a supplied path, or show a file chooser and return None on cancel.

    ``side`` is a human-readable eye label used in prompts and error messages.
    An invalid supplied path raises instead of silently opening the chooser.
    """
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


def open_video(video_path, side):
    """Return an opened VideoCapture; the caller owns its eventual release."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {side} eye video: {video_path}")
    return capture


def read_first_frame(capture, side):
    """Read a preview at the current position, advancing the capture by one frame.

    The pipeline calls this immediately after opening and resets before playback.
    """
    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(f"could not read first frame from {side} eye video")
    return frame


def reset_video(capture, side):
    """Seek to frame zero so setup previews do not remove a frame from processing."""
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
        raise RuntimeError(f"could not reset {side} eye video to frame 0")



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
    """Preview and return frame/calibration rotations for each eye.

    Frame panels show the ROI after the selected rotation; ROIs themselves are
    fixed pixel rectangles and cannot be edited here. Calibration panels rotate
    the source preview as a visual aid; they do not load a calibration image or
    automatically infer alignment. CameraCalibration later interprets that choice
    as calibration-to-source rotation, then adds the chosen frame rotation.

    Return order is left frame, right frame, left calibration, right calibration.
    Closing without confirmation raises, allowing pipeline cleanup to run.
    """
    # Local state starts from presets; confirming does not rewrite eye_pipeline.py.
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
        """Redraw one preview from its original frame to avoid cumulative rotation."""
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
        # Tk does not keep the Python image alive; retain it to prevent blank panels.
        image_labels[key].image = photo
        rotation_labels[key].configure(text=f"{title}: {rotations[key]}")

    def turn(key, direction):
        """Step through the clockwise-ordered rotation names with wraparound."""
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
        # Bind this iteration's key now; a bare closure would use the last panel.
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
        """Mark an intentional confirmation before ending Tk's event loop."""
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
    """Fit a BGR preview inside the bounds, keeping aspect ratio and no upscaling.

    PNG encoding lets Tk read the OpenCV image without a separate imaging library.
    These dimensions affect only setup thumbnails, never detection resolution.
    """
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



def rotate_frame(frame, rotation):
    """Apply a named quarter-turn/half-turn to the entire decoded image.

    With ``none``, return the original array; callers that draw should copy first.
    """
    if rotation == "clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == "none":
        return frame
    raise ValueError(f"unsupported frame rotation: {rotation}")


def video_dimensions(capture):
    """Read source (width, height) from the capture before any frame rotation."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError("video dimensions must be positive")
    return width, height


def rotated_video_dimensions(size, rotation):
    """Return processed (width, height); only quarter-turns swap the dimensions."""
    width, height = (int(value) for value in size)
    if rotation in {"clockwise", "counterclockwise"}:
        return height, width
    if rotation in {"none", "180"}:
        return width, height
    raise ValueError(f"unsupported frame rotation: {rotation}")


def matching_video_fps(left_capture, right_capture):
    """Validate FPS metadata within 0.01 FPS and use the left rate for timestamps.

    Matching rates do not establish synchronization: frame zero alignment and
    constant-rate recordings are assumptions of the pipeline's frame pairing.
    Different recording lengths are allowed; processing stops at the shorter one.
    """
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
