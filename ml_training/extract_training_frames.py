"""Extract rotated, cropped, grayscale training frames from an eye video."""

import argparse
from pathlib import Path
from tkinter import Tk, filedialog

import cv2

if __package__:
    from .video_frame_transform import FrameTransform, VALID_FRAME_ROTATIONS
else:
    from video_frame_transform import FrameTransform, VALID_FRAME_ROTATIONS


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "training_data_right_eye"
DEFAULT_RECT = {"x": 0, "y": 50, "w": 1080, "h": 648}
DEFAULT_ROTATION = "clockwise"


def crop_frame(frame, rect):
    x = rect["x"]
    y = rect["y"]
    w = rect["w"]
    h = rect["h"]
    return frame[y : y + h, x : x + w]


def gray_and_resize(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (320, 192), interpolation=cv2.INTER_AREA)


def access_file():
    root = Tk()
    root.withdraw()
    print("Select video file")
    root.update()
    video_path = filedialog.askopenfilename(
        title="Select a Video File", filetypes=[("Video Files", "*.mp4")]
    )
    root.destroy()
    return Path(video_path) if video_path else None


def extract_frames(
    video_path,
    output_dir,
    rotation=DEFAULT_ROTATION,
    rect=DEFAULT_RECT,
    start_index=235,
    sample_every=5,
    preview=True,
):
    if sample_every <= 0:
        raise ValueError("sample_every must be a positive integer")

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width, height)

    transform = FrameTransform(rotation=rotation)
    rotated_width, rotated_height = transform.output_size((width, height))
    if rect["x"] < 0 or rect["y"] < 0 or rect["w"] <= 0 or rect["h"] <= 0:
        raise ValueError(f"rect must use non-negative x/y and positive w/h: {rect}")
    if rect["x"] + rect["w"] > rotated_width or rect["y"] + rect["h"] > rotated_height:
        raise ValueError(
            f"rect {rect} does not fit inside the "
            f"{rotated_width}x{rotated_height} rotated frame"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_counter = 0
    saved_img_counter = start_index
    try:
        while True:
            ongoing, frame = video.read()
            if not ongoing:
                break

            frame = transform.apply_frame(frame)
            frame = crop_frame(frame, rect)
            frame = gray_and_resize(frame)

            if frame_counter % sample_every == 0:
                saved_img_counter += 1
                output_path = output_dir / f"image_{saved_img_counter}.jpg"
                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not write training image: {output_path}")

            if preview:
                cv2.imshow("frame", frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            frame_counter += 1
    finally:
        video.release()
        if preview:
            cv2.destroyAllWindows()

    return saved_img_counter - start_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract TinyUNet training images from an eye video"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="input video; when omitted, open a file-selection dialog",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--start-index",
        type=int,
        default=235,
        help="last existing image number (the first saved image uses the next number)",
    )
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument(
        "--rotation", choices=sorted(VALID_FRAME_ROTATIONS), default=DEFAULT_ROTATION
    )
    parser.add_argument("--crop-x", type=int, default=DEFAULT_RECT["x"])
    parser.add_argument("--crop-y", type=int, default=DEFAULT_RECT["y"])
    parser.add_argument("--crop-width", type=int, default=DEFAULT_RECT["w"])
    parser.add_argument("--crop-height", type=int, default=DEFAULT_RECT["h"])
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="extract without showing the OpenCV preview window",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video or access_file()
    if video_path is None:
        raise SystemExit("No video selected")
    rect = {
        "x": args.crop_x,
        "y": args.crop_y,
        "w": args.crop_width,
        "h": args.crop_height,
    }
    count = extract_frames(
        video_path,
        args.output_dir,
        rotation=args.rotation,
        rect=rect,
        start_index=args.start_index,
        sample_every=args.sample_every,
        preview=not args.no_preview,
    )
    print(f"Saved {count} training images to: {args.output_dir}")


if __name__ == "__main__":
    main()
