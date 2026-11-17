"""Predict a binary pupil mask for each training image in a directory."""

import argparse
import re
from pathlib import Path

import cv2

if __package__:
    from .ml_pupil_inference import MLPupilDetector
else:
    from ml_pupil_inference import MLPupilDetector


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = BASE_DIR / "training_data_right_eye"
DEFAULT_OUTPUT_DIR = BASE_DIR / "training_data_right_eye_pupil_masks"
DEFAULT_CHECKPOINT = BASE_DIR / "models" / "pupil_unet_best.pt"
DEFAULT_THRESHOLD = 0.75

IMAGE_NUMBER_PATTERN = re.compile(r"image_(\d+)\.[^.]+")


def _image_number(path):
    match = IMAGE_NUMBER_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Could not parse image number from: {path.name}")
    return int(match.group(1))


def list_images(image_dir):
    image_dir = Path(image_dir)
    paths = list(image_dir.glob("image_*.*"))
    if not paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")
    return sorted(paths, key=_image_number)


def predict_masks(image_paths, output_dir, detector):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blink_count = 0
    for image_path in image_paths:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mask = detector.predict_mask(bgr)
        if not mask.any():
            blink_count += 1

        output_path = output_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(output_path), mask):
            raise OSError(f"Could not write mask: {output_path}")

    return blink_count


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Predict a pupil mask for every image in a folder, in filename "
            "order, using a TinyUNet checkpoint"
        )
    )
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    detector = MLPupilDetector.from_checkpoint(
        args.checkpoint,
        device=args.device,
        threshold=args.threshold,
    )
    print("Using ML pupil detector on:", detector.device)

    image_paths = list_images(args.images)
    blink_count = predict_masks(image_paths, args.output, detector)

    print(f"Saved {len(image_paths)} masks to: {args.output}")
    print(f"Empty (likely blink) masks: {blink_count}")


if __name__ == "__main__":
    main()
