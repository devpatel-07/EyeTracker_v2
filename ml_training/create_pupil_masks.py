"""Render binary pupil masks from VIA image annotations."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = BASE_DIR / "pupil_training_labelled_json.json"
DEFAULT_IMAGE_DIR = BASE_DIR / "training_data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "pupil_masks"


def render_mask(image_shape, regions):
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions:
        shape = region["shape_attributes"]
        shape_name = shape["name"]

        if shape_name == "ellipse":
            center = (round(shape["cx"]), round(shape["cy"]))
            axes = (round(shape["rx"]), round(shape["ry"]))
            angle_degrees = math.degrees(shape.get("theta", 0.0))
            cv2.ellipse(mask, center, axes, angle_degrees, 0, 360, 255, -1)
        elif shape_name in ("polyline", "polygon"):
            points = np.column_stack(
                (shape["all_points_x"], shape["all_points_y"])
            ).astype(np.int32)
            cv2.fillPoly(mask, [points], 255)
        else:
            raise ValueError(f"Unsupported VIA shape: {shape_name}")

    return mask


def convert_annotations(json_path, image_dir, output_dir):
    json_path = Path(json_path)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    annotations = json.loads(json_path.read_text(encoding="utf-8"))
    entries = annotations.get("_via_img_metadata", annotations)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"total": 0, "pupil": 0, "blink": 0}

    for entry in entries.values():
        filename = entry["filename"]
        image_path = image_dir / filename
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read source image: {image_path}")

        regions = entry.get("regions", [])
        mask = render_mask(image.shape, regions)
        output_path = output_dir / f"{Path(filename).stem}.png"

        if not cv2.imwrite(str(output_path), mask):
            raise OSError(f"Could not write mask: {output_path}")

        summary["total"] += 1
        summary["pupil" if regions else "blink"] += 1

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create binary pupil masks from VIA JSON annotations"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = convert_annotations(args.annotations, args.image_dir, args.output_dir)
    print(f"Created {summary['total']} masks in: {args.output_dir}")
    print(f"Pupil masks: {summary['pupil']}")
    print(f"Blink masks: {summary['blink']}")


if __name__ == "__main__":
    main()
