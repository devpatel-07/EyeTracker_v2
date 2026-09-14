"""Create the original training/validation manifest from VIA annotations."""

import argparse
import json
import os
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANNOTATIONS = BASE_DIR / "pupil_training_labelled_json.json"
DEFAULT_IMAGE_DIR = BASE_DIR / "training_data"
DEFAULT_MASK_DIR = BASE_DIR / "pupil_masks"
DEFAULT_OUTPUT = BASE_DIR / "pupil_dataset_split.json"
VALIDATION_START_IMAGE = 236


def _manifest_path(path, manifest_root):
    path = Path(path).resolve()
    manifest_root = Path(manifest_root).resolve()
    try:
        return Path(os.path.relpath(path, manifest_root)).as_posix()
    except ValueError:
        return path.as_posix()


def build_split(
    entries,
    image_dir,
    mask_dir,
    manifest_root,
    validation_start=VALIDATION_START_IMAGE,
):
    samples = []

    for entry in entries:
        filename = entry["filename"]
        match = re.fullmatch(r"image_(\d+)\.[^.]+", filename)
        if match is None:
            raise ValueError(f"Could not parse image number from: {filename}")

        number = int(match.group(1))
        image_path = Path(image_dir) / filename
        mask_path = Path(mask_dir) / f"{Path(filename).stem}.png"
        samples.append(
            {
                "filename": filename,
                "image": _manifest_path(image_path, manifest_root),
                "mask": _manifest_path(mask_path, manifest_root),
                "image_number": number,
                "blink": len(entry.get("regions", [])) == 0,
            }
        )

    samples.sort(key=lambda sample: sample["image_number"])
    train = [sample for sample in samples if sample["image_number"] < validation_start]
    validation = [
        sample for sample in samples if sample["image_number"] >= validation_start
    ]

    if not train or not validation:
        raise ValueError("Both training and validation splits must contain samples")

    def count_samples(split):
        blink_count = sum(sample["blink"] for sample in split)
        return {
            "total": len(split),
            "pupil": len(split) - blink_count,
            "blink": blink_count,
        }

    return {
        "validation_start_image": validation_start,
        "train": train,
        "validation": validation,
        "summary": {
            "train": count_samples(train),
            "validation": count_samples(validation),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the original pupil training/validation manifest"
    )
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-start", type=int, default=VALIDATION_START_IMAGE
    )
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    metadata = annotations.get("_via_img_metadata", annotations)
    manifest_root = args.output.resolve().parent
    manifest = build_split(
        metadata.values(),
        args.image_dir,
        args.mask_dir,
        manifest_root,
        args.validation_start,
    )

    for split_name in ("train", "validation"):
        for sample in manifest[split_name]:
            for path_key in ("image", "mask"):
                path = manifest_root / sample[path_key]
                if not path.exists():
                    raise FileNotFoundError(f"Missing {path_key}: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train = manifest["summary"]["train"]
    validation = manifest["summary"]["validation"]
    print(
        f"Train: {train['total']} "
        f"({train['pupil']} pupil, {train['blink']} blink)"
    )
    print(
        f"Validation: {validation['total']} "
        f"({validation['pupil']} pupil, {validation['blink']} blink)"
    )
    print(f"Saved split to: {args.output}")


if __name__ == "__main__":
    main()
