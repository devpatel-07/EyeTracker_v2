"""Combine an existing manifest with corrected/remasked training samples."""

import argparse
import json
import os
import random
import re
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MANIFEST = BASE_DIR / "pupil_dataset_split.json"
DEFAULT_IMAGE_DIR = BASE_DIR / "images_to_remask"
DEFAULT_MASK_DIR = BASE_DIR / "remasked_pupils"
DEFAULT_OUTPUT = BASE_DIR / "pupil_finetune_split.json"
DEFAULT_HOLDOUT = 5
DEFAULT_SEED = 42

IMAGE_NUMBER_PATTERN = re.compile(r"image_(\d+)\.[^.]+")


def _image_number(name):
    match = IMAGE_NUMBER_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Could not parse image number from: {name}")
    return int(match.group(1))


def _is_blink(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return not (mask >= 128).any()


def _manifest_path(path, manifest_root):
    path = Path(path).resolve()
    manifest_root = Path(manifest_root).resolve()
    try:
        return Path(os.path.relpath(path, manifest_root)).as_posix()
    except ValueError:
        return path.as_posix()


def collect_new_samples(image_dir, mask_dir, root):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted(
        image_dir.glob("image_*.*"), key=lambda path: _image_number(path.name)
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    samples = []
    for image_path in image_paths:
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        samples.append(
            {
                "filename": image_path.name,
                "image": _manifest_path(image_path, root),
                "mask": _manifest_path(mask_path, root),
                "image_number": _image_number(image_path.name),
                "blink": _is_blink(mask_path),
                "source": "remasked",
            }
        )
    return samples


def _tag_source(samples, source):
    tagged = []
    for sample in samples:
        entry = dict(sample)
        entry["source"] = source
        tagged.append(entry)
    return tagged


def _rebase_samples(samples, source_root, output_root):
    rebased = []
    for sample in samples:
        entry = dict(sample)
        for key in ("image", "mask"):
            source_path = Path(entry[key])
            if not source_path.is_absolute():
                source_path = source_root / source_path
            entry[key] = _manifest_path(source_path, output_root)
        rebased.append(entry)
    return rebased


def _count(split):
    blink = sum(sample["blink"] for sample in split)
    remasked = sum(sample["source"] == "remasked" for sample in split)
    return {
        "total": len(split),
        "pupil": len(split) - blink,
        "blink": blink,
        "remasked": remasked,
        "original": len(split) - remasked,
    }


def build_manifest(base_manifest, new_samples, holdout, seed):
    if holdout < 0 or holdout >= len(new_samples):
        raise ValueError(
            f"holdout must be between 0 and {len(new_samples) - 1}, got {holdout}"
        )

    shuffled = list(new_samples)
    random.Random(seed).shuffle(shuffled)
    new_validation = sorted(shuffled[:holdout], key=lambda sample: sample["image_number"])
    new_train = sorted(shuffled[holdout:], key=lambda sample: sample["image_number"])

    train = _tag_source(base_manifest["train"], "original") + new_train
    validation = _tag_source(base_manifest["validation"], "original") + new_validation

    return {
        "base_manifest_summary": base_manifest.get("summary"),
        "holdout": holdout,
        "seed": seed,
        "train": train,
        "validation": validation,
        "summary": {
            "train": _count(train),
            "validation": _count(validation),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine an existing labeled split with newly remasked images "
            "into a single fine-tuning manifest"
        )
    )
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--masks", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--holdout",
        type=int,
        default=DEFAULT_HOLDOUT,
        help="new images reserved for validation so new-camera gains are measurable",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.output.resolve().parent
    base_manifest_path = args.base_manifest.resolve()
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_manifest["train"] = _rebase_samples(
        base_manifest["train"], base_manifest_path.parent, root
    )
    base_manifest["validation"] = _rebase_samples(
        base_manifest["validation"], base_manifest_path.parent, root
    )
    new_samples = collect_new_samples(args.images, args.masks, root)
    manifest = build_manifest(base_manifest, new_samples, args.holdout, args.seed)

    for split_name in ("train", "validation"):
        for sample in manifest[split_name]:
            for key in ("image", "mask"):
                path = root / sample[key]
                if not path.exists():
                    raise FileNotFoundError(f"Missing {key}: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train = manifest["summary"]["train"]
    validation = manifest["summary"]["validation"]
    print(
        f"Train:      {train['total']:3d} "
        f"({train['pupil']} pupil, {train['blink']} blink | "
        f"{train['original']} original, {train['remasked']} remasked)"
    )
    print(
        f"Validation: {validation['total']:3d} "
        f"({validation['pupil']} pupil, {validation['blink']} blink | "
        f"{validation['original']} original, {validation['remasked']} remasked)"
    )
    print(f"Saved manifest to: {args.output}")


if __name__ == "__main__":
    main()
