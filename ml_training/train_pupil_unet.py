"""Train or fine-tune TinyUNet for binary pupil segmentation."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .pupil_unet_model import TinyUNet
else:
    from pupil_unet_model import TinyUNet


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BASE_DIR / "pupil_dataset_split.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models"


class PupilDataset(Dataset):
    def __init__(self, samples, root=Path("."), augment=False):
        self.samples = list(samples)
        self.root = Path(root)
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.root / sample["image"]
        mask_path = self.root / sample["mask"]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask dimensions differ for: {image_path}")

        image = image.astype(np.float32) / 255.0
        mask = (mask >= 128).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            gain = random.uniform(0.85, 1.15)
            offset = random.uniform(-0.05, 0.05)
            image = np.clip(image * gain + offset, 0.0, 1.0)

        image = np.ascontiguousarray(image[None, :, :])
        mask = np.ascontiguousarray(mask[None, :, :])
        return torch.from_numpy(image), torch.from_numpy(mask)


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * targets).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + (1.0 - dice.mean())


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    total_samples = 0
    pupil_dice = 0.0
    pupil_samples = 0
    correct_blinks = 0
    blink_samples = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = combined_loss(logits, masks)
            if training:
                loss.backward()
                optimizer.step()

        predictions = (torch.sigmoid(logits.detach()) >= 0.5).float()
        dimensions = (1, 2, 3)
        intersections = (predictions * masks).sum(dim=dimensions)
        denominators = predictions.sum(dim=dimensions) + masks.sum(dim=dimensions)
        scores = torch.where(
            denominators > 0,
            (2.0 * intersections) / denominators.clamp_min(1.0),
            torch.ones_like(denominators),
        )
        target_has_pupil = masks.sum(dim=dimensions) > 0
        predicted_areas = predictions.sum(dim=dimensions)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_dice += scores.sum().item()
        total_samples += batch_size

        if target_has_pupil.any():
            pupil_dice += scores[target_has_pupil].sum().item()
            pupil_samples += target_has_pupil.sum().item()
        blink_mask = ~target_has_pupil
        if blink_mask.any():
            correct_blinks += (predicted_areas[blink_mask] < 20).sum().item()
            blink_samples += blink_mask.sum().item()

    return {
        "loss": total_loss / total_samples,
        "dice": total_dice / total_samples,
        "pupil_dice": pupil_dice / pupil_samples if pupil_samples else None,
        "blink_accuracy": correct_blinks / blink_samples if blink_samples else None,
    }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_initial_weights(args, device, output_dir):
    """Load weights from an existing checkpoint so training fine-tunes it."""
    init_path = Path(args.init_checkpoint).resolve()
    if not init_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {init_path}")

    destination = (Path(output_dir) / "pupil_unet_best.pt").resolve()
    if init_path == destination:
        raise ValueError(
            f"--init-checkpoint {init_path} is the same file this run would "
            "overwrite; pass a different --output-dir to keep the original"
        )

    checkpoint = torch.load(init_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    checkpoint_channels = config.get("base_channels")
    if checkpoint_channels is not None and checkpoint_channels != args.base_channels:
        raise ValueError(
            f"checkpoint base_channels {checkpoint_channels} does not match "
            f"--base-channels {args.base_channels}"
        )

    print(
        f"Fine-tuning from: {init_path} "
        f"(epoch {checkpoint.get('epoch', 'unknown')})"
    )
    return checkpoint["model_state_dict"]


def train_model(args):
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    set_seed(args.seed)
    train_dataset = PupilDataset(manifest["train"], root=root, augment=True)
    validation_dataset = PupilDataset(
        manifest["validation"], root=root, augment=False
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_options
    )

    model = TinyUNet(base_channels=args.base_channels).to(device)
    if args.init_checkpoint:
        model.load_state_dict(_load_initial_weights(args, device, output_dir))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    checkpoint_path = output_dir / "pupil_unet_best.pt"
    history_path = output_dir / "pupil_unet_history.json"
    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device)
        scheduler.step(validation_metrics["loss"])

        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} dice {train_metrics['dice']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} "
            f"dice {validation_metrics['dice']:.4f} "
            f"pupil {validation_metrics['pupil_dice']:.4f} "
            f"blink {validation_metrics['blink_accuracy']:.4f}"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "config": {
                        "base_channels": args.base_channels,
                        "input_height": 192,
                        "input_width": 320,
                        "input_channels": 1,
                    },
                    "split_summary": manifest["summary"],
                    "torch_version": torch.__version__,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs")
            break

    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved history to: {history_path}")
    return checkpoint_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train pupil segmentation U-Net")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "load weights from this checkpoint before training, turning the "
            "run into a fine-tune instead of training from scratch"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_model(parse_args())
