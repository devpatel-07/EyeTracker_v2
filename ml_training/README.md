# Pupil-model training tools

This folder is a self-contained copy of the EyeTracker pupil-segmentation
workflow. Its Python modules import only the standard library, third-party
packages in `requirements.txt`, and other modules in this folder. Data, videos,
VIA annotation JSON, and model checkpoints may live anywhere and are supplied
as command-line paths.

Install the Python dependencies with:

```console
python -m pip install -r ml_training/requirements.txt
```

All defaults are resolved relative to this folder, not the process's current
working directory. The examples below use explicit paths so the same commands
work when the data is stored elsewhere. Run any utility with `--help` for its
complete interface.

## Initial training workflow

The workflow is:

```text
eye video
  -> extract_training_frames.py
  -> manually annotate pupils in VIA
  -> create_pupil_masks.py
  -> prepare_pupil_split.py
  -> train_pupil_unet.py
  -> pupil_unet_best.pt
```

Extract every fifth frame after the original clockwise rotation, crop, grayscale
conversion, and 320x192 resize. `--start-index` is the last already-used image
number, so this example's first file is `image_1.jpg`:

```console
python ml_training/extract_training_frames.py --video data/eye.mp4 --output-dir data/original/images --start-index 0
```

The preview window is enabled by default and `q` stops extraction. Add
`--no-preview` for unattended extraction. The historical defaults remain a
clockwise rotation, crop `(x=0, y=50, width=1080, height=648)`, every fifth
frame, and last-used image number 235. The crop and rotation can be adjusted
with `--rotation`, `--crop-x`, `--crop-y`, `--crop-width`, and `--crop-height`.

After annotating those images in VIA, render ellipse, polygon, and polyline
regions as masks. Images with no regions become all-zero blink masks:

```console
python ml_training/create_pupil_masks.py --annotations data/original/via_annotations.json --image-dir data/original/images --output-dir data/original/masks
```

Build the deterministic image-number split. Image numbers below 236 train and
numbers 236 onward validate by default:

```console
python ml_training/prepare_pupil_split.py --annotations data/original/via_annotations.json --image-dir data/original/images --mask-dir data/original/masks --output data/original/pupil_dataset_split.json
```

Change the split boundary with `--validation-start`. Manifest image and mask
paths are written relative to the manifest where possible, so training does not
depend on the directory from which the command was launched.

Train with the same augmentation, BCE-plus-Dice loss, AdamW optimizer,
validation metrics, learning-rate scheduler, early stopping, and best-checkpoint
saving as the original script:

```console
python ml_training/train_pupil_unet.py --manifest data/original/pupil_dataset_split.json --output-dir outputs/initial_model
```

The best checkpoint is `outputs/initial_model/pupil_unet_best.pt`; the epoch
history is `outputs/initial_model/pupil_unet_history.json`.

## Optional model-improvement workflow

The workflow is:

```text
new eye images + existing pupil_unet_best.pt
  -> predict_pupil_masks.py
  -> manually inspect/correct annotations or masks
  -> create_pupil_masks.py
  -> prepare_finetune_split.py
  -> train_pupil_unet.py --init-checkpoint ...
  -> improved pupil_unet_best.pt
```

Generate masks to accelerate review or annotation:

```console
python ml_training/predict_pupil_masks.py --images data/finetune/images_to_remask --checkpoint outputs/initial_model/pupil_unet_best.pt --output data/finetune/predicted_masks --threshold 0.75
```

`--device` accepts `auto`, `cuda`, or `cpu`. After manual correction in VIA,
regenerate the authoritative masks with the same mask utility used above:

```console
python ml_training/create_pupil_masks.py --annotations data/finetune/corrected_via_annotations.json --image-dir data/finetune/images_to_remask --output-dir data/finetune/remasked_pupils
```

Combine the original split with the corrected images. Five corrected images are
held out for validation by default; `--seed 42` makes that selection repeatable:

```console
python ml_training/prepare_finetune_split.py --base-manifest data/original/pupil_dataset_split.json --images data/finetune/images_to_remask --masks data/finetune/remasked_pupils --output data/finetune/pupil_finetune_split.json --holdout 5
```

The utility rebases paths from the base manifest relative to the new manifest,
so the two manifests need not share a parent directory.

Fine-tune into a different output directory so the source checkpoint cannot be
overwritten:

```console
python ml_training/train_pupil_unet.py --manifest data/finetune/pupil_finetune_split.json --init-checkpoint outputs/initial_model/pupil_unet_best.pt --output-dir outputs/improved_model
```

The checkpoint's `base_channels` value must match `--base-channels` (16 by
default). `pupil_unet_model.py` preserves the original architecture and state
dictionary names, and both training and inference import it directly.

## Files

- `pupil_unet_model.py`: unchanged `ConvBlock` and `TinyUNet` architecture.
- `train_pupil_unet.py`: dataset loading, augmentation, metrics, training,
  fine-tune initialization, early stopping, and checkpoint/history output.
- `ml_pupil_inference.py`: checkpoint loading and mask/ellipse inference.
- `extract_training_frames.py`: video rotation, crop, grayscale resize, sampling,
  preview, and numbered image output (copied from `ML_pupil_detection.py`).
- `video_frame_transform.py`: local frame, point, camera-axis, and camera-matrix
  transforms used by frame extraction.
- `create_pupil_masks.py`: VIA annotation-to-mask conversion and blink masks.
- `prepare_pupil_split.py`: original image-number training/validation split.
- `predict_pupil_masks.py`: batch mask prediction for numbered images.
- `prepare_finetune_split.py`: original/remasked manifest combination and holdout.

No training data or checkpoint is bundled. Copying this folder elsewhere only
requires installing its dependencies and passing paths to those runtime inputs.
