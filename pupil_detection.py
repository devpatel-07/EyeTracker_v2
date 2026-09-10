# Import dependencies

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.torch_version import TorchVersion


class _ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


# Defines TinyUNet model used for pupil segmentation - class

class TinyUNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        channels = int(base_channels)
        self.encoder1 = _ConvBlock(1, channels)
        self.encoder2 = _ConvBlock(channels, channels * 2)
        self.encoder3 = _ConvBlock(channels * 2, channels * 4)
        self.bottleneck = _ConvBlock(channels * 4, channels * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, 2, stride=2)
        self.decoder3 = _ConvBlock(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, 2, stride=2)
        self.decoder2 = _ConvBlock(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, 2, stride=2)
        self.decoder1 = _ConvBlock(channels * 2, channels)
        self.output = nn.Conv2d(channels, 1, 1)

    def forward(self, inputs):
        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))
        decoder3 = self.decoder3(torch.cat((self.up3(bottleneck), encoder3), 1))
        decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), 1))
        decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), 1))
        return self.output(decoder1)


# Stores pupil ellipse, blink state, and confidence - data class

@dataclass(frozen=True)
class PupilObservation:
    ellipse: tuple | None
    blink: bool
    confidence: float


# Converts ROI to grayscale and resizes image for TinyUNet - function

def prepare_tinyunet_input(roi, input_size, device):
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    elif roi.ndim == 2:
        gray = roi
    else:
        raise ValueError("pupil ROI must be a grayscale or BGR image")
    resized = cv2.resize(gray, input_size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    return tensor[None, None].to(device)


# Runs TinyUNet and generates probability map - function

def generate_probability_map(model, tensor, roi_size):
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))
    probability = probabilities[0, 0].cpu().numpy()
    return cv2.resize(probability, roi_size, interpolation=cv2.INTER_LINEAR)


# Converts probability map into binary pupil mask using confidence threshold - function

def create_pupil_mask(probability, threshold):
    return (probability >= float(threshold)).astype(np.uint8) * 255


# Finds pupil contours and removes contours outside expected pupil size - function

def select_pupil_contour(mask, min_area_ratio=0.002, max_area_ratio=0.35):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    roi_area = int(mask.shape[0] * mask.shape[1])
    minimum_area = max(20.0, roi_area * float(min_area_ratio))
    maximum_area = roi_area * float(max_area_ratio)
    candidates = [
        contour
        for contour in contours
        if len(contour) >= 5
        and minimum_area <= cv2.contourArea(contour) <= maximum_area
    ]
    return max(candidates, key=cv2.contourArea) if candidates else None


# Fits ellipse to selected pupil contour - function

def fit_pupil_ellipse(contour, roi_origin):
    (center_x, center_y), axes, angle = cv2.fitEllipse(contour)
    x, y = roi_origin
    local_ellipse = ((center_x, center_y), axes, angle)
    full_ellipse = ((center_x + x, center_y + y), axes, angle)
    return local_ellipse, full_ellipse


# Calculates pupil confidence using ML probability and contour/ellipse agreement - function

def calculate_pupil_confidence(probability, mask, contour, local_ellipse):
    contour_mask = np.zeros_like(mask)
    ellipse_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    cv2.ellipse(ellipse_mask, local_ellipse, 255, -1)
    contour_pixels = contour_mask > 0
    ellipse_pixels = ellipse_mask > 0
    union = np.count_nonzero(contour_pixels | ellipse_pixels)
    intersection = np.count_nonzero(contour_pixels & ellipse_pixels)
    agreement = intersection / union if union else 0.0
    mean_probability = (
        float(np.mean(probability[contour_pixels]))
        if np.any(contour_pixels)
        else 0.0
    )
    return float(np.clip(mean_probability * agreement, 0.0, 1.0))


class PupilDetector:
    def __init__(
        self,
        model,
        device,
        mask_threshold,
        input_size=(320, 192),
        min_area_ratio=0.002,
        max_area_ratio=0.35,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.input_size = tuple(int(value) for value in input_size)
        self.mask_threshold = float(mask_threshold)
        self.min_area_ratio = float(min_area_ratio)
        self.max_area_ratio = float(max_area_ratio)
        if len(self.input_size) != 2 or any(value <= 0 for value in self.input_size):
            raise ValueError("TinyUNet input dimensions must be positive")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask threshold must be between zero and one")

    def warm_up(self):
        width, height = self.input_size
        example = torch.zeros((1, 1, height, width), device=self.device)
        with torch.inference_mode():
            self.model(example)

    # Organizes pupil detection operations and returns pupil observation - function

    def detect(self, frame, roi_rect):
        frame_height, frame_width = frame.shape[:2]
        x, y, width, height = (int(value) for value in roi_rect)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("ROI must contain nonnegative x/y and positive size")
        if x + width > frame_width or y + height > frame_height:
            raise ValueError(
                f"ROI {roi_rect} does not fit frame {(frame_width, frame_height)}"
            )

        roi = frame[y:y + height, x:x + width]
        tensor = prepare_tinyunet_input(roi, self.input_size, self.device)
        probability = generate_probability_map(
            self.model,
            tensor,
            (width, height),
        )
        mask = create_pupil_mask(probability, self.mask_threshold)
        contour = select_pupil_contour(
            mask,
            self.min_area_ratio,
            self.max_area_ratio,
        )
        if contour is None:
            return PupilObservation(None, True, 0.0)

        local_ellipse, full_ellipse = fit_pupil_ellipse(contour, (x, y))
        confidence = calculate_pupil_confidence(
            probability,
            mask,
            contour,
            local_ellipse,
        )
        return PupilObservation(full_ellipse, False, confidence)


# Loads TinyUNet checkpoint and warms model before Frame Loop - function

def load_pupil_detector(checkpoint_path, device, mask_threshold):
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"TinyUNet checkpoint not found: {path}")
    if device == "auto":
        selected_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    # Keeps restricted loading enabled while allowing stored PyTorch version metadata
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            path,
            map_location=selected_device,
            weights_only=True,
        )
    try:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]
        base_channels = int(config["base_channels"])
        input_size = (int(config["input_width"]), int(config["input_height"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"TinyUNet checkpoint has invalid metadata: {error}") from error

    model = TinyUNet(base_channels=base_channels)
    model.load_state_dict(state_dict)
    detector = PupilDetector(
        model,
        selected_device,
        input_size=input_size,
        mask_threshold=mask_threshold,
    )
    detector.warm_up()
    return detector
