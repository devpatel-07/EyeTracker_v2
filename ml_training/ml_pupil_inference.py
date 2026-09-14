"""TinyUNet checkpoint loading and pupil-mask inference."""

from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from .pupil_unet_model import TinyUNet
else:
    from pupil_unet_model import TinyUNet


class MLPupilDetector:
    def __init__(
        self,
        model,
        device,
        input_size=(320, 192),
        threshold=0.5,
        min_area_ratio=0.002,
        max_area_ratio=0.35,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.input_size = tuple(input_size)
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device="auto", threshold=0.5):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ML pupil checkpoint not found: {checkpoint_path}")

        if device == "auto":
            selected_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            selected_device = torch.device(device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=selected_device,
            weights_only=False,
        )
        config = checkpoint["config"]
        model = TinyUNet(base_channels=config["base_channels"])
        model.load_state_dict(checkpoint["model_state_dict"])

        detector = cls(
            model=model,
            device=selected_device,
            input_size=(config["input_width"], config["input_height"]),
            threshold=threshold,
        )
        detector.warm_up()
        return detector

    def warm_up(self):
        width, height = self.input_size
        example = torch.zeros((1, 1, height, width), device=self.device)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        with torch.inference_mode():
            self.model(example)

    def predict_probability_map(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        model_input = torch.from_numpy(
            resized.astype(np.float32) / 255.0
        )[None, None].to(self.device)

        with torch.inference_mode():
            logits = self.model(model_input)
            probabilities = torch.sigmoid(logits)

        probability_map = probabilities[0, 0].to("cpu").numpy()
        return cv2.resize(
            probability_map,
            (roi.shape[1], roi.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    def predict_mask(self, roi):
        probability_map = self.predict_probability_map(roi)
        return (probability_map >= self.threshold).astype(np.uint8) * 255

    def detect_with_confidence(self, frame, rect, debug=False):
        frame_height, frame_width = frame.shape[:2]
        x = max(0, int(rect["x"]))
        y = max(0, int(rect["y"]))
        right = min(frame_width, x + max(0, int(rect["w"])))
        bottom = min(frame_height, y + max(0, int(rect["h"])))
        width = right - x
        height = bottom - y

        display = frame.copy()
        if width <= 0 or height <= 0:
            return display, 0, True, 0.0

        roi = frame[y:bottom, x:right]
        probability_map = self.predict_probability_map(roi)
        mask = (probability_map >= self.threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        roi_area = width * height
        minimum_area = max(20.0, roi_area * self.min_area_ratio)
        maximum_area = roi_area * self.max_area_ratio
        candidates = [
            contour
            for contour in contours
            if len(contour) >= 5
            and minimum_area <= cv2.contourArea(contour) <= maximum_area
        ]

        cv2.rectangle(display, (x, y), (right, bottom), (255, 0, 0), 2)
        if not candidates:
            if debug:
                return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0, True, 0.0
            return display, 0, True, 0.0

        contour = max(candidates, key=cv2.contourArea)
        (center_x, center_y), axes, angle = cv2.fitEllipse(contour)
        local_ellipse = ((center_x, center_y), axes, angle)
        full_ellipse = ((center_x + x, center_y + y), axes, angle)

        contour_mask = np.zeros_like(mask)
        ellipse_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        cv2.ellipse(ellipse_mask, local_ellipse, 255, -1)
        contour_pixels = contour_mask > 0
        union = np.count_nonzero((contour_mask > 0) | (ellipse_mask > 0))
        intersection = np.count_nonzero(
            (contour_mask > 0) & (ellipse_mask > 0)
        )
        ellipse_agreement = intersection / union if union else 0.0
        mean_probability = (
            float(np.mean(probability_map[contour_pixels]))
            if np.any(contour_pixels)
            else 0.0
        )
        confidence = float(
            np.clip(mean_probability * ellipse_agreement, 0.0, 1.0)
        )

        cv2.ellipse(display, full_ellipse, (0, 255, 0), 2)

        if debug:
            debug_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(debug_display, local_ellipse, (0, 255, 0), 2)
            return debug_display, full_ellipse, False, confidence
        return display, full_ellipse, False, confidence

    def detect(self, frame, rect, debug=False):
        display, ellipse, blink, _ = self.detect_with_confidence(
            frame, rect, debug=debug
        )
        return display, ellipse, blink
