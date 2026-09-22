# Import dependencies

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import time

@dataclass(frozen=True)
class PupilObservation:
    ellipse: tuple | None
    blink: bool
    confidence: float

def fit_pupil_ellipse(contour, roi_origin):
    (center_x, center_y), axes, angle = cv2.fitEllipse(contour)
    x, y = roi_origin
    full_ellipse = ((center_x + x, center_y + y), axes, angle)
    return full_ellipse

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


class PupilDetector:
    #cache kernal here
    def __init__(self):
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        self.iterations = 2


    def detect(self, frame, roi_rect):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        x, y, width, height = (int(value) for value in roi_rect)
        roi = frame[y:y + height, x:x + width]

        _, pupil_mask = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
        cv2.morphologyEx(
            pupil_mask, 
            cv2.MORPH_CLOSE, 
            self.kernel, 
            dst=pupil_mask, # dont allocate new object. work in place
            iterations=self.iterations,

        )
        contour = select_pupil_contour(pupil_mask)
        if contour is None:
            return PupilObservation(None, True, 0.0)
        
        full_ellipse = fit_pupil_ellipse(contour, (x, y))
        
        return PupilObservation(full_ellipse, False, 1.0)


def load_pupil_detector(checkpoint_path, device, mask_threshold):
    return PupilDetector();