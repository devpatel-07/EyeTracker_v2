"""Per-eye pupil crops, independent of calibration and 3D model history.

The shared neural detector stays stateless. Create one tracker for each eye.
Tracking uses raw rotated-frame pixels; corrected gaze/model coordinates must
never be used to cut pixels from an image. This module does not authorize 3D
updates: the pipeline's blink/recovery gate still makes that decision.
"""
import math

from pupil_quality import assess_pupil_quality


def ellipse_bounds(ellipse):
    """Axis-aligned bounds of a rotated OpenCV ellipse with diameter axes."""
    (cx, cy), (a, b), angle = ellipse
    if not all(math.isfinite(v) for v in (cx, cy, a, b, angle)) or min(a, b) <= 0:
        raise ValueError("ellipse must have finite coordinates and positive diameters")
    theta = math.radians(angle)
    rx = .5 * math.hypot(a * math.cos(theta), b * math.sin(theta))
    ry = .5 * math.hypot(a * math.sin(theta), b * math.cos(theta))
    return cx-rx, cy-ry, cx+rx, cy+ry


def ellipse_fits(ellipse, roi, margin=0.):
    """Whether the complete fitted oval plus margin fits in this pixel crop.

    Containment of the fitted oval is a tracking check, not proof that a real
    pupil is fully visible. Raw-image blink/boundary checks remain necessary.
    """
    if ellipse is None:
        return False
    try:
        left, top, right, bottom = ellipse_bounds(ellipse)
    except (ValueError, TypeError, OverflowError):
        return False
    x, y, w, h = roi
    return (left-margin >= x and top-margin >= y
            and right+margin <= x+w-1 and bottom+margin <= y+h-1)


class PupilRoiTracker:
    """Try one predicted crop, then at most one broad retry on this frame.

    ``search_roi`` is the unchanged broad eye crop used for human labels and
    eyelid evidence. ``min_confidence`` only controls this 2D tracker. Its
    decisions do not depend on baseline/filtered mode or pye3d reliability,
    so paired runs receive the same detections. A blink/loss clears the track
    and keeps searching broadly through ``recovery_s`` of good observations.
    The neural input resolution is unchanged: a crop is not a promised speedup.
    """
    def __init__(self, search_roi, min_confidence=.7, recovery_s=.05,
                 max_gap_s=.1, refresh_s=.5):
        if len(search_roi) != 4 or any(type(v) is not int for v in search_roi):
            raise ValueError("search ROI must contain four integer pixel values")
        x, y, w, h = search_roi
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError("invalid search ROI")
        if (not all(math.isfinite(v) for v in
                    (min_confidence, recovery_s, max_gap_s, refresh_s))
                or not 0 < min_confidence <= 1 or recovery_s < 0
                or min(max_gap_s, refresh_s) <= 0):
            raise ValueError("invalid tracking confidence or time limits")
        self.search_roi = tuple(search_roi)
        self.min_confidence = min_confidence
        self.recovery_s, self.max_gap_s, self.refresh_s = recovery_s, max_gap_s, refresh_s
        self.last = self.previous = None
        self.last_timestamp = self.last_search = None
        self.recover_until = -math.inf

    def _choose(self, timestamp):
        if self.last is None or timestamp-self.last_search >= self.refresh_s:
            return self.search_roi
        ellipse, seen = self.last
        left, top, right, bottom = ellipse_bounds(ellipse)
        cx, cy = ellipse[0]
        move_x = move_y = 0.
        if self.previous is not None:
            old, old_time = self.previous
            dt = seen-old_time
            if dt > 0:
                ahead = min(1., max(0., (timestamp-seen)/dt))
                move_x = (cx-old[0][0])*ahead
                move_y = (cy-old[0][1])*ahead
        # Reserve half a pupil diameter plus predicted motion on EACH side.
        # Keep the broad crop's aspect ratio to avoid an extra shape stretch.
        padding = max(24., .5*max(ellipse[1]))
        w = right-left+2*(padding+abs(move_x))
        h = bottom-top+2*(padding+abs(move_y))
        sx, sy, sw, sh = self.search_roi
        w, h = max(w, h*sw/sh), max(h, w*sh/sw)
        w, h = min(sw, math.ceil(w)), min(sh, math.ceil(h))
        x = max(sx, min(sx+sw-w, math.floor(cx+move_x-w/2)))
        y = max(sy, min(sy+sh-h, math.floor(cy+move_y-h/2)))
        roi = (x, y, w, h)
        if w*h >= .85*sw*sh or not ellipse_fits(ellipse, roi, 3):
            return self.search_roi
        return roi

    def detect(self, detector, frame, timestamp_s):
        """Return (PupilObservation, tracking diagnostics) for the current frame.

        The narrow result is discarded before any temporal/model gate if it
        fails image quality or lies too close to its own crop edge. A broad
        retry prevents tracking loss from being mistaken for a confirmed blink.
        Only the final current observation leaves this function; no old pupil
        measurement is reused as a fresh result.
        """
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("tracking timestamp must be finite")
        x, y, w, h = self.search_roi
        if x+w > frame.shape[1] or y+h > frame.shape[0]:
            raise ValueError("search ROI does not fit frame")
        if self.last_timestamp is not None and (
                timestamp <= self.last_timestamp or timestamp-self.last_timestamp > self.max_gap_s):
            self.last = self.previous = None
            self.recover_until = timestamp+self.recovery_s
        requested = self._choose(timestamp)
        chosen = requested
        fallback = None
        calls = 1
        if chosen == self.search_roi:
            pupil = detector.detect(frame, chosen)
        else:
            pupil = detector.detect(frame, chosen, evidence_roi=self.search_roi)
            quality = assess_pupil_quality(pupil, self.min_confidence)
            if not quality.allow_model_update:
                fallback = quality.reason
            elif not ellipse_fits(pupil.ellipse, chosen, max(3., .05*min(pupil.ellipse[1]))):
                fallback = "fitted pupil too close to tracking crop edge"
            if fallback:
                chosen = self.search_roi
                pupil = detector.detect(frame, chosen)
                calls += 1
        if chosen == self.search_roi:
            self.last_search = timestamp
        quality = assess_pupil_quality(pupil, self.min_confidence)
        if not quality.allow_model_update:
            self.last = self.previous = None
            self.recover_until = timestamp+self.recovery_s
        elif timestamp >= self.recover_until:
            self.previous, self.last = self.last, (pupil.ellipse, timestamp)
        self.last_timestamp = timestamp
        return pupil, dict(mode="tracked", requested_roi=list(requested),
                           detection_roi=list(chosen), search_roi=list(self.search_roi),
                           fallback_reason=fallback, network_calls=calls,
                           area_fraction=chosen[2]*chosen[3]/(w*h))
