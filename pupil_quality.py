"""Observation data and frame-quality decisions, independent of neural inference.

Start with assess_pupil_quality for one frame, then TemporalQualityTracker for
per-eye recovery. These decisions do not themselves update the 3D model.
The detector imports these shared records; this module never imports the detector.
"""

from dataclasses import dataclass
import math
import time
import numpy as np


@dataclass(frozen=True)
class BoundaryEvidence:
    """Measured image evidence; unknown/warning is not a confirmed blink.

    Fractions describe valid sampled locations, not confidence probabilities.
    A warning can remain eligible for model learning but blocks robot output.
    """
    status: str = "unknown"  # clear, warning, rejected, unknown
    reason: str = "boundary not assessed"
    contour_margin_px: float | None = None
    valid_fraction: float = 0.0
    upper_contrast: float | None = None
    lower_contrast: float | None = None
    upper_interior_difference: float | None = None
    weak_upper_fraction: float | None = None
    border_connection_fraction: float = 0.0


@dataclass(frozen=True)
class EyelidObservation:
    """Experimental image evidence, not a confirmed blink or a model-update gate.

    A supported bright-to-dark boundary can be the upper lid/lash margin.
    States are ``unknown``, ``no_closure_evidence``, ``occlusion_possible``, or
    ``closed_possible``. A single frame cannot distinguish closing from reopening.
    Fields:
        state: Short machine-readable label for the kind of evidence found.
        boundary: Tuple of candidate (x, y) samples in the full rotated frame,
            or None if no suitable edge was located. These are separate points,
            not a continuous anatomical eyelid contour.
        support: Fraction of searched image columns supporting the chosen edge
            (0 to 1). This is not a probability that the eye is closed.
        reason: Human-readable explanation, especially useful for "unknown".

    ``@dataclass`` supplies a constructor for these fields. ``frozen=True``
    prevents reassigning them, so downstream code cannot silently change a
    detection. EyelidObservation(reason="...") uses the other fields' defaults.
    """

    state: str = "unknown"
    boundary: tuple | None = None
    support: float = 0.0
    reason: str = "no supported eyelid edge"


@dataclass(frozen=True)
class PupilObservation:
    """Immutable result for one frame, before camera distortion correction.

    ellipse is ((center_x, center_y), (axis_1, axis_2), angle_degrees) in full
    frame pixels. OpenCV axes are full diameters, not radii. blink means no
    acceptable contour was found; it can also indicate a detection failure.
    confidence is a segmentation/shape score, not a calibrated probability.
    eyelid carries separate experimental image evidence. The legacy blink flag
    is retained for compatibility; eyelid analysis does not overwrite it.
    """

    ellipse: tuple | None
    blink: bool
    confidence: float
    eyelid: EyelidObservation | None = None
    # Optional crop/boundary evidence; baseline explicitly hides this field.
    boundary_rejection: str | None = None
    boundary_evidence: BoundaryEvidence | None = None


@dataclass(frozen=True)
class FrameQualityDecision:
    """A proposed decision about one pupil observation, plus its explanation.

    allow_model_update=True means the observation passes the checks below,
    not that the eye is proven open or the measurement is guaranteed accurate.
    False proposes skipping this frame. Creating this result does not itself
    call, reset, or change the 3D model; the caller decides how to use it.
    """

    allow_model_update: bool
    reason: str
    # The single-frame function keeps these defaults. TemporalQualityTracker
    # adds a quality state and elapsed recovery time without changing old calls.
    state: str = "single_frame"
    recovery_elapsed_s: float = 0.0


def assess_pupil_quality(observation, min_confidence=0.60):
    """Propose accept/skip for one PupilObservation without changing it.

    Inputs:
        observation: The result of PupilDetector.detect(), with a full-frame
            ellipse, pupil confidence, and optional eyelid evidence.
        min_confidence: Whole-observation cutoff in [0, 1], NOT the per-pixel
            mask threshold. The pipeline passes its MIN_CONFIDENCE here so
            this check and the existing eye-model confidence check agree.

    Returns:
        FrameQualityDecision containing a boolean and a human-readable reason.
        Missing/invalid measurements and low confidence propose a skip. With
        those checks passed, possible closure/coverage also proposes a skip.
        Uncertain eyelid evidence falls back to the pupil checks: it does not
        prove closure and should not reject every frame the preview cannot read.

    A malformed configuration raises ValueError; a bad observation instead
    returns a skip decision. This function keeps no history, so each call is
    independent. It does not identify complete blink events or recovery timing.

    Example:
        decision = assess_pupil_quality(pupil, min_confidence=0.60)
        print(decision.allow_model_update, decision.reason)
    """
    try:
        cutoff = float(min_confidence)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("minimum pupil confidence must be finite and between 0 and 1") from None
    if not np.isfinite(cutoff) or not 0.0 <= cutoff <= 1.0:
        raise ValueError("minimum pupil confidence must be finite and between 0 and 1")

    # Validate scores before comparing them: NaN would otherwise evade a normal
    # `confidence < cutoff` check because comparisons with NaN return False.
    try:
        confidence = float(observation.confidence)
    except (TypeError, ValueError, OverflowError):
        return FrameQualityDecision(False, "invalid pupil confidence")
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return FrameQualityDecision(False, "invalid pupil confidence")
    if observation.ellipse is None:
        return FrameQualityDecision(False, "no pupil")

    # An ellipse is ((center_x, center_y), (diameter_a, diameter_b), angle).
    # Reject malformed/nonfinite geometry, including zero or negative diameters.
    # The crop is not an input here, so this cannot assess image-border clipping.
    try:
        (cx, cy), (axis_a, axis_b), angle = observation.ellipse
        geometry = np.asarray([cx, cy, axis_a, axis_b, angle], dtype=float)
        valid_geometry = (geometry.shape == (5,) and np.all(np.isfinite(geometry))
                          and geometry[2] > 0 and geometry[3] > 0)
    except (TypeError, ValueError, OverflowError):
        valid_geometry = False
    if not valid_geometry:
        return FrameQualityDecision(False, "invalid pupil geometry")
    if confidence < cutoff:  # Equality passes, matching EyeModelEstimator.update.
        return FrameQualityDecision(False, "low pupil confidence")

    # Missing pupils/low scores take precedence over uncertain eyelid causes.
    # The legacy `blink` boolean only reports segmentation failure; it is not
    # independent blink evidence, so this decision does not consult it.
    if observation.boundary_rejection is not None:
        return FrameQualityDecision(False, observation.boundary_rejection)
    eyelid_state = observation.eyelid.state if observation.eyelid is not None else "unknown"
    if eyelid_state == "closed_possible":
        return FrameQualityDecision(False, "possible eyelid closure")
    if eyelid_state == "occlusion_possible":
        return FrameQualityDecision(False, "possible pupil coverage")
    if eyelid_state == "no_closure_evidence":
        return FrameQualityDecision(True, "pupil checks passed")
    return FrameQualityDecision(True, "pupil checks passed; eyelid uncertain")


class TemporalQualityTracker:
    """Keep causal quality/recovery history for ONE eye, using video seconds.

    Create a separate instance for each eye. The shared TinyUNet remains
    stateless; this tracker stores only timestamps and recovery status, not
    images or old pupil measurements. Call update for every received frame.

    Any failed single-frame check immediately returns state="rejected". After
    that, consecutive observations must pass both the single-frame checks and
    the higher recovery_confidence threshold for recovery_duration_s before
    state="usable" returns. The waiting state is "recovering". These quality
    states do not claim physiological closing, closed, or reopening phases.

    This is hysteresis: a usable stream may keep passing at min_confidence,
    while recovery needs stronger evidence. Timing defaults are engineering
    settings for evaluation, not research-established values for this model.
    At 30 FPS, a 0.05-second hold needs three good samples spanning 0.0667 s.
    max_gap_s limits the gap between supplied timestamps; pass 1.5 / fps for
    the expected recording cadence. This protects recovery only when those
    timestamps expose a gap. The current pipeline uses frame_index / fps and
    therefore cannot reveal dropped capture frames or variable-rate timing.

    Results are proposals only. This class does not call/reset pye3d, backdate
    decisions, inspect future frames, or infer a blink solely from missing data.
    """

    def __init__(self, min_confidence=0.60, recovery_confidence=0.70,
                 recovery_duration_s=0.05, max_gap_s=0.05):
        """Configure score thresholds (0..1) and time intervals (seconds).

        min_confidence must not exceed recovery_confidence. A zero recovery
        duration allows the first strong observation to resume immediately.
        max_gap_s must be positive; it should reflect the expected sample rate.
        """
        try:
            values = tuple(float(value) for value in (
                min_confidence, recovery_confidence, recovery_duration_s, max_gap_s,
            ))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("quality tracker settings must be finite numbers") from None
        if not all(np.isfinite(value) for value in values):
            raise ValueError("quality tracker settings must be finite numbers")
        low, high, duration, gap = values
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("require 0 <= minimum confidence <= recovery confidence <= 1")
        if duration < 0.0 or gap <= 0.0:
            raise ValueError("recovery duration must be nonnegative and sample gap positive")
        self.min_confidence = low
        self.recovery_confidence = high
        self.recovery_duration_s = duration
        self.max_gap_s = gap
        self.reset()

    def reset(self):
        """Forget this tracker's history when starting/seeking a new recording.

        This resets only quality history; it has no access to the 3D eye model.
        At startup, the first observation can pass the usual single-frame checks.
        """
        self._last_timestamp_s = None
        self._recovery_started_s = None
        self._needs_recovery = False

    def reject_current(self, reason):
        """Invalidate the just-checked frame after coordinate validation.

        Called once after update for the SAME frame, before any pye3d call.
        Keep its timestamp but restart recovery; do not call update twice with
        the same timestamp just to apply a later geometry failure.
        """
        if self._last_timestamp_s is None or not isinstance(reason, str) or not reason.strip():
            raise ValueError("reject_current requires a checked frame and a reason")
        self._needs_recovery = True
        self._recovery_started_s = None
        return FrameQualityDecision(False, reason, state="rejected")

    def update(self, observation, timestamp_s, additional_rejection_reason=None):
        """Return a FrameQualityDecision for the current observation and time.

        observation is a PupilObservation. timestamp_s is its recording time,
        not processing/wall-clock time. It must be finite, nonnegative and
        strictly greater than the previous time. Invalid times raise ValueError
        before changing history; reset() explicitly permits restarting at zero.

        additional_rejection_reason optionally supplies a later-stage failure,
        such as a pupil center that moved outside pye3d's image during lens
        correction. It can only reject an otherwise acceptable pupil; it cannot
        override missing geometry, low confidence, or eyelid evidence. Like any
        other rejection it starts this eye's normal recovery sequence.
        """
        if additional_rejection_reason is None:
            later_rejection = None
        elif isinstance(additional_rejection_reason, str):
            later_rejection = additional_rejection_reason.strip()
            if not later_rejection:
                raise ValueError("additional rejection reason must be nonempty")
        else:
            raise ValueError("additional rejection reason must be text or None")

        try:
            timestamp = float(timestamp_s)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("quality timestamp must be finite and nonnegative") from None
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("quality timestamp must be finite and nonnegative")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ValueError("quality timestamps must strictly increase; reset before seeking")

        instant = assess_pupil_quality(observation, self.min_confidence)
        if instant.allow_model_update and later_rejection is not None:
            instant = FrameQualityDecision(False, later_rejection)
        # A long gap contains no evidence about the intervening eye images.
        # Restart the recovery timer rather than crediting unobserved time.
        if (self._last_timestamp_s is not None
                and timestamp - self._last_timestamp_s > self.max_gap_s + 1e-9):
            self._needs_recovery = True
            self._recovery_started_s = None
        self._last_timestamp_s = timestamp

        if not instant.allow_model_update:
            self._needs_recovery = True
            self._recovery_started_s = None
            return FrameQualityDecision(False, instant.reason, state="rejected")
        if not self._needs_recovery:
            return FrameQualityDecision(True, instant.reason, state="usable")

        # Passing 0.60 alone is enough while usable, but not while recovering.
        # A weaker or rejected sample interrupts the run of strong observations.
        if float(observation.confidence) < self.recovery_confidence:
            self._recovery_started_s = None
            return FrameQualityDecision(False, "waiting for recovery confidence",
                                        state="recovering")
        if self._recovery_started_s is None:
            self._recovery_started_s = timestamp
        elapsed = timestamp - self._recovery_started_s
        # Tiny numerical tolerance prevents rounding (e.g. 0.05 vs 0.049999999)
        # from accidentally imposing an extra video-frame delay.
        if elapsed + 1e-9 >= self.recovery_duration_s:
            self._needs_recovery = False
            self._recovery_started_s = None
            return FrameQualityDecision(True, instant.reason, state="usable")
        return FrameQualityDecision(False, "waiting for stable pupil measurements",
                                    state="recovering", recovery_elapsed_s=elapsed)


@dataclass(frozen=True)
class RobotCommandDecision:
    """Software eligibility only; hardware limits/watchdog remain controller duties."""
    allowed: bool
    reason: str
    expires_at_monotonic_s: float | None = None
    target_robot: tuple[float, float, float] | None = None


class RobotCommandGate:
    """Last check before a future robot adapter; disabled for recorded videos.

    Both eyes are required for binocular control. Capture timestamps must use
    the SAME monotonic clock as this gate (not frame_index/fps or wall time).
    The caller must supply an independently calibrated robot-space target;
    camera pupil normals cannot be sent as robot coordinates. Age/skew defaults
    are engineering starting points, not validated robot safety limits.

    dispatch rechecks at send time, rejects replayed observations, and calls
    invalidate on EVERY blocked update. The downstream controller MUST expire
    old commands autonomously even if this process stops or the camera stalls.
    This class does not implement physical stop behavior or a hardware driver.
    """

    def __init__(self, *, live=False, mapping_verified=False, max_age_s=.1,
                 max_skew_s=.02, clock=time.monotonic):
        if any(not math.isfinite(v) or v <= 0 for v in (max_age_s, max_skew_s)):
            raise ValueError("age/skew limits must be finite and positive")
        self.live = live
        self.mapping_verified = mapping_verified
        self.max_age_s, self.max_skew_s = max_age_s, max_skew_s
        self.clock = clock
        self._last_sent = None

    def evaluate(self, eyes, *, filter_mode, capture_times=None, target_robot=None):
        """eyes = ((left observation, quality, estimate), (right ...))."""
        deny = lambda reason: RobotCommandDecision(False, reason)
        if filter_mode != "filtered":
            return deny("robot output blocked in baseline/comparison mode")
        if not self.live:
            return deny("recorded-video pipeline cannot authorize robot commands")
        if not self.mapping_verified:
            return deny("robot-space calibration has not been verified")
        try:
            now = float(self.clock())
            times = tuple(float(t) for t in capture_times)
            if len(times) != 2 or not math.isfinite(now) or any(not math.isfinite(t) for t in times):
                return deny("missing or invalid monotonic capture timestamps")
        except (TypeError, ValueError, OverflowError):
            return deny("missing or invalid monotonic capture timestamps")
        if any(t > now or now-t >= self.max_age_s for t in times):
            return deny("stale or future eye observations")
        if abs(times[0]-times[1]) > self.max_skew_s:
            return deny("eye observations are not synchronized")
        if self._last_sent is not None and any(t <= old for t, old in zip(times, self._last_sent)):
            return deny("observations already used or out of order")
        if len(eyes) != 2:
            return deny("both eyes are required")
        for side, (pupil, quality, estimate) in zip(("left", "right"), eyes):
            if not quality.allow_model_update or quality.state != "usable":
                return deny(f"{side} eye rejected or recovering")
            if not assess_pupil_quality(pupil).allow_model_update:
                return deny(f"{side} image quality failed")
            evidence = pupil.boundary_evidence
            if evidence is None or evidence.status != "clear":
                return deny(f"{side} boundary evidence is uncertain")
            if pupil.eyelid is None or pupil.eyelid.state != "no_closure_evidence":
                return deny(f"{side} eyelid evidence is uncertain")
            if not estimate.ready or estimate.gaze_direction_camera is None:
                return deny(f"{side} geometry unavailable")
            direction = np.asarray(estimate.gaze_direction_camera, dtype=float)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 0:
                return deny(f"{side} direction invalid")
            if estimate.calibration_identity_status != "matched":
                return deny(f"{side} camera identity unverified")
            if estimate.model_diagnostics is None or estimate.model_diagnostics.range_status != "passed":
                return deny(f"{side} model checks have not passed")
            if estimate.model_stability is None or estimate.model_stability.status != "stable":
                return deny(f"{side} model is not stable")
        try:
            target = tuple(float(v) for v in target_robot)
            if len(target) != 3 or not all(math.isfinite(v) for v in target):
                return deny("invalid robot-space target")
        except (TypeError, ValueError, OverflowError):
            return deny("missing robot-space target")
        return RobotCommandDecision(True, "current binocular checks passed",
                                    min(times)+self.max_age_s, target)

    def dispatch(self, eyes, *, send, invalidate, filter_mode, capture_times=None,
                 target_robot=None):
        """Only this path should call an adapter's send; never reuse a saved decision.

        send receives the target and expiry in the decision. invalidate receives
        the blocked decision and must revoke any previous command in the adapter.
        For disconnected/crashed senders the receiver's watchdog remains essential.
        """
        try:
            decision = self.evaluate(eyes, filter_mode=filter_mode,
                                     capture_times=capture_times, target_robot=target_robot)
        except Exception:
            invalidate(RobotCommandDecision(False, "invalid robot gate input"))
            raise
        if not decision.allowed:
            invalidate(decision)
            return decision
        self._last_sent = tuple(float(t) for t in capture_times)
        try:
            send(decision)
        except Exception:
            invalidate(RobotCommandDecision(False, "robot adapter send failed"))
            raise
        return decision
