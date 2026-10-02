"""Format model measurements and draw diagnostics on rotated video frames.

The pupil observation already uses full-frame display pixels. The estimator
also restores its projected geometry to these distorted display coordinates.
Drawing functions therefore need no calibration or ROI coordinate conversion.
"""

import math
from dataclasses import asdict

import cv2


# OpenCV color tuples use blue, green, red channel order.
EYE_SPHERE_COLOR = (255, 50, 50)
EYE_CENTER_COLOR = (255, 255, 0)
EYE_TO_PUPIL_COLOR = (255, 150, 50)
PUPIL_ELLIPSE_COLOR = (20, 255, 255)
GAZE_RAY_COLOR = (200, 255, 0)


def frame_record(side, frame_index, timestamp_s, estimate, quality_decision, metadata, *, pupil=None):
    """Build one strict-JSON record, including skips and reliability diagnostics.

    ``metadata`` binds results to video/calibration bytes and camera orientation.
    Geometry availability, upstream ranges, temporal consistency and calibration
    identity are separate fields: none alone measures gaze accuracy. Nonfinite
    optional measurements become null, never JSON's nonstandard NaN token.
    Optional ``pupil`` preserves the original detector output, including its
    selected boundary reason and eyelid evidence, on accepted AND skipped
    frames. Omission keeps older callers compatible. Invalid numerical input
    is stored as null alongside the gate's rejection reason.
    """
    record = dict(metadata)
    record.update(asdict(estimate))
    record.update(eye=side, frame_index=int(frame_index), timestamp_s=float(timestamp_s),
                  model_input="accepted" if quality_decision.allow_model_update else "skipped",
                  quality=asdict(quality_decision))

    if pupil is not None:
        # Preserve the detector observation even when the quality gate prevents
        # calibration or a 3D update. This is measured input, not an accepted
        # gaze estimate. Centers/boundary points remain in rotated raw-frame
        # pixels; never crop or drive a robot using corrected/model coordinates.
        record["pupil_observation"] = asdict(pupil)
        record["pupil_observation_coordinate_system"] = "rotated_raw_frame_pixels"

    def strict(value):
        if isinstance(value, dict):
            return {key: strict(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return strict(record)


def _value(value, digits=3):
    """Give missing/nonfinite measurements a readable console/UI placeholder."""
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _xyz(value):
    """Format the three camera-relative coordinates with consistent precision."""
    if value is None:
        return "n/a"
    return "(" + ", ".join(_value(component, 2) for component in value) + ")"


def _pixel_point(point):
    """Round valid floating-point display coordinates for OpenCV drawing."""
    if point is None or len(point) != 2:
        return None
    if not all(math.isfinite(float(value)) for value in point):
        return None
    return tuple(int(round(float(value))) for value in point)


def _draw_text(frame, text, origin, color, scale=0.55):
    """Draw in place with a dark outline for contrast on light eye images."""
    x, y = origin
    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def print_frame_features(side, timestamp_s, estimate, quality_decision=None):
    """Print one eye/frame record; millimeter values use the assumed eye radius.

    Timestamps are video seconds, while ``pye3d_ms`` covers the model call only.
    This is a console diagnostic, not a persisted data export.

    quality_decision is an optional FrameQualityDecision from pupil_detection.
    When supplied, its action reflects the gate actually enforced by the pipeline.
    Omitting it preserves the older three-argument function call.
    """
    quality_text = ""
    if quality_decision is not None:
        action = "accepted" if quality_decision.allow_model_update else "skipped"
        # !r quotes the reason, keeping its spaces readable as one log value.
        quality_text = f" model_input={action} quality_reason={quality_decision.reason!r}"
        quality_text += (f" quality_state={quality_decision.state}"
                         f" recovery_elapsed_s={quality_decision.recovery_elapsed_s:.3f}")
    diagnostics = getattr(estimate, "model_diagnostics", None)
    diagnostic_text = " model_checks=unavailable"
    if diagnostics is not None:
        # These native values use pye3d's radius, before EYE_RADIUS_MM scaling.
        # Comparing our scaled display values to upstream ranges is incorrect.
        diagnostic_text = (
            f" model_checks={diagnostics.range_status}"
            f" model_failed_checks={diagnostics.failed_checks!r}"
            f" model_unavailable_checks={diagnostics.unavailable_checks!r}"
            f" native_eye_center_mm={_xyz(diagnostics.native_eye_center_mm)}"
            f" native_pupil_diameter_mm={_value(diagnostics.native_pupil_diameter_mm)}"
            f" phi_offset_deg={_value(diagnostics.phi_offset_deg)}"
            f" theta_offset_deg={_value(diagnostics.theta_offset_deg)}"
        )
    stability = getattr(estimate, "model_stability", None)
    stability_text = " temporal_consistency=unavailable"
    if stability is not None:
        stability_text = (f" temporal_consistency={stability.status}"
                          f" native_center_displacement_mm={_value(stability.displacement_mm)}"
                          f" native_center_spread_mm={_value(stability.recent_spread_mm)}")
    calibration_text = f" calibration_identity={getattr(estimate, 'calibration_identity_status', 'unverified')}"
    print(
        f"{side} t={timestamp_s:.3f}s "
        f"pupil_diameter_mm={_value(estimate.pupil_diameter_mm)} "
        f"eye_center_mm={_xyz(estimate.eye_center_mm)} "
        f"pupil_center_mm={_xyz(estimate.pupil_center_mm)} "
        f"pupil_confidence={estimate.pupil_confidence:.3f} "
        f"model_confidence={_value(estimate.model_confidence)} "
        f"pye3d_ms={estimate.update_time_ms:.2f} "
        f"status={estimate.status}{quality_text}{diagnostic_text}{stability_text}{calibration_text}",
        flush=True,
    )


def draw_pupil_ellipse(frame, pupil_observation):
    """Draw the measured ellipse in place, before lens-distortion correction."""
    if pupil_observation.ellipse is not None:
        cv2.ellipse(
            frame,
            pupil_observation.ellipse,
            PUPIL_ELLIPSE_COLOR,
            2,
            cv2.LINE_AA,
        )


def draw_pye3d_features(frame, estimate):
    """Draw available model projections and a direction cue in place.

    The sphere outline comes from pye3d's projection and display redistortion;
    its size is in pixels, not a fixed 12 mm drawing radius. The extended line
    doubles the 2D center-to-center displacement. It is a visual cue, not a
    calibrated gaze estimate or a projection of a 3D gaze ray.
    """
    if not estimate.ready:
        return

    if estimate.projected_eye_sphere is not None:
        cv2.ellipse(
            frame,
            estimate.projected_eye_sphere,
            EYE_SPHERE_COLOR,
            2,
            cv2.LINE_AA,
        )

    eye_center = _pixel_point(estimate.projected_eye_center)
    pupil_center = _pixel_point(estimate.projected_pupil_center)
    if eye_center is not None:
        cv2.circle(frame, eye_center, 8, EYE_CENTER_COLOR, -1, cv2.LINE_AA)
    if eye_center is None or pupil_center is None:
        return

    cv2.line(
        frame,
        eye_center,
        pupil_center,
        EYE_TO_PUPIL_COLOR,
        2,
        cv2.LINE_AA,
    )
    direction_x = pupil_center[0] - eye_center[0]
    direction_y = pupil_center[1] - eye_center[1]
    extended = (
        eye_center[0] + 2 * direction_x,
        eye_center[1] + 2 * direction_y,
    )
    cv2.line(
        frame,
        pupil_center,
        extended,
        GAZE_RAY_COLOR,
        3,
        cv2.LINE_AA,
    )


def create_output_frame(frame, side, roi, pupil_observation, estimate,
                        quality_decision=None):
    """Copy the frame, overlay the full-frame ROI/features, and label status.

    A missing pupil takes display priority over model status. "Ready" reflects
    available geometry, not convergence; model confidence is shown separately.

    Inputs:
        frame: Full rotated BGR image, a (height, width, 3) uint8 array.
        side: Display label such as "Left eye".
        roi: (x, y, width, height) rectangle in that full frame's pixels.
        pupil_observation: PupilObservation including optional eyelid data.
        estimate: Current EyeModelEstimate, providing 3D status and projections.
        quality_decision: Optional FrameQualityDecision containing the enforced
            accept/skip action and reason. The pipeline applies it before this call.

    Returns a NEW image with overlays. The source image and observations stay
    unchanged; display drawing must not alter future measurements.
    """
    output = frame.copy()
    x, y, width, height = (int(value) for value in roi)
    cv2.rectangle(output, (x, y), (x + width, y + height), (255, 0, 0), 1)
    draw_pupil_ellipse(output, pupil_observation)
    draw_pye3d_features(output, estimate)

    if pupil_observation.ellipse is None:
        # A missing segmentation is not independent evidence of a blink.
        status = "no pupil"
        color = (0, 180, 255)
    elif estimate.ready:
        diagnostics = getattr(estimate, "model_diagnostics", None)
        checks = "unavailable" if diagnostics is None else diagnostics.range_status
        status = f"geometry available | model checks {checks}"
        color = (0, 220, 0) if checks == "passed" else (0, 180, 255)
    else:
        status = estimate.status
        color = (0, 180, 255)
    _draw_text(output, f"{side}: {status}", (10, 30), color, 0.65)
    confidence_text = (
        f"pupil {estimate.pupil_confidence:.2f}  "
        f"model {_value(estimate.model_confidence, 2)}"
    )
    _draw_text(output, confidence_text, (10, 56), color, 0.5)
    stability = getattr(estimate, "model_stability", None)
    if stability is not None:
        _draw_text(output,
                   f"temporal consistency: {stability.status} | displacement "
                   f"{_value(stability.displacement_mm, 2)} native mm (not accuracy)",
                   (10, 186), (0, 180, 255), 0.5)
    identity = getattr(estimate, "calibration_identity_status", "unverified")
    _draw_text(output, f"camera identity: {identity} | gaze accuracy unmeasured",
               (10, 212), (0, 180, 255), 0.5)
    diagnostics = getattr(estimate, "model_diagnostics", None)
    if diagnostics is not None:
        details = []
        if diagnostics.failed_checks:
            details.append("outside default range: " + ", ".join(diagnostics.failed_checks))
        if diagnostics.unavailable_checks:
            details.append("unavailable: " + ", ".join(diagnostics.unavailable_checks))
        if details:
            _draw_text(output, " | ".join(details), (10, 160), (0, 180, 255), 0.5)
    # Eyelid evidence is attached by PupilDetector.detect(). None also supports
    # observations created before that optional field was introduced.
    eyelid = pupil_observation.eyelid
    if eyelid is not None:
        # Map the short code labels to readable text. "Unknown" is uncertainty,
        # not a claim that the eye is open; support is column coverage, not odds.
        labels = {
            "unknown": "uncertain",
            "no_closure_evidence": "no closure evidence",
            "occlusion_possible": "possible pupil coverage",
            "closed_possible": "possibly closed",
        }
        # Eyelid evidence remains diagnostic, separate from the model's ready status.
        eyelid_color = (180, 100, 255)  # OpenCV uses BGR, giving purple here.
        label = labels.get(eyelid.state, "uncertain")
        text = f"eyelid preview: {label} | edge support {eyelid.support:.2f}"
        _draw_text(output, text, (10, 82), eyelid_color, 0.5)
        if eyelid.boundary:
            # Show measured samples; connecting the two flanks would invent a
            # measured boundary across the pupil that this method excludes.
            for sample in eyelid.boundary:
                # The detector already restored the ROI offset. Convert the
                # floating-point full-frame coordinates to integers for drawing;
                # adding the ROI origin again here would shift them incorrectly.
                point = _pixel_point(sample)
                if point is not None:
                    # Radius 4 pixels; thickness -1 fills the circle. LINE_AA
                    # smooths its drawn edge and has no role in detection itself.
                    cv2.circle(output, point, 4, eyelid_color, -1, cv2.LINE_AA)
    if quality_decision is not None:
        action = "accepted" if quality_decision.allow_model_update else "skipped"
        quality_text = f"model input: {action}: {quality_decision.reason}"
        quality_color = (0, 220, 0) if quality_decision.allow_model_update else (0, 180, 255)
        _draw_text(output, quality_text, (10, 108), quality_color, 0.5)
        if quality_decision.state != "single_frame":
            # These are measurement-quality states, not confirmed blink phases.
            state_text = f"quality state: {quality_decision.state}"
            if quality_decision.state == "recovering":
                state_text += f" | stable samples span {quality_decision.recovery_elapsed_s * 1000:.0f} ms"
            _draw_text(output, state_text, (10, 134), quality_color, 0.5)
    return output
