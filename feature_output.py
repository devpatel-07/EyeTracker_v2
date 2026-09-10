# Import dependencies

import math

import cv2


EYE_SPHERE_COLOR = (255, 50, 50)
EYE_CENTER_COLOR = (255, 255, 0)
EYE_TO_PUPIL_COLOR = (255, 150, 50)
PUPIL_ELLIPSE_COLOR = (20, 255, 255)
GAZE_RAY_COLOR = (200, 255, 0)


def _value(value, digits=3):
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _xyz(value):
    if value is None:
        return "n/a"
    return "(" + ", ".join(_value(component, 2) for component in value) + ")"


def _pixel_point(point):
    if point is None or len(point) != 2:
        return None
    if not all(math.isfinite(float(value)) for value in point):
        return None
    return tuple(int(round(float(value))) for value in point)


def _draw_text(frame, text, origin, color, scale=0.55):
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


# Displays per frame pupil diameter, pye3d pupil and eye center coordinates, and confidence metrics - function

def print_frame_features(side, timestamp_s, estimate):
    print(
        f"{side} t={timestamp_s:.3f}s "
        f"pupil_diameter_mm={_value(estimate.pupil_diameter_mm)} "
        f"eye_center_mm={_xyz(estimate.eye_center_mm)} "
        f"pupil_center_mm={_xyz(estimate.pupil_center_mm)} "
        f"pupil_confidence={estimate.pupil_confidence:.3f} "
        f"model_confidence={_value(estimate.model_confidence)} "
        f"pye3d_ms={estimate.update_time_ms:.2f} "
        f"status={estimate.status}",
        flush=True,
    )


# Displays uncalibrated pupil ellipse onto output frame - function

def draw_pupil_ellipse(frame, pupil_observation):
    if pupil_observation.ellipse is not None:
        cv2.ellipse(
            frame,
            pupil_observation.ellipse,
            PUPIL_ELLIPSE_COLOR,
            2,
            cv2.LINE_AA,
        )


# Displays pye3d features onto output frame. This includes a dot at the eye center projected onto the 2d frame. A circle with 12 mm radius
# around the eye center. A ray with twice the length from the eye center to pupil center to visualize the gaze vector - function

def draw_pye3d_features(frame, estimate):
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


def create_output_frame(frame, side, roi, pupil_observation, estimate):
    output = frame.copy()
    x, y, width, height = (int(value) for value in roi)
    cv2.rectangle(output, (x, y), (x + width, y + height), (255, 0, 0), 1)
    draw_pupil_ellipse(output, pupil_observation)
    draw_pye3d_features(output, estimate)

    if pupil_observation.blink:
        status = "no pupil / blink"
        color = (0, 180, 255)
    elif estimate.ready:
        status = "pye3d ready"
        color = (0, 220, 0)
    else:
        status = estimate.status
        color = (0, 180, 255)
    _draw_text(output, f"{side}: {status}", (10, 30), color, 0.65)
    confidence_text = (
        f"pupil {estimate.pupil_confidence:.2f}  "
        f"model {_value(estimate.model_confidence, 2)}"
    )
    _draw_text(output, confidence_text, (10, 56), color, 0.5)
    return output
