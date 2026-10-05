# Import dependencies

import csv
import math


CSV_COLUMNS = (
    "frame",
    "time_s",
    "left_eye_x",
    "left_eye_y",
    "left_eye_z",
    "left_gaze_x",
    "left_gaze_y",
    "left_gaze_z",
    "right_eye_x",
    "right_eye_y",
    "right_eye_z",
    "right_gaze_x",
    "right_gaze_y",
    "right_gaze_z",
    "gaze_x",
    "gaze_y",
    "gaze_z",
    "gaze_miss_mm",
    "gaze_status",
)


# Keeps unavailable vector components blank - function

def _vector_components(vector):
    if vector is None:
        return (None, None, None)
    try:
        components = tuple(vector)
    except TypeError:
        return (None, None, None)
    if len(components) != 3:
        return (None, None, None)

    values = []
    for component in components:
        try:
            valid = math.isfinite(component)
        except (TypeError, ValueError, OverflowError):
            valid = False
        values.append(component if valid else None)
    return tuple(values)


# Records numerical outputs without changing eye geometry - class

class DataLogger:
    def __init__(self, output_path):
        self.file = open(output_path, "w", newline="", encoding="utf-8")
        try:
            self.writer = csv.DictWriter(self.file, fieldnames=CSV_COLUMNS)
            self.writer.writeheader()
        except Exception:
            self.file.close()
            raise

    def log_frame(
        self,
        frame,
        time_s,
        left_eye_center=None,
        left_gaze=None,
        right_eye_center=None,
        right_gaze=None,
        gaze=None,
    ):
        row = {"frame": frame, "time_s": time_s}

        # Gaze point is in head frame coordinates. Eye centers and gaze directions are in each eye camera's frame
        gaze_point = None
        if gaze is not None:
            gaze_point = gaze.point_mm
            row["gaze_miss_mm"] = gaze.miss_distance_mm
            row["gaze_status"] = gaze.status

        for prefix, vector in (
            ("left_eye", left_eye_center),
            ("left_gaze", left_gaze),
            ("right_eye", right_eye_center),
            ("right_gaze", right_gaze),
            ("gaze", gaze_point),
        ):
            for axis, value in zip(("x", "y", "z"), _vector_components(vector)):
                row[f"{prefix}_{axis}"] = value
        self.writer.writerow(row)

    def close(self):
        self.file.close()
