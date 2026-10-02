"""Compare exported gaze directions with independent, frame-aligned references.

This is an offline validation tool, not another stage in the video pipeline.
The pipeline writes one JSON object per eye/frame to a JSONL file. A separate
target JSON document describes directions measured independently of that gaze
output. Both directions must use the *processed eye camera* coordinate system:
the camera axes after the configured video rotation, with the same sign
conventions as ``gaze_direction_camera`` in the export. Screen/world target
positions cannot be copied here directly: camera extrinsics, the target's 3D
position, and the appropriate eye origin are needed to construct those rays.

Target JSON schema (all metadata maps are keyed by ``left`` and/or ``right``)::

    {
      "schema_version": 1,
      "data_kind": "real_reference",
      "direction_source": "Describe how the reference rays were measured",
      "recording_id": "Exact recording_id from the pipeline export",
      "coordinate_system": "processed_eye_camera",
      "camera_ids": {"left": "physical-camera-serial"},
      "calibration_sha256": {"left": "64 hexadecimal characters"},
      "frame_rotation": {"left": "Exact exported rotation value"},
      "calibration_rotation": {"left": "Exact exported rotation value"},
      "samples": [
        {"eye": "left", "frame_index": 120, "timestamp_s": 2.0,
         "direction_camera": [0.0, 0.0, -1.0], "split": "validation"}
      ]
    }

Use ``data_kind: "synthetic"`` for generated fixtures. Synthetic results test
software and geometry; they are not evidence of real-world gaze accuracy.
Every target requires an exact frame match, including its timestamp. Missing,
skipped, and startup estimates are never filled using adjacent frames.

An optional per-eye rotation maps optical directions toward visual reference
directions. Only earlier ``calibration`` samples fit that rotation; disjoint,
later ``validation`` samples determine every reported error. This limited
rotation cannot correct lens distortion, drifting eye centers, or parallax.

Sources:
* Pupil Labs, camera/eye coordinate definitions:
  https://docs.pupil-labs.com/core/terminology/#coordinate-system
* Pupil Labs, separate calibration and accuracy testing:
  https://docs.pupil-labs.com/core/software/pupil-capture/#gaze-mapping-and-accuracy
* Lawrence, Bernal & Witzgall (2019), proper Kabsch/Umeyama rotation:
  https://doi.org/10.6028/jres.124.028

Example::

    python gaze_validation.py --records gaze.jsonl --targets targets.json \
        --output evaluation.json --fit-rotation
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Any

import numpy as np


COORDINATE_SYSTEM = "processed_eye_camera"
TIMESTAMP_TOLERANCE_S = 1e-6
_EYES = {"left", "right"}
_ROTATIONS = {"none", "clockwise", "180", "counterclockwise"}
_IDENTITY_MAPS = {
    "camera_ids": "camera_id",
    "calibration_sha256": "calibration_sha256",
    "frame_rotation": "frame_rotation",
    "calibration_rotation": "calibration_rotation",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _frame(value: Any, name: str) -> int:
    # bool is an int subclass in Python, but True is not a frame index.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _timestamp(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _direction(value: Any, name: str) -> np.ndarray:
    """Normalize a ray without overflowing when its finite components are large."""
    # Check the original values before NumPy coercion: [True, 0, 1] and
    # ["1", 0, 1] otherwise silently become floating-point vectors.
    try:
        components = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must have three finite numeric components") from exc
    if len(components) != 3 or any(
        isinstance(component, (bool, np.bool_)) or not isinstance(component, Real)
        for component in components
    ):
        raise ValueError(f"{name} must have three finite numeric components")
    try:
        vector = np.asarray(components, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have three finite numeric components") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must have three finite numeric components")
    scale = float(np.max(np.abs(vector)))
    if scale == 0:
        raise ValueError(f"{name} must be a nonzero direction")
    scaled = vector / scale
    return scaled / np.linalg.norm(scaled)


def _digest(value: Any, name: str) -> str:
    value = _text(value, name)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _rotation(value: Any, name: str) -> str:
    value = _text(value, name)
    if value not in _ROTATIONS:
        raise ValueError(f"{name} must be one of {', '.join(sorted(_ROTATIONS))}")
    return value


def fit_direction_rotation(
    predicted: Iterable[Iterable[float]], reference: Iterable[Iterable[float]]
) -> np.ndarray:
    """Fit a proper rotation R satisfying ``reference ~= R @ predicted``.

    Directions are unit rays from a common origin, so this Kabsch/Wahba fit
    deliberately does not subtract a point-cloud centroid. SVD finds the best
    orthogonal transform. The final sign correction enforces determinant +1,
    preventing an impossible mirror reflection from being called a rotation.
    At least three distinct, noncollinear directions are required on each side;
    repeated/collinear rays cannot adequately constrain a calibration fit.
    """
    source = np.asarray(
        [_direction(value, "predicted calibration direction") for value in predicted]
    )
    target = np.asarray(
        [_direction(value, "reference calibration direction") for value in reference]
    )
    if source.shape != target.shape or source.ndim != 2 or source.shape[0] < 3:
        raise ValueError("Rotation fitting needs at least three paired calibration directions")
    for name, matrix in (("predicted", source), ("reference", target)):
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        centered_values = np.linalg.svd(matrix - matrix.mean(axis=0), compute_uv=False)
        if (singular_values[1] <= singular_values[0] * 1e-6
                or centered_values[1] <= max(centered_values[0], 1e-12) * 1e-6):
            raise ValueError(f"{name} calibration directions are collinear or degenerate")
    # Rows store vectors, hence H = source.T @ target and R = V @ U.T.
    u, _, vt = np.linalg.svd(source.T @ target)
    parity = np.eye(3)
    parity[2, 2] = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    return vt.T @ parity @ u.T


def _errors_degrees(predicted: list[np.ndarray], reference: list[np.ndarray]) -> list[float]:
    # atan2(||a x b||, a dot b) is stable at both zero and 180 degrees.
    return [
        math.degrees(math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b))))
        for a, b in zip(predicted, reference)
    ]


def _statistics(errors: list[float]) -> dict[str, int | float | None]:
    if not errors:
        return {"count": 0, "mean_deg": None, "median_deg": None, "p95_deg": None, "max_deg": None}
    values = np.asarray(errors)
    return {
        "count": len(errors),
        "mean_deg": float(np.mean(values)),
        "median_deg": float(np.median(values)),
        "p95_deg": float(np.percentile(values, 95)),
        "max_deg": float(np.max(values)),
    }


def _coverage(reasons: Counter[str]) -> dict[str, Any]:
    total = sum(reasons.values())
    usable = reasons.get("usable", 0)
    return {
        "target_samples": total,
        "usable_samples": usable,
        "usable_fraction": usable / total if total else None,
        "excluded_samples": total - usable,
        "counts": dict(sorted(reasons.items())),
    }


def evaluate_gaze(
    records: Iterable[Mapping[str, Any]],
    targets: Mapping[str, Any],
    *,
    fit_rotation: bool = False,
) -> dict[str, Any]:
    """Validate identities/alignment and evaluate only held-out target frames.

    A geometry estimate is usable here when it was accepted, is ready, and has
    a finite nonzero direction. Diagnostic range failures do not silently
    remove difficult predictions from the accuracy statistics. Their optional
    pipeline diagnostics remain in the source export for separate inspection.
    Malformed input, unknown camera identities, missing target frames, and
    training/evaluation leakage raise ValueError instead of yielding a score.
    """
    if not isinstance(targets, Mapping):
        raise ValueError("targets must be a JSON object")
    if type(targets.get("schema_version")) is not int or targets.get("schema_version") != 1:
        raise ValueError("targets.schema_version must be 1")
    kind = targets.get("data_kind")
    if not isinstance(kind, str) or kind not in {"real_reference", "synthetic"}:
        raise ValueError("targets.data_kind must be 'real_reference' or 'synthetic'")
    direction_source = _text(targets.get("direction_source"), "targets.direction_source")
    recording_id = _text(targets.get("recording_id"), "targets.recording_id")
    if targets.get("coordinate_system") != COORDINATE_SYSTEM:
        raise ValueError(f"targets.coordinate_system must be {COORDINATE_SYSTEM!r}")
    samples = targets.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("targets.samples must be a nonempty list; no accuracy can be evaluated without targets")

    references: dict[tuple[str, int], dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ValueError(f"target sample {index} must be an object")
        eye = sample.get("eye")
        if not isinstance(eye, str) or eye not in _EYES:
            raise ValueError(f"target sample {index} has an invalid eye")
        frame_index = _frame(sample.get("frame_index"), f"target {index}.frame_index")
        key = (eye, frame_index)
        if key in references:
            raise ValueError(f"Duplicate target frame {key}; calibration/validation samples must be disjoint")
        split = sample.get("split")
        if not isinstance(split, str) or split not in {"calibration", "validation"}:
            raise ValueError(f"target {key}.split must be 'calibration' or 'validation'")
        references[key] = {
            "timestamp_s": _timestamp(sample.get("timestamp_s"), f"target {key}.timestamp_s"),
            "direction": _direction(sample.get("direction_camera"), f"target {key}.direction_camera"),
            "split": split,
        }
    eyes = sorted({key[0] for key in references})
    identities: dict[str, dict[str, str]] = {}
    for eye in eyes:
        identities[eye] = {}
        for header_field, record_field in _IDENTITY_MAPS.items():
            values = targets.get(header_field)
            if not isinstance(values, Mapping):
                raise ValueError(f"targets.{header_field} must be a per-eye object")
            value = _text(values.get(eye), f"targets.{header_field}.{eye}")
            if record_field == "calibration_sha256":
                _digest(value, f"targets.{header_field}.{eye}")
            elif record_field in {"frame_rotation", "calibration_rotation"}:
                _rotation(value, f"targets.{header_field}.{eye}")
            identities[eye][record_field] = value
        per_eye = sorted((frame, ref) for (sample_eye, frame), ref in references.items() if sample_eye == eye)
        if any(b[1]["timestamp_s"] <= a[1]["timestamp_s"] for a, b in zip(per_eye, per_eye[1:])):
            raise ValueError(f"{eye} target timestamps must increase with frame indices")
        calibration_frames = [frame for frame, ref in per_eye if ref["split"] == "calibration"]
        validation_frames = [frame for frame, ref in per_eye if ref["split"] == "validation"]
        if not validation_frames:
            raise ValueError(f"{eye} needs at least one held-out validation target")
        # Enforce the declared temporal split even when fitting is disabled so
        # this document cannot silently change meaning in a subsequent run.
        if calibration_frames and max(calibration_frames) >= min(validation_frames):
            raise ValueError(f"{eye} calibration targets must occur before all validation targets")

    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"record {index} must be an object")
        if type(record.get("schema_version")) is not int or record.get("schema_version") != 1:
            raise ValueError(f"record {index}.schema_version must be 1")
        if record.get("recording_id") != recording_id:
            raise ValueError(f"record {index} recording_id does not match the target recording")
        if record.get("coordinate_system") != COORDINATE_SYSTEM:
            raise ValueError(f"record {index} coordinate_system does not match the targets")
        eye = record.get("eye")
        if not isinstance(eye, str) or eye not in _EYES:
            raise ValueError(f"record {index} has an invalid eye")
        frame_index = _frame(record.get("frame_index"), f"record {index}.frame_index")
        key = (eye, frame_index)
        if key in indexed:
            raise ValueError(f"Duplicate exported record for {key}")
        timestamp = _timestamp(record.get("timestamp_s"), f"record {key}.timestamp_s")
        for field in ("frame_rotation", "calibration_rotation"):
            _rotation(record.get(field), f"record {key}.{field}")
        if eye in identities:
            for field, expected in identities[eye].items():
                if record.get(field) != expected:
                    raise ValueError(f"record {key} {field} does not match target metadata")
        model_input = record.get("model_input")
        if not isinstance(model_input, str) or model_input not in {"accepted", "skipped"}:
            raise ValueError(f"record {key}.model_input must be 'accepted' or 'skipped'")
        ready = record.get("ready")
        if not isinstance(ready, bool):
            raise ValueError(f"record {key}.ready must be a boolean")
        raw_direction = record.get("gaze_direction_camera")
        direction = None if raw_direction is None else _direction(raw_direction, f"record {key}.gaze_direction_camera")
        if model_input == "skipped":
            if ready or direction is not None:
                raise ValueError(f"skipped record {key} must not contain ready/stale gaze geometry")
            reason = "skipped"
        elif not ready:
            reason = "not_ready"
        elif direction is None:
            reason = "missing_gaze"
        else:
            reason = "usable"
        indexed[key] = {"timestamp_s": timestamp, "direction": direction, "reason": reason}
    if not indexed:
        raise ValueError("records must contain at least one exported frame")
    for eye in _EYES:
        per_eye = sorted((frame, rec) for (sample_eye, frame), rec in indexed.items() if sample_eye == eye)
        if any(b[1]["timestamp_s"] <= a[1]["timestamp_s"] for a, b in zip(per_eye, per_eye[1:])):
            raise ValueError(f"{eye} exported timestamps must increase with frame indices")
    for key, reference in references.items():
        if key not in indexed:
            raise ValueError(f"Target frame {key} is absent from the export; adjacent frames cannot substitute")
        if abs(indexed[key]["timestamp_s"] - reference["timestamp_s"]) > TIMESTAMP_TOLERANCE_S:
            raise ValueError(f"Target frame {key} timestamp does not match the export")

    per_eye_reports: dict[str, Any] = {}
    all_raw_errors: list[float] = []
    all_corrected_errors: list[float] = []
    all_validation_reasons: Counter[str] = Counter()
    for eye in eyes:
        groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {"calibration": [], "validation": []}
        for key, reference in sorted(references.items()):
            if key[0] == eye:
                groups[reference["split"]].append((indexed[key], reference))
        coverage = {split: _coverage(Counter(rec["reason"] for rec, _ in pairs)) for split, pairs in groups.items()}
        usable_calibration = [(rec, ref) for rec, ref in groups["calibration"] if rec["reason"] == "usable"]
        usable_validation = [(rec, ref) for rec, ref in groups["validation"] if rec["reason"] == "usable"]
        rotation = None
        if fit_rotation:
            rotation = fit_direction_rotation(
                [rec["direction"] for rec, _ in usable_calibration],
                [ref["direction"] for _, ref in usable_calibration],
            )
        predicted = [rec["direction"] for rec, _ in usable_validation]
        reference = [ref["direction"] for _, ref in usable_validation]
        raw_errors = _errors_degrees(predicted, reference)
        corrected_errors = [] if rotation is None else _errors_degrees([rotation @ vector for vector in predicted], reference)
        all_raw_errors.extend(raw_errors)
        all_corrected_errors.extend(corrected_errors)
        all_validation_reasons.update(rec["reason"] for rec, _ in groups["validation"])
        per_eye_reports[eye] = {
            "identity": identities[eye],
            "coverage": coverage,
            "raw_angular_error": _statistics(raw_errors),
            "calibrated_angular_error": None if rotation is None else _statistics(corrected_errors),
            "rotation_predicted_to_reference": None if rotation is None else rotation.tolist(),
            "rotation_fit_sample_count": 0 if rotation is None else len(usable_calibration),
        }
    return {
        "schema_version": 1,
        "recording_id": recording_id,
        "coordinate_system": COORDINATE_SYSTEM,
        "data_kind": kind,
        "direction_source": direction_source,
        "real_reference_supplied": kind == "real_reference",
        "accuracy_verdict": None,
        "interpretation": (
            "Synthetic reference results validate this software only; real gaze accuracy remains unmeasured."
            if kind == "synthetic" else
            "Angular agreement with the supplied real references; their independent measurement is declared by the supplier. No accuracy pass/fail threshold is imposed."
        ),
        "metric_population": "Held-out validation samples with accepted, ready, finite nonzero gaze. Model range-check failures remain included.",
        "coverage_population": "Every declared validation target, including skipped, not-ready, and missing-gaze records.",
        "timestamp_tolerance_s": TIMESTAMP_TOLERANCE_S,
        "rotation_fitted": bool(fit_rotation),
        "validation_coverage": _coverage(all_validation_reasons),
        "raw_angular_error": _statistics(all_raw_errors),
        "calibrated_angular_error": _statistics(all_corrected_errors) if fit_rotation else None,
        "eyes": per_eye_reports,
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read pipeline records with file/line context for malformed JSON."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each record must be an object")
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", required=True, type=Path, help="Pipeline frame JSONL export")
    parser.add_argument("--targets", required=True, type=Path, help="Independent target-direction JSON")
    parser.add_argument("--output", required=True, type=Path, help="Write the JSON evaluation report")
    parser.add_argument("--fit-rotation", action="store_true", help="Fit a per-eye rotation using only earlier calibration samples")
    args = parser.parse_args(argv)
    if args.output.resolve() in {args.records.resolve(), args.targets.resolve()}:
        parser.error("--output must differ from both input paths")
    try:
        with args.targets.open(encoding="utf-8") as handle:
            targets = json.load(handle)
        report = evaluate_gaze(read_jsonl(args.records), targets, fit_rotation=args.fit_rotation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Wrote {args.output}")
    print(report["interpretation"])
    print(f"Validation coverage: {report['validation_coverage']['usable_samples']}/{report['validation_coverage']['target_samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
