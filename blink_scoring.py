"""Pure scoring and report functions for human-reviewed blink validation.

Accept exported record/label dictionaries; return report dictionaries or text.
No server, videos or filesystem writes are required. Start with score_records
(single run), compare_records (paired runs), or reviewer_agreement.
"""

from collections import Counter
import math
import statistics


def _bv_key(item, description):
    """Use source frame numbers, never row order, to join labels and exports."""
    if not isinstance(item, dict):
        raise ValueError(f"{description} must be a JSON object.")
    eye, frame = item.get("eye"), item.get("frame_index")
    if eye not in ("left", "right"):
        raise ValueError(f"{description}: eye must be left or right.")
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"{description}: frame_index must be a nonnegative integer.")
    return eye, frame


def _bv_binding(item, description):
    """A human label applies only to the exact video, rotation and crop reviewed."""
    digest = item.get("video_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{description}: video_sha256 must be the video's SHA-256 digest.")
    rotation = item.get("frame_rotation")
    if rotation not in ("none", "clockwise", "counterclockwise", "180"):
        raise ValueError(f"{description}: unsupported frame_rotation {rotation!r}.")
    roi = item.get("roi")
    if (not isinstance(roi, (tuple, list)) or len(roi) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in roi)
            or min(roi[:2]) < 0 or min(roi[2:]) <= 0):
        raise ValueError(f"{description}: roi must be [x, y, width, height] with positive size.")
    return digest, rotation, tuple(roi)


def _bv_index_records(records):
    """Reject partial joins and time reversals before calculating any percentages."""
    indexed, previous = {}, {}
    for record in records:
        key = _bv_key(record, "Export record")
        if key in indexed:
            raise ValueError(f"Duplicate export record for {key[0]} frame {key[1]}.")
        _bv_binding(record, f"Export {key}")
        timestamp = record.get("timestamp_s")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp < 0):
            raise ValueError(f"Export {key}: timestamp_s must be finite and nonnegative.")
        if key[0] in previous:
            old_frame, old_time = previous[key[0]]
            if key[1] <= old_frame or timestamp <= old_time:
                raise ValueError(f"Export records for {key[0]} must have increasing frames and timestamps.")
        if record.get("model_input") not in ("accepted", "skipped"):
            raise ValueError(f"Export {key}: model_input must be accepted or skipped.")
        if not isinstance(record.get("ready"), bool):
            raise ValueError(f"Export {key}: ready must be a boolean.")
        previous[key[0]] = key[1], timestamp
        indexed[key] = record
    if not indexed:
        raise ValueError("The export contains no frames. Run the pipeline before scoring.")
    return indexed


def _bv_validated_labels(labels):
    result, seen = [], set()
    for label in labels:
        key = _bv_key(label, "Label")
        if key in seen:
            raise ValueError(f"Duplicate label for {key[0]} frame {key[1]}; score one reviewer at a time.")
        seen.add(key)
        _bv_binding(label, f"Label {key}")
        if label.get("label") not in ("usable", "unusable", "uncertain"):
            raise ValueError(f"Label {key}: choose usable, unusable or uncertain.")
        if label.get("split") not in ("development", "held_out"):
            raise ValueError(f"Label {key}: split must be development or held_out.")
        if not isinstance(label.get("reviewed"), bool):
            raise ValueError(f"Label {key}: reviewed must be a boolean.")
        if not isinstance(label.get("reason"), str):
            raise ValueError(f"Label {key}: reason must be text.")
        if not isinstance(label.get("reviewer"), str) or not label["reviewer"].strip():
            raise ValueError(f"Label {key}: supply a reviewer name.")
        result.append(label)
    return result


def _bv_rate(count, total):
    # None means no evidence; zero percent would incorrectly imply success.
    return {"count": count, "total": total,
            "percent": 100.0 * count / total if total else None}


def _bv_reason_group(record):
    reason = str((record.get("quality") or {}).get("reason", record.get("status", "unspecified"))).lower()
    if any(word in reason for word in ("recovery", "stable pupil")):
        return "recovery"
    if any(word in reason for word in ("eyelid", "closure", "coverage")):
        return "closure_or_coverage"
    if any(word in reason for word in ("geometry", "ellipse", "pye3d", "calibrat", "image bounds")):
        return "geometry"
    if "confidence" in reason:
        return "confidence"
    if "no pupil" in reason:
        return "missing_pupil"
    return "other"


def _bv_recovery(pairs):
    """Measure only observed, contiguous unusable-to-usable label transitions.

    Missing/uncertain/unreviewed labels break an interval. A zero delay after
    admitting every preceding bad frame is reported explicitly as a missed bad
    interval, so it cannot be mistaken for successful blink recovery.
    """
    events = []
    for eye in ("left", "right"):
        sequence = sorted((pair for pair in pairs if pair[0]["eye"] == eye),
                          key=lambda pair: pair[0]["frame_index"])
        for index in range(1, len(sequence)):
            before, start = sequence[index - 1], sequence[index]
            if (before[0]["label"] != "unusable" or start[0]["label"] != "usable"
                    or start[0]["frame_index"] != before[0]["frame_index"] + 1):
                continue
            bad_start = index - 1
            while (bad_start > 0 and sequence[bad_start - 1][0]["label"] == "unusable"
                   and sequence[bad_start - 1][0]["frame_index"] + 1
                   == sequence[bad_start][0]["frame_index"]):
                bad_start -= 1
            usable_end = index
            while (usable_end + 1 < len(sequence)
                   and sequence[usable_end + 1][0]["label"] == "usable"
                   and sequence[usable_end + 1][0]["frame_index"]
                   == sequence[usable_end][0]["frame_index"] + 1):
                usable_end += 1
            bad_pairs = sequence[bad_start:index]
            first_accepted = next((pair for pair in sequence[index:usable_end + 1]
                                   if pair[1]["model_input"] == "accepted"), None)
            rejected_bad = sum(pair[1]["model_input"] == "skipped" for pair in bad_pairs)
            delay = (first_accepted[1]["timestamp_s"] - start[1]["timestamp_s"]
                     if first_accepted else None)
            events.append({
                "eye": eye, "bad_start_frame": bad_pairs[0][0]["frame_index"],
                "first_usable_frame": start[0]["frame_index"],
                "last_observed_usable_frame": sequence[usable_end][0]["frame_index"],
                "first_accepted_frame": first_accepted[0]["frame_index"] if first_accepted else None,
                "delay_s": delay, "censored": first_accepted is None,
                "observed_usable_duration_s": sequence[usable_end][1]["timestamp_s"] - start[1]["timestamp_s"],
                "preceding_bad_frames": len(bad_pairs), "preceding_bad_frames_rejected": rejected_bad,
                "fully_missed_bad_interval": rejected_bad == 0,
            })
    # Delays from wholly missed intervals remain visible in events, but are not
    # included in the recovery average for intervals where rejection occurred.
    delays = [event["delay_s"] for event in events
              if not event["fully_missed_bad_interval"] and event["delay_s"] is not None]
    return {
        "events": events, "event_count": len(events),
        "detected_prior_bad_events": sum(not event["fully_missed_bad_interval"] for event in events),
        "missed_prior_bad_events": sum(event["fully_missed_bad_interval"] for event in events),
        "censored_count": sum(event["censored"] for event in events),
        "measured_detected_events": len(delays),
        "mean_delay_s": statistics.mean(delays) if delays else None,
        "max_delay_s": max(delays) if delays else None,
        "note": "Delay summaries include only observed recovery after at least one bad-frame rejection; censored and fully missed intervals are reported separately.",
    }


def _bv_score_pairs(pairs):
    good = [pair for pair in pairs if pair[0]["label"] == "usable"]
    bad = [pair for pair in pairs if pair[0]["label"] == "unusable"]
    exact_reasons, reason_groups = Counter(), Counter()
    for label, record in pairs:
        outcome = record["model_input"]
        reason = str((record.get("quality") or {}).get("reason", record.get("status", "unspecified")))
        exact_reasons[f"{label['label']} / {outcome} / {reason}"] += 1
        reason_groups[f"{label['label']} / {outcome} / {_bv_reason_group(record)}"] += 1
    return {
        "scored_frames": len(good) + len(bad),
        "bad_frames_admitted": _bv_rate(sum(record["model_input"] == "accepted" for _, record in bad), len(bad)),
        "good_frames_discarded": _bv_rate(sum(record["model_input"] == "skipped" for _, record in good), len(good)),
        "accepted_frames": sum(record["model_input"] == "accepted" for _, record in good + bad),
        "skipped_frames": sum(record["model_input"] == "skipped" for _, record in good + bad),
        "reason_groups": dict(sorted(reason_groups.items())),
        "exact_reasons": dict(sorted(exact_reasons.items())),
        "recovery": _bv_recovery(pairs),
    }


def score_records(records, labels, *, split="development"):
    """Score one reviewer's human-reviewed labels against a complete export.

    Imported suggestions (reviewed=False) and uncertain human judgments are
    counted as exclusions. No denominator means ``percent=None``, not 0% error.
    This measures the accept/reject task; it does not measure gaze accuracy.
    """
    if split not in ("development", "held_out"):
        raise ValueError("Choose split='development' or split='held_out'.")
    indexed = _bv_index_records(records)
    selected = [label for label in _bv_validated_labels(labels) if label["split"] == split]
    reviewers = {label["reviewer"] for label in selected}
    if len(reviewers) > 1:
        raise ValueError("Score one reviewer at a time; use reviewer_agreement for multiple reviewers.")
    pairs = []
    for label in selected:
        key = _bv_key(label, "Label")
        record = indexed.get(key)
        if record is None:
            raise ValueError(f"Missing export for {key[0]} frame {key[1]}; run the entire labeled segment.")
        if _bv_binding(label, "Label") != _bv_binding(record, "Export"):
            raise ValueError(f"Video/rotation/ROI mismatch at {key}; re-review labels for this configuration.")
        if label["reviewed"] and label["label"] != "uncertain":
            pairs.append((label, record))
    aggregate = _bv_score_pairs(pairs)
    return {
        "schema_version": 1, "split": split,
        "reviewer": next(iter(reviewers)) if reviewers else None,
        "exported_frames": len(indexed), "selected_labels": len(selected),
        "excluded": {
            "without_label": len(indexed) - len(selected),
            "unreviewed": sum(not label["reviewed"] for label in selected),
            "uncertain_reviewed": sum(label["reviewed"] and label["label"] == "uncertain" for label in selected),
        },
        "aggregate": aggregate,
        # Only already-reviewed, certain labels enter this list. Keep the source
        # frame identity so the UI can jump to the exact image, never row order.
        "mistakes": [dict(
            eye=label["eye"], frame_index=label["frame_index"],
            timestamp_s=record["timestamp_s"], label=label["label"],
            kind=("bad_frame_admitted" if label["label"] == "unusable"
                  else "good_frame_discarded"),
            reviewer_note=label["reason"],
            boundary_evidence=record.get("boundary_evidence"),
            # Old runs have no raw observation. Do not reconstruct it from
            # corrected/projected geometry or pretend a later run supplied it.
            pupil_observation=record.get("pupil_observation"),
            pupil_observation_coordinate_system=record.get("pupil_observation_coordinate_system"),
            filter_reason=str((record.get("quality") or {}).get("reason", "unspecified")),
            pupil_confidence=(record.get("pupil_confidence")
                if isinstance(record.get("pupil_confidence"), (int, float))
                and math.isfinite(record["pupil_confidence"]) else None),
        ) for label, record in sorted(pairs, key=lambda pair: (pair[0]["eye"], pair[0]["frame_index"]))
          if ((label["label"] == "unusable" and record["model_input"] == "accepted")
              or (label["label"] == "usable" and record["model_input"] == "skipped"))],
        "by_eye": {eye: _bv_score_pairs([pair for pair in pairs if pair[0]["eye"] == eye])
                   for eye in ("left", "right")},
        "evidence_status": "human_reviewed" if pairs else "no_scorable_human_labels",
        "limitations": [
            "Frame accept/reject performance is separate from gaze accuracy.",
            "Development labels may guide tuning; held-out labels must not be used to choose thresholds.",
            "Recovery requires consecutive reviewed source frames; label gaps are not interpolated.",
        ],
    }


# These fields determine what was processed and how. Missing optional fields
# must match too, so a new exporter cannot silently compare against an old run.
_BV_PAIRED_METADATA = (
    "schema_version", "recording_id", "camera_id", "coordinate_system",
    "video_sha256", "calibration_sha256", "frame_rotation", "calibration_rotation",
    "camera_matrix", "calibration_identity_status", "calibration_source",
    "direction_kind", "fps", "model_sha256", "roi", "eye_radius_mm",
    "focal_normalization", "min_confidence", "mask_threshold",
    "recovery_confidence", "recovery_duration_s", "source_fingerprint",
    "pipeline_source_sha256", "source_sha256", "device", "max_frames",
)

_BV_ESTIMATE_FIELDS = {
    "frame_pair_timing_ms", "robot_command", "ready", "eye_center_mm", "pupil_center_mm", "pupil_diameter_mm",
    "pupil_confidence", "model_confidence", "update_time_ms", "projected_eye_sphere",
    "projected_eye_center", "projected_pupil_center", "status", "model_diagnostics",
    "gaze_direction_camera", "model_stability", "model_input", "quality", "filter_mode",
}


def _bv_canonical(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _bv_canonical(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_bv_canonical(item) for item in value)
    return value


def _bv_center(record):
    """Native pye3d center avoids conflating this diagnostic with radius scaling."""
    center = (record.get("model_diagnostics") or {}).get("native_eye_center_mm")
    if not isinstance(center, (tuple, list)) or len(center) != 3:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in center):
        return None
    return center


def _bv_distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    percentile = ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "p95": percentile, "max": max(values)}


def _bv_model_comparison(baseline, filtered, eye=None):
    keys = sorted(key for key in baseline if eye is None or key[0] == eye)
    common = {key for key in keys if baseline[key]["ready"] and filtered[key]["ready"]}
    baseline_steps, filtered_steps = [], []
    for key in sorted(common):
        previous = key[0], key[1] - 1
        if previous not in common:
            continue
        centers = [_bv_center(source[index]) for source in (baseline, filtered)
                   for index in (previous, key)]
        if any(center is None for center in centers):
            continue
        baseline_steps.append(math.dist(centers[0], centers[1]))
        filtered_steps.append(math.dist(centers[2], centers[3]))
    return {
        "frames": len(keys),
        "baseline_accepted": sum(baseline[key]["model_input"] == "accepted" for key in keys),
        "filtered_accepted": sum(filtered[key]["model_input"] == "accepted" for key in keys),
        "baseline_ready": sum(baseline[key]["ready"] for key in keys),
        "filtered_ready": sum(filtered[key]["ready"] for key in keys),
        "common_ready": len(common),
        "adjacent_common_ready_center_steps": len(baseline_steps),
        "baseline_center_step_native_mm": _bv_distribution(baseline_steps),
        "filtered_center_step_native_mm": _bv_distribution(filtered_steps),
    }


def compare_records(baseline_records, filtered_records, labels, *, split="development", run_summaries=None):
    """Pair identical inputs and compare labels, coverage and center-step size.

    Center motion is evaluated on the same adjacent ready frames in both runs.
    A skip breaks adjacency: no interpolation hides missing output. These are
    descriptive stability measurements, not proof of geometric accuracy.
    """
    baseline_records, filtered_records, labels = list(baseline_records), list(filtered_records), list(labels)
    baseline, filtered = _bv_index_records(baseline_records), _bv_index_records(filtered_records)
    if baseline.keys() != filtered.keys():
        raise ValueError("Comparison requires identical eye/frame sets; rerun both modes for the same complete recording.")
    for key in baseline:
        left, right = baseline[key], filtered[key]
        if left.get("filter_mode") != "baseline" or right.get("filter_mode") != "filtered":
            raise ValueError("Supply a baseline export first and a filtered export second.")
        for name in ("recording_id", "calibration_sha256", "model_sha256"):
            if not left.get(name) or not right.get(name):
                raise ValueError(f"Comparison requires {name}; create fresh exports with the current pipeline.")
        # Treat unfamiliar exported fields as input metadata until explicitly
        # identified as model outputs. New configuration cannot silently differ.
        metadata_names = (set(_BV_PAIRED_METADATA) | set(left) | set(right)) - _BV_ESTIMATE_FIELDS
        for name in sorted(metadata_names):
            if _bv_canonical(left.get(name)) != _bv_canonical(right.get(name)):
                raise ValueError(f"Comparison input mismatch for {name} at {key}; keep inputs and settings identical.")
    report = {
        "schema_version": 1, "split": split,
        "baseline": score_records(baseline_records, labels, split=split),
        "filtered": score_records(filtered_records, labels, split=split),
        "model_output": _bv_model_comparison(baseline, filtered),
        "model_output_by_eye": {eye: _bv_model_comparison(baseline, filtered, eye)
                                for eye in ("left", "right")},
        "runtime": {"available": False, "baseline_elapsed_seconds": None,
                    "filtered_elapsed_seconds": None, "note": "A single paired run is descriptive, not a runtime benchmark."},
        "limitations": [
            "Baseline retains confidence and geometry protections; filtering additionally applies eyelid checks and temporal recovery.",
            "Smaller model-center steps do not establish greater gaze accuracy; inspect output coverage as well.",
            "Center steps use exact adjacent source frames ready in both runs and native pye3d millimeters.",
            "The current recordings were used during development; use fresh recordings for independent final evaluation.",
        ],
    }
    if run_summaries is not None:
        if not isinstance(run_summaries, dict):
            raise ValueError("run_summaries must contain baseline and filtered summary objects.")
        durations = {}
        for mode in ("baseline", "filtered"):
            summary = run_summaries.get(mode)
            if not isinstance(summary, dict):
                raise ValueError(f"Missing {mode} run summary.")
            elapsed = summary.get("elapsed_seconds", summary.get("wall_time_seconds"))
            if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                    or not math.isfinite(elapsed) or elapsed <= 0):
                raise ValueError(f"{mode} run summary needs a finite positive elapsed_seconds.")
            durations[mode] = float(elapsed)
        report["runtime"].update(available=True, baseline_elapsed_seconds=durations["baseline"],
                                 filtered_elapsed_seconds=durations["filtered"], summaries=run_summaries)
    return report


def reviewer_agreement(labels_all):
    """Compare independent human reviews without choosing a 'correct' reviewer.

    Each pair is compared only where both people reviewed the same frame with
    identical video, orientation, crop and split. Uncertain counts as a distinct
    judgment; report it rather than silently treating it as usable.
    """
    by_reviewer = {}
    for label in labels_all:
        _bv_key(label, "Label")
        reviewer = label.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("Every label needs a reviewer name.")
        by_reviewer.setdefault(reviewer, []).append(label)
    indexed = {}
    for reviewer, labels in by_reviewer.items():
        indexed[reviewer] = {_bv_key(label, "Label"): label
                             for label in _bv_validated_labels(labels) if label["reviewed"]}
    pairs = []
    names = sorted(indexed)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1:]:
            common = sorted(indexed[first].keys() & indexed[second].keys())
            disagreements, agreed, uncertain = [], 0, 0
            for key in common:
                a, b = indexed[first][key], indexed[second][key]
                if (_bv_binding(a, "First review") != _bv_binding(b, "Second review")
                        or a["split"] != b["split"]):
                    raise ValueError(f"Reviewer bindings/splits differ at {key}; compare the same review task.")
                uncertain += a["label"] == "uncertain" or b["label"] == "uncertain"
                if a["label"] == b["label"]:
                    agreed += 1
                else:
                    disagreements.append({"eye": key[0], "frame_index": key[1],
                                          "first_label": a["label"], "second_label": b["label"],
                                          "first_reason": a["reason"], "second_reason": b["reason"]})
            pairs.append({"reviewers": [first, second], "common_reviewed_frames": len(common),
                          "agreement": _bv_rate(agreed, len(common)),
                          "frames_with_uncertain_judgment": uncertain, "disagreements": disagreements})
    return {"reviewers": names, "pairs": pairs,
            "note": "Agreement measures consistency between reviewers, not ground-truth accuracy. Discuss disagreements before creating a consensus review."}


def format_comparison_report(report):
    """Make a compact readable report while retaining the JSON as full evidence."""
    def percentage(item):
        return (f"{item['count']}/{item['total']} ({item['percent']:.1f}%)"
                if item["percent"] is not None else "No reviewed examples")

    lines = ["# Blink-filter comparison", "", f"Label split: **{report['split']}**.", "",
             "| Measure | Baseline | Blink filtering |", "|---|---:|---:|"]
    for title, name in (("Bad frames admitted", "bad_frames_admitted"),
                        ("Good frames discarded", "good_frames_discarded")):
        lines.append(f"| {title} | {percentage(report['baseline']['aggregate'][name])} | {percentage(report['filtered']['aggregate'][name])} |")
    model = report["model_output"]
    lines.extend([f"| Model inputs accepted | {model['baseline_accepted']} | {model['filtered_accepted']} |",
                  f"| Geometry available | {model['baseline_ready']} | {model['filtered_ready']} |", "",
                  f"Compared native model-center steps on {model['adjacent_common_ready_center_steps']} identical adjacent frame pairs.", ""])
    for mode in ("baseline", "filtered"):
        score = report[mode]
        recovery = score["aggregate"]["recovery"]
        delay = (f"{recovery['mean_delay_s']:.3f} seconds" if recovery["mean_delay_s"] is not None else "unavailable")
        lines.append(f"- **{mode.capitalize()}:** {score['aggregate']['scored_frames']} reviewed usable/unusable frames; "
                     f"{score['excluded']['without_label']} without labels, {score['excluded']['unreviewed']} unreviewed suggestions "
                     f"and {score['excluded']['uncertain_reviewed']} uncertain judgments excluded. "
                     f"Mean observed recovery delay: {delay}; {recovery['missed_prior_bad_events']} fully missed bad intervals; "
                     f"{recovery['censored_count']} recovery intervals without observed acceptance.")
    if report["runtime"]["available"]:
        runtime = report["runtime"]
        lines.extend(["", f"Full-run elapsed time: baseline {runtime['baseline_elapsed_seconds']:.2f}s; filtering {runtime['filtered_elapsed_seconds']:.2f}s.",
                      runtime["note"]])
    lines.extend(["", "These results measure frame filtering and describe model stability. They do not establish gaze accuracy.", ""])
    return "\n".join(lines)
