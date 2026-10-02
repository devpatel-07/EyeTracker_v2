"""Detect candidate corneal reflections; this is NOT a gaze estimator.

Run ``python glint_detection.py --help`` beside the existing project modules.
Only NumPy/OpenCV are needed to import the detector. The offline CLI lazily
loads the project's pupil detector and blink gate. No robot/model is updated.

Coordinates: distorted pixels in the ROTATED FULL FRAME, just like the pupil
detector. Candidate numbers are local to one frame, never physical LED IDs.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import math
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class GlintConfig:
    """Engineering starting values for 8-bit images, NOT calibrated confidence.

    A bright seed must reach seed_threshold. Its connected support includes
    pixels down to support_threshold, preserving more of the reflection's edge.
    Pixel-area limits depend on image resolution; validate them on new cameras.
    """

    seed_threshold: int = 235
    support_threshold: int = 200
    min_area: int = 3
    max_area: int = 900
    min_contrast: float = 25.0
    max_aspect: float = 2.5
    min_fill: float = 0.4
    ring_px: int = 5
    pupil_radius_multiple: float = 2.5
    max_components: int = 2000

    def __post_init__(self):
        for name in ("seed_threshold", "support_threshold", "min_area", "max_area",
                     "ring_px", "max_components"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.support_threshold < self.seed_threshold <= 255:
            raise ValueError("require 0 <= support threshold < seed threshold <= 255")
        if not 1 <= self.min_area <= self.max_area or self.ring_px < 1 or self.max_components < 1:
            raise ValueError("invalid component size/ring limits")
        if not all(math.isfinite(v) for v in (self.min_contrast, self.max_aspect,
                                             self.min_fill, self.pupil_radius_multiple)):
            raise ValueError("settings must be finite")
        if not (0 <= self.min_contrast <= 255 and self.max_aspect >= 1
                and 0 < self.min_fill <= 1 and self.pupil_radius_multiple > 0):
            raise ValueError("invalid contrast, shape or pupil-distance setting")


class GlintDetector:
    """Stateless candidate detection: missing frames never reuse old centers."""

    def __init__(self, config=None):
        self.config = config or GlintConfig()

    def detect(self, frame, roi, *, quality, pupil_ellipse):
        """Return JSON-compatible diagnostics and candidate centers.

        frame: uint8 gray or BGR image, already rotated like pupil detection.
        roi: (x,y,width,height), the broad eye crop in that full image.
        quality: existing FrameQualityDecision from TemporalQualityTracker.
        pupil_ellipse: ((cx,cy),(diameter1,diameter2),angle_degrees), FULL pixels.

        The quality gate is checked BEFORE image processing. Passing it only
        permits candidate extraction; it does not validate LED correspondence.
        """
        started = time.perf_counter()
        result = dict(status="skipped", reason=quality.reason,
                      coordinate_system="rotated_raw_frame_pixels",
                      candidates=[], rejected=[], component_count=0,
                      led_correspondence="unassigned", gaze_available=False,
                      robot_allowed=False)

        def finish(status, reason):
            result.update(status=status, reason=reason,
                          detection_ms=(time.perf_counter()-started)*1000)
            return result

        if not quality.allow_model_update or quality.state != "usable":
            return finish("skipped", quality.reason)
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("frame must be a uint8 array")
        if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] != 3):
            raise ValueError("frame must be gray or BGR")
        if len(roi) != 4 or any(type(v) is not int for v in roi):
            raise ValueError("ROI must contain four integers")
        x0, y0, width, height = roi
        if min(x0, y0) < 0 or min(width, height) <= 0 or x0+width > frame.shape[1] or y0+height > frame.shape[0]:
            raise ValueError("ROI must fit entirely within frame")
        if pupil_ellipse is None:
            return finish("unavailable", "missing pupil ellipse")
        (px, py), axes, angle = pupil_ellipse
        if not np.all(np.isfinite([px, py, *axes, angle])) or min(axes) <= 0:
            raise ValueError("pupil ellipse must be finite with positive diameters")
        if not (x0 <= px < x0+width and y0 <= py < y0+height):
            return finish("unavailable", "pupil center outside broad ROI")

        cfg = self.config
        crop = frame[y0:y0+height, x0:x0+width]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        # Label once in native OpenCV. There is no Python loop over image pixels
        # and no morphology on the pupil mask (which could alter pupil fitting).
        support = (gray >= cfg.support_threshold).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
        result["component_count"] = count-1
        if count-1 > cfg.max_components:
            return finish("unavailable", "too many bright components; check exposure/noise")

        radius = cfg.pupil_radius_multiple*max(axes)/2
        for label in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[label])
            # Reject large areas before building local arrays. Broad reflections
            # and overexposed skin must not dominate the candidate centroid.
            reasons = []
            if not cfg.min_area <= area <= cfg.max_area:
                reasons.append("area_out_of_range")
            if x == 0 or y == 0 or x+w == width or y+h == height:
                reasons.append("touches_crop_border")
            if max(w/h, h/w) > cfg.max_aspect:
                reasons.append("elongated_or_merged")
            if area/(w*h) < cfg.min_fill:
                reasons.append("low_fill")
            box = [x+x0, y+y0, w, h]
            if reasons:
                result["rejected"].append(dict(bbox=box, area_px=area, reasons=reasons))
                continue

            local_mask = labels[y:y+h, x:x+w] == label
            values = gray[y:y+h, x:x+w][local_mask]
            if int(values.max()) < cfg.seed_threshold:
                result["rejected"].append(dict(bbox=box, area_px=area, reasons=["no_bright_seed"]))
                continue
            # A local background distinguishes a bright spot from a uniformly
            # bright patch. Exclude every thresholded component from the ring.
            r = cfg.ring_px
            xa, ya, xb, yb = max(0, x-r), max(0, y-r), min(width, x+w+r), min(height, y+h+r)
            patch = gray[ya:yb, xa:xb]
            background = patch[labels[ya:yb, xa:xb] == 0]
            if background.size < 8:
                result["rejected"].append(dict(bbox=box, area_px=area, reasons=["insufficient_background"]))
                continue
            level = float(np.median(background))
            contrast = float(values.mean())-level
            # Intensity above background supplies subpixel weights; saturation
            # gives equal weights over the plateau, so precision is not implied.
            yy, xx = np.nonzero(local_mask)
            weights = np.maximum(values.astype(np.float64)-level, 0)
            if weights.sum() <= 0:
                continue
            cx = float(np.dot(xx, weights)/weights.sum()+x+x0)
            cy = float(np.dot(yy, weights)/weights.sum()+y+y0)
            if contrast < cfg.min_contrast:
                reasons.append("low_local_contrast")
            if math.hypot(cx-px, cy-py) > radius:
                reasons.append("outside_pupil_neighborhood")
            if reasons:
                result["rejected"].append(dict(bbox=box, area_px=area, reasons=reasons))
                continue
            saturation = float(np.mean(values >= 254))
            result["candidates"].append(dict(center_px=[cx, cy], bbox=box,
                area_px=area, contrast=contrast, peak=int(values.max()),
                saturation_fraction=saturation, led_id=None,
                warnings=(["saturated_centroid_may_be_biased"] if saturation > .25 else [])))

        # Stable display order ONLY: left-to-right order is not LED identity.
        result["candidates"].sort(key=lambda item: (item["center_px"][0], item["center_px"][1]))
        for index, item in enumerate(result["candidates"], 1):
            item["candidate_id"] = index
        return finish("candidates" if result["candidates"] else "no_candidates",
                      "unverified bright-spot candidates" if result["candidates"] else "no spots passed candidate checks")


def draw_glints(frame, roi, result, pupil_ellipse=None):
    """Return a broad-crop BGR preview. Amber markers are NOT verified LEDs."""
    x0, y0, w, h = roi
    panel = frame[y0:y0+h, x0:x0+w].copy()
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
    if pupil_ellipse is not None:
        (x, y), axes, angle = pupil_ellipse
        cv2.ellipse(panel, ((x-x0, y-y0), axes, angle), (255, 180, 0), 1)
    for item in result["candidates"]:
        cx, cy = item["center_px"]
        point = (round(cx-x0), round(cy-y0))
        cv2.circle(panel, point, 9, (0, 190, 255), 2)
        cv2.putText(panel, f'C{item["candidate_id"]}', (point[0]+10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 190, 255), 1)
    cv2.rectangle(panel, (0, 0), (w, 47), (20, 20, 20), -1)
    cv2.putText(panel, f'{result["status"]}: {len(result["candidates"])} candidates (LED IDs unknown)',
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    cv2.putText(panel, result["reason"][:105], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
    return panel


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--eye", choices=("left", "right"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwrites an old run")
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the whole video")
    parser.add_argument("--preview-every", type=int, default=100)
    parser.add_argument("--write-video", action="store_true", help="Also export a cropped annotated MP4")
    parser.add_argument("--seed-threshold", type=int, default=235)
    parser.add_argument("--support-threshold", type=int, default=200)
    args = parser.parse_args(argv)
    cfg = GlintConfig(seed_threshold=args.seed_threshold, support_threshold=args.support_threshold)
    if args.max_frames < 0 or args.preview_every < 1:
        parser.error("frame limit must be nonnegative and preview interval positive")
    if not args.video.is_file():
        parser.error("video does not exist")

    # Reuse the current project's presets and exact blink implementation rather
    # than maintain a second set of pupil weights, thresholds or recovery rules.
    import eye_pipeline as pipeline
    from pupil_detection import load_pupil_detector
    from pupil_quality import TemporalQualityTracker
    from video_preparation import rotate_frame

    roi = pipeline.LEFT_ROI if args.eye == "left" else pipeline.RIGHT_ROI
    rotation = pipeline.LEFT_FRAME_ROTATION if args.eye == "left" else pipeline.RIGHT_FRAME_ROTATION
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    summary = dict(status="running", eye=args.eye, frames=0, eligible_frames=0,
                   frames_with_candidates=0, candidates=0, skipped_frames=0,
                   geometry_available=False, robot_commands=0)
    detection_times = []
    try:
        if not capture.isOpened():
            raise ValueError("could not open video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("video must have valid FPS")
        source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        processed_size = source_size[::-1] if rotation in ("clockwise", "counterclockwise") else source_size
        args.output_dir.mkdir(parents=True, exist_ok=False)
        source_dir = Path(pipeline.__file__).resolve().parent
        source_names = ("pupil_detection.py", "pupil_quality.py", "eye_pipeline.py", "video_preparation.py")
        metadata = dict(video=str(args.video.resolve()), video_sha256=file_sha256(args.video),
                        eye=args.eye, fps=fps, reported_frame_count=source_count,
                        source_size=source_size, processed_size=processed_size,
                        roi=list(roi), rotation=rotation, config=asdict(cfg),
                        minimum_confidence=pipeline.MIN_CONFIDENCE,
                        recovery_confidence=pipeline.RECOVERY_CONFIDENCE,
                        recovery_s=pipeline.RECOVERY_DURATION_S,
                        boundary_policy=pipeline.BOUNDARY_POLICY,
                        model_sha256=file_sha256(pipeline.MODEL_PATH),
                        source_sha256={name: file_sha256(source_dir/name) for name in source_names},
                        detector_sha256=file_sha256(__file__),
                        opencv_version=cv2.__version__, numpy_version=np.__version__,
                        scope="2D candidates only; no LED assignment, 3D gaze or robot output")
        (args.output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
        pupil_detector = load_pupil_detector(pipeline.MODEL_PATH, pipeline.DEVICE,
                                             pipeline.MASK_THRESHOLD, pipeline.BOUNDARY_POLICY)
        tracker = TemporalQualityTracker(pipeline.MIN_CONFIDENCE, pipeline.RECOVERY_CONFIDENCE,
                                         pipeline.RECOVERY_DURATION_S, max_gap_s=1.5/fps)
        detector = GlintDetector(cfg)
        started = time.perf_counter()
        with (args.output_dir/"detections.jsonl").open("x") as stream:
            while args.max_frames == 0 or summary["frames"] < args.max_frames:
                ok, frame = capture.read()
                if not ok:
                    if source_count > 0 and summary["frames"] < source_count:
                        raise RuntimeError("video decode ended before reported frame count")
                    break
                frame = rotate_frame(frame, rotation)
                index = summary["frames"]
                timestamp = index/fps
                pupil = pupil_detector.detect(frame, roi)
                quality = tracker.update(pupil, timestamp)
                result = detector.detect(frame, roi, quality=quality, pupil_ellipse=pupil.ellipse)
                record = dict(frame_index=index, timestamp_s=timestamp, eye=args.eye,
                              pupil_ellipse=pupil.ellipse, pupil_confidence=pupil.confidence,
                              quality=asdict(quality), glints=result)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                summary["frames"] += 1
                summary["eligible_frames"] += int(quality.allow_model_update)
                summary["skipped_frames"] += int(result["status"] == "skipped")
                summary["frames_with_candidates"] += int(bool(result["candidates"]))
                summary["candidates"] += len(result["candidates"])
                if quality.allow_model_update:
                    detection_times.append(result["detection_ms"])
                if index % args.preview_every == 0 or args.write_video:
                    panel = draw_glints(frame, roi, result, pupil.ellipse)
                    if index % args.preview_every == 0:
                        if not cv2.imwrite(str(args.output_dir/f"frame_{index:06d}.jpg"), panel):
                            raise RuntimeError("could not save preview")
                    if args.write_video:
                        if writer is None:
                            writer = cv2.VideoWriter(str(args.output_dir/"overlay.mp4"),
                                cv2.VideoWriter_fourcc(*"mp4v"), fps, (panel.shape[1], panel.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError("could not open video writer")
                        writer.write(panel)
                if index % 100 == 0:
                    print(f'{args.eye}: {index+1} frames, {summary["candidates"]} candidate measurements', flush=True)
        if not summary["frames"]:
            raise RuntimeError("video contains no decoded frames")
        summary.update(status="complete", processing_s=time.perf_counter()-started,
                       mean_candidate_detection_ms=float(np.mean(detection_times)) if detection_times else None,
                       p95_candidate_detection_ms=float(np.percentile(detection_times, 95)) if detection_times else None,
                       timing_scope="candidate extraction on eligible frames only; excludes pupil inference",
                       completion="requested frame limit" if args.max_frames and summary["frames"] == args.max_frames else "end of video")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        # Only write into a directory this invocation initialized. Failed runs
        # remain recognizable, rather than masquerading as completed evidence.
        if 'metadata' in locals():
            if summary["status"] == "running":
                summary["status"] = "incomplete"
            (args.output_dir/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
