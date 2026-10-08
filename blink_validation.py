"""Local, dependency-free browser workflow for reviewing and testing blink filtering.

Run with the project's Python: ``.venv/bin/python blink_validation.py``.
The server binds only to this computer. Frames are decoded with OpenCV; review
labels are saved atomically on disk, never just in browser storage. Each reviewer
has a separate file. Existing development labels are not treated as human truth.
The two comparison runs invoke the real pipeline in fresh Python processes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser

import cv2
import numpy as np


# Re-export the public functions for existing callers of blink_validation.
from blink_scoring import (
    score_records, compare_records, reviewer_agreement, format_comparison_report,
)

HERE = Path(__file__).resolve().parent
# Resolve beside this module, not the terminal's current working directory.
# Keep CSS and JavaScript in one HTML file to avoid a frontend build system.
HTML = (HERE / "blink_review.html").read_text(encoding="utf-8")
EYES = ("left", "right")
LABELS = ("usable", "unusable", "uncertain")
ROTATIONS = ("none", "clockwise", "180", "counterclockwise")
# These exact recordings were used during development. They must not be
# presented as unseen held-out data just because the user labels new frames.
KNOWN_DEVELOPMENT_HASHES = {
    "26d77fa90da195ec883488fc5b3de22e0a2961cd7888c8bcb6f37a9333b93232",
    "306d9d68b5aabde13384f66f610dd437f20b88721174880ae99cc651b42c9b98",
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    """Replace one complete JSON document; a failed write preserves the last save."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reviewer_name(value):
    """Use a short stable reviewer identifier, safely usable as a filename."""
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", value) is None:
        raise ValueError("Use a reviewer name of 1–40 letters, numbers, hyphens or underscores.")
    return value.lower()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class ReviewWorkspace:
    """One recording pair and its independently saved reviewer files and runs."""

    def __init__(self, project, data_dir, left_video, right_video, *, split="development"):
        self.project = Path(project).resolve()
        self.lock = threading.RLock()
        self.frame_lock = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.job = None
        self.worker = None
        self.stopping = threading.Event()
        self.process = None
        self.frame_cache = {}
        if split not in {"development", "held_out"}:
            raise ValueError("split must be development or held_out")
        # Import the configured project rather than duplicate its rotation/ROI
        # defaults. The launcher starts a fresh process for a fresh configuration.
        sys.path.insert(0, str(self.project))
        import eye_pipeline as pipeline
        self.config = pipeline
        self.videos = {}
        for eye, path in zip(EYES, (left_video, right_video)):
            path = Path(path).expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"{eye} video does not exist: {path}")
            capture = cv2.VideoCapture(str(path))
            try:
                if not capture.isOpened():
                    raise ValueError(f"cannot open {eye} video")
                count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                ok, frame = capture.read()
                if not ok or count < 1 or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"{eye} video has no readable frames or invalid timing")
                rotation = getattr(pipeline, eye.upper() + "_FRAME_ROTATION")
                roi = list(getattr(pipeline, eye.upper() + "_ROI"))
                from video_preparation import rotate_frame
                processed = rotate_frame(frame, rotation)
                x, y, w, h = roi
                if min(x, y) < 0 or min(w, h) <= 0 or x+w > processed.shape[1] or y+h > processed.shape[0]:
                    raise ValueError(f"{eye} ROI does not fit; configure eye_pipeline.py before reviewing")
                self.videos[eye] = dict(path=str(path), video_sha256=file_sha256(path),
                    frame_count=count, fps=fps, frame_rotation=rotation, roi=roi)
            finally:
                capture.release()
        if abs(self.videos["left"]["fps"]-self.videos["right"]["fps"]) > 1e-3:
            raise ValueError("paired videos must have matching FPS")
        if split == "held_out" and any(v["video_sha256"] in KNOWN_DEVELOPMENT_HASHES for v in self.videos.values()):
            raise ValueError("These recordings were used in development; use fresh recordings for held-out evaluation.")
        self.frame_count = min(v["frame_count"] for v in self.videos.values())
        self.fps = self.videos["left"]["fps"]
        pair_hashes = {eye: self.videos[eye]["video_sha256"] for eye in EYES}
        self.recording_id = hashlib.sha256(json.dumps(pair_hashes, sort_keys=True).encode()).hexdigest()
        # Orientation/crop changes form a distinct review workspace. A label
        # of a clipped ROI cannot silently migrate to a different crop.
        binding = dict(videos={eye: {key: value for key, value in video.items() if key != "path"}
                               for eye, video in self.videos.items()}, split=split)
        dataset_id = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()[:20]
        self.root = Path(data_dir).expanduser().resolve() / dataset_id
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "labels").mkdir(exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.split = split
        manifest = dict(schema_version=1, recording_id=self.recording_id,
                        videos=self.videos, split=split, frame_count=self.frame_count,
                        timestamp_convention="zero-based frame_index / fps")
        manifest_path = self.root / "recording.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            for eye in EYES:
                if any(previous["videos"][eye][key] != self.videos[eye][key]
                       for key in ("video_sha256", "frame_rotation", "roi")):
                    raise ValueError("Saved review recording identity does not match")
        else:
            atomic_json(manifest_path, manifest)
        self.windows = [dict(id="all", name="Entire recording", start=0, end=self.frame_count-1)]
        if all(video["video_sha256"] in KNOWN_DEVELOPMENT_HASHES for video in self.videos.values()):
            for start, end in ((0,23), (168,184), (345,365), (375,414), (956,974), (1160,1179)):
                self.windows.append(dict(id=str(start), name=f"Review frames {start}–{end}", start=start, end=end))

    def labels_for(self, reviewer):
        reviewer = reviewer_name(reviewer)
        path = self.root / "labels" / (reviewer + ".json")
        if not path.exists():
            return dict(schema_version=1, reviewer=reviewer, revision=0,
                        recording_id=self.recording_id, labels=[])
        result = json.loads(path.read_text())
        if result.get("recording_id") != self.recording_id or result.get("reviewer") != reviewer:
            raise ValueError("Saved reviewer file has a mismatched identity")
        return result

    def save_label(self, body):
        """One deliberate human action, with optimistic locking across browser tabs."""
        reviewer = reviewer_name(body.get("reviewer"))
        eye, frame, label = body.get("eye"), body.get("frame_index"), body.get("label")
        if eye not in EYES or type(frame) is not int or not 0 <= frame < self.frame_count:
            raise ValueError("Choose a valid eye and frame")
        if label not in LABELS and label is not None:
            raise ValueError("Choose usable, unusable, uncertain or clear")
        reason = body.get("reason", "")
        if not isinstance(reason, str) or len(reason) > 500:
            raise ValueError("Notes must be text of at most 500 characters")
        with self.lock, (self.root / "labels" / (reviewer + ".lock")).open("a+") as lockfile:
            # Atomic replacement avoids torn JSON; an OS lock additionally
            # prevents lost updates if two launcher processes are open.
            fcntl.flock(lockfile, fcntl.LOCK_EX)
            document = self.labels_for(reviewer)
            if type(body.get("revision")) is not int or body["revision"] != document["revision"]:
                raise ValueError("Labels changed in another tab. Reload this reviewer before saving.")
            document["labels"] = [row for row in document["labels"]
                                  if (row["eye"], row["frame_index"]) != (eye, frame)]
            if label is not None:
                document["labels"].append(dict(eye=eye, frame_index=frame,
                    timestamp_s=frame/self.fps, label=label, reason=reason,
                    reviewer=reviewer, reviewed=True, split=self.split,
                    video_sha256=self.videos[eye]["video_sha256"],
                    frame_rotation=self.videos[eye]["frame_rotation"], roi=self.videos[eye]["roi"],
                    saved_at=datetime.now(timezone.utc).isoformat()))
            document["labels"].sort(key=lambda row:(row["eye"],row["frame_index"]))
            document["revision"] += 1
            atomic_json(self.root / "labels" / (reviewer + ".json"), document)
            return document

    def frame_jpeg(self, eye, frame_index, full=False):
        """Serve raw, rotated image pixels. No prediction/label overlay is drawn."""
        if eye not in EYES or type(frame_index) is not int or not 0 <= frame_index < self.frame_count:
            raise ValueError("invalid frame")
        key = eye, frame_index, full
        with self.frame_lock:
            if key in self.frame_cache:
                return self.frame_cache[key]
            video = self.videos[eye]
            capture = cv2.VideoCapture(video["path"])
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-(frame_index+1)) > .5:
                    raise ValueError("Could not decode the requested frame exactly")
            finally:
                capture.release()
            from video_preparation import rotate_frame
            frame = rotate_frame(frame, video["frame_rotation"])
            if not full:
                x,y,w,h = video["roi"]
                frame = frame[y:y+h, x:x+w]
            else:
                scale = min(1., 900/max(frame.shape[:2]))
                frame = cv2.resize(frame, (round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("Could not prepare image")
            if len(self.frame_cache) >= 32:
                self.frame_cache.pop(next(iter(self.frame_cache)))
            self.frame_cache[key] = encoded.tobytes()
            return self.frame_cache[key]

    def runs(self):
        result = []
        for path in sorted((self.root/"runs").glob("*/job.json"), reverse=True):
            job = json.loads(path.read_text())
            if job.get("status") == "running" and (self.job is None or job["id"] != self.job["id"]):
                job["status"] = "interrupted"
            result.append(job)
        return result

    def state(self, reviewer=None):
        with self.lock:
            document = self.labels_for(reviewer) if reviewer else None
            return dict(recording_id=self.recording_id, frame_count=self.frame_count,
                        fps=self.fps, videos=self.videos, split=self.split, windows=self.windows,
                        labels=document, root=str(self.root), job=self.job, runs=self.runs(),
                        settings=dict(roi_mode=getattr(self.config,"ROI_MODE","fixed"),
                                      min_confidence=self.config.MIN_CONFIDENCE,
                                      recovery_confidence=self.config.RECOVERY_CONFIDENCE,
                                      recovery_duration=self.config.RECOVERY_DURATION_S),
                        reviewers=[p.stem for p in sorted((self.root/"labels").glob("*.json"))])

    def start_run(self, body):
        """Run the same recording twice with controlled settings and fresh models."""
        settings = {}
        for name in ("min_confidence", "recovery_confidence", "recovery_duration"):
            value = body.get(name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
                raise ValueError("Thresholds must be finite numbers")
            settings[name] = float(value)
        if not 0 < settings["min_confidence"] <= settings["recovery_confidence"] <= 1:
            raise ValueError("Require 0 < minimum confidence <= recovery confidence <= 1")
        if not 0 <= settings["recovery_duration"] <= 5:
            raise ValueError("Recovery duration must be between 0 and 5 seconds")
        limit = body.get("max_frames",0)
        if type(limit) is not int or limit < 0 or limit > self.frame_count:
            raise ValueError("Frame limit must be zero (all) or within this recording")
        settings["max_frames"] = limit
        settings["roi_mode"] = body.get("roi_mode", "fixed")
        if settings["roi_mode"] not in ("fixed", "tracked"):
            raise ValueError("Choose fixed or tracked ROI mode")
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                raise ValueError("A comparison is already running")
            if self.split == "held_out":
                if limit:
                    raise ValueError("Held-out comparisons must process the complete recording")
                plan = dict(settings=settings, source_fingerprint=self.config._source_fingerprint())
                plan_path = self.root / "evaluation_plan.json"
                with (self.root/"evaluation_plan.lock").open("a+") as lockfile:
                    fcntl.flock(lockfile, fcntl.LOCK_EX)
                    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
                        raise ValueError("Held-out settings and implementation are frozen. Tune on development recordings, then collect fresh evaluation data.")
                    if not plan_path.exists():
                        atomic_json(plan_path,plan)
            identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(3)
            run_dir = self.root/"runs"/identifier
            run_dir.mkdir()
            self.job = dict(id=identifier, status="running", phase="starting", progress="Preparing baseline",
                            settings=settings, path=str(run_dir), recording_id=self.recording_id,
                            split=self.split, started_at=datetime.now(timezone.utc).isoformat())
            atomic_json(run_dir/"job.json", self.job)
            self.worker = threading.Thread(target=self._run_pair, args=(run_dir,settings), daemon=True)
            self.worker.start()
            return dict(self.job)

    def _run_pair(self, run_dir, settings):
        try:
            for mode in ("baseline", "filtered"):
                if self.stopping.is_set():
                    raise RuntimeError("Comparison stopped because the application closed")
                with self.lock:
                    self.job.update(phase=mode, progress=f"Starting {mode}")
                command = [sys.executable, "-B", str(self.project/"eye_pipeline.py"),
                           "--headless", "--filter-mode", mode, "--device", "cpu",
                           "--roi-mode", settings.get("roi_mode", "fixed"),
                           "--output",str(run_dir/(mode+".jsonl")),
                           "--run-summary",str(run_dir/(mode+"_summary.json")),
                           "--progress-every","50","--max-frames",str(settings["max_frames"]),
                           "--min-confidence",str(settings["min_confidence"]),
                           "--recovery-confidence",str(settings["recovery_confidence"]),
                           "--recovery-duration",str(settings["recovery_duration"])]
                for eye in EYES:
                    command.extend(["--"+eye+"-video", self.videos[eye]["path"],
                                    "--"+eye+"-frame-rotation", self.videos[eye]["frame_rotation"],
                                    "--"+eye+"-calibration", str(getattr(self.config,eye.upper()+"_CALIBRATION_PATH")),
                                    "--"+eye+"-camera-id", getattr(self.config,eye.upper()+"_CAMERA_ID"),
                                    "--"+eye+"-calibration-rotation", getattr(self.config,eye.upper()+"_CALIBRATION_ROTATION")])
                if self.config.REQUIRE_CALIBRATION_IDENTITY:
                    command.append("--require-calibration-identity")
                with (run_dir/(mode+".log")).open("w",encoding="utf-8") as log:
                    # Argument lists avoid shell interpolation of video paths.
                    process = subprocess.Popen(command,cwd=self.project,stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT,text=True,bufsize=1)
                    with self.lock:
                        self.process = process
                        if self.stopping.is_set():
                            process.terminate()
                    for line in process.stdout:
                        log.write(line); log.flush()
                        with self.lock:
                            self.job["progress"] = line.strip()[-500:]
                        if self.stopping.is_set():
                            process.terminate()
                    code = process.wait()
                    with self.lock:
                        self.process = None
                if code:
                    raise RuntimeError(f"{mode} failed; see {mode}.log in the run folder")
                summary = json.loads((run_dir/(mode+"_summary.json")).read_text())
                if summary["status"] != "complete":
                    raise RuntimeError(f"{mode} stopped early or recording length was unknown; inspect its log")
            # Check exact paired run compatibility now, even before human labels.
            report = compare_records(read_jsonl(run_dir/"baseline.jsonl"),read_jsonl(run_dir/"filtered.jsonl"),[],
                                     split=self.split,run_summaries={mode:json.loads((run_dir/(mode+"_summary.json")).read_text()) for mode in ("baseline","filtered")})
            atomic_json(run_dir/"unlabeled_comparison.json",report)
            with self.lock:
                self.job.update(status="complete",phase="complete",progress="Both runs complete. Score your saved labels.")
        except Exception as error:
            with self.lock:
                self.job.update(status="failed",phase="failed",progress=str(error))
        finally:
            with self.lock:
                atomic_json(run_dir/"job.json",self.job)

    def score(self, body):
        reviewer = reviewer_name(body.get("reviewer"))
        identifier = body.get("run_id")
        if not isinstance(identifier,str) or re.fullmatch(r"[0-9]{8}T[0-9]{6}_[a-f0-9]{6}",identifier) is None:
            raise ValueError("Select a completed comparison")
        run_dir = self.root/"runs"/identifier
        job = json.loads((run_dir/"job.json").read_text())
        if job["status"] != "complete":
            raise ValueError("The selected comparison has not completed successfully")
        with self.lock:
            document = self.labels_for(reviewer)
        baseline,filtered = (read_jsonl(run_dir/(mode+".jsonl")) for mode in ("baseline","filtered"))
        summaries = {mode:json.loads((run_dir/(mode+"_summary.json")).read_text()) for mode in ("baseline","filtered")}
        report = compare_records(baseline,filtered,document["labels"],split=self.split,run_summaries=summaries)
        report["reviewer"] = reviewer
        report["label_revision"] = document["revision"]
        report["run_id"] = identifier
        report["recording_id"] = self.recording_id
        report["reviewer_agreement"] = reviewer_agreement([row for p in (self.root/"labels").glob("*.json") for row in json.loads(p.read_text())["labels"]])
        atomic_json(run_dir/("report_"+reviewer+".json"),report)
        text = format_comparison_report(report)
        (run_dir/("report_"+reviewer+".md")).write_text(text,encoding="utf-8")
        return dict(report=report, markdown=text, path=str(run_dir/("report_"+reviewer+".md")))

    def close(self):
        self.stopping.set()
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
        if self.worker is not None:
            self.worker.join(timeout=5)


def make_handler(workspace):
    """No arbitrary file serving or command execution; only explicit local actions."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_bytes(self, status, data, content_type):
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(data)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            self.wfile.write(data)

        def reply(self,value,status=200):
            self.send_bytes(status,json.dumps(value,allow_nan=False).encode(),"application/json")

        def local_request(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected:
                self.reply({"error":"Only the local application address is allowed"},403)
                return False
            return True

        def do_GET(self):
            if not self.local_request(): return
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    if query.get("token",[None])[0] != workspace.token:
                        self.reply({"error":"Open the URL printed by Blink Review.command"},403);return
                    html = HTML.replace("__SESSION_TOKEN__",workspace.token)
                    self.send_bytes(200,html.encode(),"text/html; charset=utf-8")
                elif parsed.path == "/api/state":
                    if self.headers.get("X-Session") != workspace.token:
                        self.reply({"error":"Session expired; reopen the launcher URL"},403);return
                    self.reply(workspace.state(query.get("reviewer",[None])[0]))
                elif parsed.path == "/frame":
                    if query.get("token",[None])[0] != workspace.token:
                        self.reply({"error":"Session expired"},403);return
                    data = workspace.frame_jpeg(query.get("eye",[""])[0],int(query.get("index",["-1"])[0]),query.get("full",["0"])[0]=="1")
                    self.send_bytes(200,data,"image/jpeg")
                else:
                    self.reply({"error":"Not found"},404)
            except (ValueError,OSError,KeyError,TypeError) as error:
                self.reply({"error":str(error)},400)

        def do_POST(self):
            if not self.local_request(): return
            if self.headers.get("X-Session") != workspace.token:
                self.reply({"error":"Session expired; reopen the launcher URL"},403);return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://127.0.0.1:{self.server.server_port}":
                self.reply({"error":"Cross-origin requests are not allowed"},403);return
            try:
                length = int(self.headers.get("Content-Length","0"))
                if not 0 < length <= 16384: raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body,dict): raise ValueError("Request must be an object")
                actions = {"/api/label":workspace.save_label,"/api/run":workspace.start_run,"/api/score":workspace.score}
                action = actions.get(urlparse(self.path).path)
                if action is None:
                    self.reply({"error":"Not found"},404);return
                self.reply(action(body))
            except (ValueError,OSError,KeyError,TypeError) as error:
                self.reply({"error":str(error)},400)
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=HERE)
    parser.add_argument("--data-dir",type=Path)
    parser.add_argument("--left-video",type=Path)
    parser.add_argument("--right-video",type=Path)
    parser.add_argument("--split",choices=("development","held_out"),default="development")
    parser.add_argument("--port",type=int,default=0,help="zero chooses a free local port")
    parser.add_argument("--no-browser",action="store_true")
    args = parser.parse_args(argv)
    videos = []
    for eye, explicit in (("left",args.left_video),("right",args.right_video)):
        candidates = [explicit] if explicit else sorted(args.project.glob("*"+eye+"eye.mp4"))
        if len(candidates) != 1:
            parser.error(f"Specify --{eye}-video; expected exactly one {eye}-eye MP4 in the project")
        videos.append(candidates[0])
    try:
        workspace = ReviewWorkspace(args.project,args.data_dir or args.project/"blink_validation_data",*videos,split=args.split)
        server = ThreadingHTTPServer(("127.0.0.1",args.port),make_handler(workspace))
    except (ValueError,OSError) as error:
        parser.error(str(error))
    url = f"http://127.0.0.1:{server.server_port}/?token={workspace.token}"
    print("Blink Review is ready: "+url,flush=True)
    print("Labels and reports: "+str(workspace.root),flush=True)
    print("Keep this terminal open. Press Control-C to stop.",flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        workspace.close()


if __name__ == "__main__":
    main()
