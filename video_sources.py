# Import dependencies

import threading
import time

import cv2
import numpy as np
import zmq

from video_preparation import (
    matching_video_fps,
    open_video,
    read_first_frame,
    reset_video,
)


# Receives the newest frame from one eye camera stream started with webcam_stream.py. A background thread keeps
# receiving so the Frame Loop always gets the most recent frame instead of an old queued one - class

class EyeStream:
    def __init__(self, host, port, side, timeout_s):
        self.address = f"tcp://{host}:{port}"
        self.side = side
        self.timeout_s = float(timeout_s)
        self.start_time = time.perf_counter()

        self.frame = None
        self.frame_time = None
        self.frame_count = 0
        self.last_read_count = 0
        self.new_frame = threading.Condition()

        self.running = True
        self.thread = threading.Thread(target=self._receive_frames, daemon=True)
        self.thread.start()

    # Runs on background thread. Decodes each JPEG frame and stores it with its arrival time - function

    def _receive_frames(self):
        socket = zmq.Context.instance().socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.RCVTIMEO, 200)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(self.address)
        try:
            while self.running:
                try:
                    message = socket.recv()
                except zmq.Again:
                    continue

                # Arrival time is used for pye3d timestamps since each stream machine has its own clock
                arrival_time = time.perf_counter()
                frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with self.new_frame:
                    self.frame = frame
                    self.frame_time = arrival_time - self.start_time
                    self.frame_count += 1
                    self.new_frame.notify_all()
        finally:
            socket.close()

    # Waits for a frame newer than the last one read. Returns (None, None) if stream times out - function

    def read(self):
        with self.new_frame:
            received = self.new_frame.wait_for(
                lambda: self.frame_count > self.last_read_count,
                timeout=self.timeout_s,
            )
            if not received:
                return None, None
            self.last_read_count = self.frame_count
            return self.frame, self.frame_time

    def preview_frame(self):
        frame, _ = self.read()
        if frame is None:
            raise RuntimeError(
                f"no frames from {self.side} eye stream at {self.address} "
                f"after {self.timeout_s} s. Is webcam_stream.py running?"
            )
        return frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)


# Reads frames from a recorded eye video. Timestamps come from frame number and video FPS - class

class VideoFileSource:
    def __init__(self, capture, side, fps):
        self.capture = capture
        self.side = side
        self.fps = fps
        self.frame_index = 0

    def read(self):
        success, frame = self.capture.read()
        if not success:
            return None, None
        timestamp_s = self.frame_index / self.fps
        self.frame_index += 1
        return frame, timestamp_s

    # Reads first frame for rotation setup GUI, then rewinds video back to frame 0 - function

    def preview_frame(self):
        frame = read_first_frame(self.capture, self.side)
        reset_video(self.capture, self.side)
        return frame

    def release(self):
        self.capture.release()


# Opens left and right eye videos and checks they have matching FPS - function

def open_video_pair(left_path, right_path):
    left_capture = open_video(left_path, "left")
    right_capture = None
    try:
        right_capture = open_video(right_path, "right")
        fps = matching_video_fps(left_capture, right_capture)
    except Exception:
        left_capture.release()
        if right_capture is not None:
            right_capture.release()
        raise
    return (
        VideoFileSource(left_capture, "left", fps),
        VideoFileSource(right_capture, "right", fps),
    )
