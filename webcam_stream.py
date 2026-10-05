# Import dependencies

import argparse

import cv2
import zmq


# Stream preset variables. Frame size must match camera_calibration.npz

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
JPEG_QUALITY = 80


# Opens eye camera at the calibrated frame size - function

def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera {camera_index}")

    # MJPG lets most USB cameras reach full frame rate at 1920x1080
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
        print(f"Warning: camera gives {width}x{height}, calibration expects {FRAME_WIDTH}x{FRAME_HEIGHT}")
    return camera


# Sends each camera frame as a JPEG. eye_pipeline only keeps the newest frame - function

def stream_camera(camera_index, port):
    camera = open_camera(camera_index)
    socket = zmq.Context().socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.bind(f"tcp://*:{port}")
    print(f"Streaming camera {camera_index} on port {port}")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                socket.send(jpg.tobytes())
    finally:
        camera.release()
        socket.close()


# Run once per eye camera, e.g. "python webcam_stream.py --camera 1 --port 5555" for the left eye

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream one eye camera to eye_pipeline")
    parser.add_argument("--camera", type=int, default=1, help="camera index on this machine")
    parser.add_argument("--port", type=int, default=5555, help="5555 for left eye, 5556 for right eye")
    args = parser.parse_args()
    stream_camera(args.camera, args.port)
