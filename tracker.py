#!/usr/bin/env python3
"""Phone-displayed ArUco tracking. All lengths are millimetres.

Independent implementation of the OpenCV marker/solvePnP workflow.
See README.md for sources, camera coordinates, and calibration requirements.
"""
import argparse
import csv
import math
from pathlib import Path
import time

import cv2
import numpy as np


DICT = cv2.aruco.DICT_4X4_100

# Parameters, can override with command-line args
MARKER_ID = 1          # ArUco ID to track / generate
MARKER_MM = 35.0       # printed/displayed side length of the black square, in mm
WIDTH = 1920           # webcam resolution; must match calibration
HEIGHT = 1080
DEFAULT_CALIBRATION = Path(__file__).resolve().parent / 'camera_calibration.npz'


def detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)


class Deblurrer:
    """Geometry-preserving defocus recovery applied to the grayscale detection image only.

    Stages are tried cheapest first until the wanted marker is found:
    raw -> unsharp mask -> Wiener deconvolution with Gaussian PSF of growing sigma.
    Neither filter moves edges, so corner positions (and pose) stay valid.
    """
    SIGMAS = (1.5, 3.0, 5.0)    # assumed defocus blur radii in px, mild to strong
    NSR = 0.01                  # Wiener noise-to-signal ratio; higher = less ringing

    def __init__(self, shape):
        self.shape = shape
        fy = np.fft.fftfreq(shape[0])[:, None]
        fx = np.fft.rfftfreq(shape[1])[None, :]
        r2 = fx ** 2 + fy ** 2
        self.filters = []
        for sigma in self.SIGMAS:
            h = np.exp(-2 * math.pi ** 2 * sigma ** 2 * r2)
            self.filters.append(h / (h ** 2 + self.NSR))     # real PSF -> real Wiener filter

    def unsharp(self, gray):
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        return cv2.addWeighted(gray, 2.0, blur, -1.0, 0)

    def wiener(self, gray, index):
        spectrum = np.fft.rfft2(gray.astype(np.float32))
        restored = np.fft.irfft2(spectrum * self.filters[index], s=self.shape)
        return np.clip(restored, 0, 255).astype(np.uint8)

    def stages(self, gray):
        yield 'raw', gray
        yield 'unsharp', self.unsharp(gray)
        for i, sigma in enumerate(self.SIGMAS):
            yield f'deblur s={sigma}', self.wiener(gray, i)


def detect_deblurred(detect, deblurrer, gray, marker_id):
    """Return (corners, ids, stage_name); stops at the first stage that finds marker_id."""
    best = (), None, 'raw'
    for name, image in deblurrer.stages(gray):
        corners, ids, _ = detect.detectMarkers(image)
        if name == 'raw':
            best = corners, ids, name
        if ids is not None and marker_id in ids.flatten():
            return corners, ids, name
    return best


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Could not write {path}')
    print(f'Saved {path.resolve()}')


def marker_image(marker_id=1, pixels=1000):
    image = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(DICT), marker_id, pixels)
    margin = pixels // 8
    return cv2.copyMakeBorder(image, margin, margin, margin, margin,
                              cv2.BORDER_CONSTANT, value=255)


def object_points(length):
    h = length / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def estimate_pose(corners, length, matrix, distortion):
    points = object_points(length)
    image_points = np.ascontiguousarray(corners, dtype=np.float64).reshape(4, 2)
    result = cv2.solvePnPGeneric(points, image_points, matrix, distortion,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    for rvec, tvec in zip(result[1], result[2]):
        rotation = cv2.Rodrigues(rvec)[0]
        depths = (rotation @ points.T + tvec.reshape(3, 1))[2]
        if not np.isfinite(tvec).all() or np.any(depths <= 0):
            continue
        projected = cv2.projectPoints(points, rvec, tvec, matrix, distortion)[0].reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
        if math.isfinite(error):
            candidates.append((error, rvec, tvec))
    return min(candidates, key=lambda item: item[0]) if candidates else None


def euler_degrees(rotation):
    """R = Rz(yaw) Ry(pitch) Rx(roll), marker to camera rotation."""
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    if sy > 1e-6:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0
    return np.degrees([roll, pitch, yaw])


def open_camera(args):
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Webcam unavailable. Grant camera permission or try --camera 1.')
    return cap


def read_frame(cap):
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError('Could not read webcam frame. Close other camera applications.')
    return frame


def text(frame, message, line, color=(255, 255, 255)):
    scale = max(.8, frame.shape[0] / 720 * .8)      # grows with resolution
    thickness = max(2, round(scale * 2))
    step = int(scale * 46)
    y = int(scale * 40) + line * step
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(message, font, scale, thickness)
    x = 12
    # semi-transparent dark box behind the text for contrast on any background
    x0, y0 = max(x - 8, 0), max(y - h - 8, 0)
    x1, y1 = min(x + w + 8, frame.shape[1]), min(y + 10, frame.shape[0])
    roi = frame[y0:y1, x0:x1]
    roi[:] = (roi * .35).astype(roi.dtype)
    cv2.putText(frame, message, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def camera_model(path, size):
    with np.load(path, allow_pickle=False) as data:
        matrix = data['camera_matrix'].astype(np.float64)
        distortion = data['dist_coeffs'].astype(np.float64)
        calibrated_size = tuple(int(x) for x in data['image_size'])
    if calibrated_size != tuple(size):
        raise ValueError(f'Calibration resolution {calibrated_size} differs from webcam {size}. '
                         'Use the same --width/--height as calibration; otherwise recalibrate.')
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Invalid calibration arrays.')
    return matrix, distortion


def track(args):
    cap = open_camera(args)
    log = None
    try:
        frame = read_frame(cap)
        height, width = frame.shape[:2]
        if args.calibration:
            matrix, distortion = camera_model(args.calibration, (width, height))
            mode = 'CALIBRATED'
        else:
            focal = width / (2 * math.tan(math.radians(args.hfov) / 2))
            matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], np.float64)
            distortion = np.zeros(5)
            mode = 'APPROXIMATE: assumed FOV; calibrate for measurements'
            print(mode)
        if args.csv:
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            log = open(args.csv, 'w', newline='', encoding='utf-8')
            writer = csv.writer(log)
            writer.writerow(['unix_time_s', 'elapsed_s', 'detected', 'id', 'calibrated',
                             'x_mm', 'y_mm', 'z_mm', 'distance_mm', 'roll_deg', 'pitch_deg',
                             'yaw_deg', 'rx_rad', 'ry_rad', 'rz_rad', 'reprojection_rms_px'])
        detect = detector()
        deblurrer = Deblurrer((height, width))
        start = time.monotonic()
        while True:
            if frame.shape[:2] != (height, width):
                raise RuntimeError('Camera resolution changed; restart and recalibrate.')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if args.deblur:
                corners, ids, stage = detect_deblurred(detect, deblurrer, gray, args.id)
            else:
                corners, ids, _ = detect.detectMarkers(gray)
                stage = 'off'
            pose = None
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                matches = np.flatnonzero(ids.flatten() == args.id)
                if matches.size:
                    pose = estimate_pose(corners[int(matches[0])], args.marker_mm, matrix, distortion)
            now = time.time()
            elapsed = time.monotonic() - start
            text(frame, mode, 0, (0, 255, 255))
            text(frame, f'q: quit | s: save annotated image | deblur: {stage}', 1)
            if pose is None:
                text(frame, f'ID {args.id}: NOT VISIBLE (no current pose)', 2, (0, 100, 255))
                if log:
                    writer.writerow([now, elapsed, 0, args.id, int(bool(args.calibration))] + [''] * 11)
            else:
                error, rvec, tvec = pose
                xyz = tvec.ravel()
                distance = float(np.linalg.norm(xyz))
                angles = euler_degrees(cv2.Rodrigues(rvec)[0])
                cv2.drawFrameAxes(frame, matrix, distortion, rvec, tvec, args.marker_mm * .5, 2)
                text(frame, f'ID {args.id} | X {xyz[0]:.1f}  Y {xyz[1]:.1f}  Z {xyz[2]:.1f} mm', 2)
                text(frame, f'Distance {distance:.1f} mm | corner error {error:.2f} px', 3)
                text(frame, f'Roll {angles[0]:.1f}  Pitch {angles[1]:.1f}  Yaw {angles[2]:.1f} deg', 4)
                if log:
                    writer.writerow([now, elapsed, 1, args.id, int(bool(args.calibration)),
                                     *xyz, distance, *angles, *rvec.ravel(), error])
            cv2.imshow('Phone ArUco pose tracker', frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                write_image(f'capture_{time.time_ns()}.png', frame)
            frame = read_frame(cap)
    finally:
        if log:
            log.close()
        cap.release()
        cv2.destroyAllWindows()


def chessboard_image():
    # 10 x 7 squares -> 9 x 6 internal corners. White margin is not a square.
    board = np.full((900, 1200), 255, np.uint8)
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                board[100 + row * 100:200 + row * 100,
                      100 + col * 100:200 + col * 100] = 0
    return board


def calibrate(args):
    cap = open_camera(args)
    image_points, world_points = [], []
    grid = np.zeros((9 * 6, 3), np.float32)
    grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * args.square_mm
    size = None
    try:
        while True:
            frame = read_frame(cap)
            current_size = (frame.shape[1], frame.shape[0])
            if size is not None and size != current_size:
                raise RuntimeError('Camera resolution changed during calibration.')
            size = current_size
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCornersSB(gray, (9, 6))
            if found:
                cv2.drawChessboardCorners(frame, (9, 6), corners, True)
            text(frame, f'Calibration views: {len(image_points)} | board {"FOUND" if found else "not found"}', 0)
            text(frame, 'SPACE: capture | c: calibrate/save (15+ views) | q: quit', 1)
            cv2.imshow('Webcam calibration', frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                return
            if key == ord(' ') and found:
                image_points.append(corners.astype(np.float32))
                world_points.append(grid.copy())
                print(f'Captured view {len(image_points)}; vary tilt, position, and distance next.')
            if key == ord('c'):
                if len(image_points) < 15:
                    print('Capture at least 15 varied views first.')
                    continue
                rms, matrix, distortion, _, _ = cv2.calibrateCamera(world_points, image_points, size, None, None)
                if not math.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
                    raise RuntimeError('Calibration failed. Try sharper, more varied views.')
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                with output.open('wb') as handle:
                    np.savez(handle, camera_matrix=matrix, dist_coeffs=distortion,
                             image_size=np.array(size), rms_px=rms)
                print(f'Saved {output.resolve()} | RMS {rms:.3f} px (lower is better).')
                if rms > 1:
                    print('RMS exceeds 1 px: consider repeating with sharper and more varied views.')
                return
    finally:
        cap.release()
        cv2.destroyAllWindows()


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    marker = sub.add_parser('marker', help='Export a phone-friendly marker PNG')
    marker.add_argument('--id', type=int, choices=range(100), default=MARKER_ID)
    marker.add_argument('--output', default='marker.png')
    sub.add_parser('board', help='Export calibration chessboard PNG').add_argument('--output', default='chessboard.png')
    cal = sub.add_parser('calibrate', help='Calibrate your webcam with the chessboard')
    cal.add_argument('--square-mm', type=positive, required=True)
    cal.add_argument('--output', default='camera.npz')
    tracking = sub.add_parser('track', help='Track marker using the laptop webcam')
    tracking.add_argument('--id', type=int, choices=range(100), default=MARKER_ID)
    tracking.add_argument('--marker-mm', type=positive, default=MARKER_MM)
    tracking.add_argument('--calibration', default=str(DEFAULT_CALIBRATION),
                          help='calibration .npz (default: %(default)s); pass "" for approximate preview')
    tracking.add_argument('--hfov', type=positive, default=60, help='Assumed horizontal FOV for uncalibrated preview')
    tracking.add_argument('--no-deblur', dest='deblur', action='store_false',
                          help='disable the defocus-recovery step before detection')
    tracking.add_argument('--csv', help='Optional output CSV; overwritten on each run')
    for camera_parser in (cal, tracking):
        camera_parser.add_argument('--camera', type=int, default=0)
        camera_parser.add_argument('--width', type=int, default=None,
                                   help=f'default: calibration width for track, else {WIDTH}')
        camera_parser.add_argument('--height', type=int, default=None,
                                   help=f'default: calibration height for track, else {HEIGHT}')
    args = parser.parse_args()
    if args.command == 'track' and args.calibration and args.width is None and args.height is None:
        try:
            with np.load(args.calibration, allow_pickle=False) as data:
                args.width, args.height = (int(x) for x in data['image_size'])
        except (OSError, KeyError, ValueError) as error:
            parser.exit(1, f'Error: cannot read calibration size: {error}\n')
    if args.command in ('calibrate', 'track'):
        if args.width is None:
            args.width = WIDTH
        if args.height is None:
            args.height = HEIGHT
    if args.command == 'track' and args.hfov >= 179:
        parser.error('--hfov must be less than 179 degrees')
    try:
        if args.command == 'marker':
            write_image(args.output, marker_image(args.id))
        elif args.command == 'board':
            write_image(args.output, chessboard_image())
        elif args.command == 'calibrate':
            calibrate(args)
        else:
            track(args)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
