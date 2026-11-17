"""Geometry and detection checks, independent of a physical webcam."""
import unittest
import cv2
import numpy as np
import tracker


class TrackingTests(unittest.TestCase):
    class FakeCapture:
        def __init__(self):
            self.focus_values = []

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_FOCUS:
                self.focus_values.append(value)
            return True

    def test_generated_marker_detects_correct_id(self):
        for marker_id in (0, 17, 99):
            corners, ids, _ = tracker.detector().detectMarkers(tracker.marker_image(marker_id))
            self.assertEqual(ids.flatten().tolist(), [marker_id])
            self.assertEqual(len(corners), 1)

    def test_detector_uses_relaxed_tolerances(self):
        params = tracker.detector_parameters()
        self.assertEqual(params.minMarkerPerimeterRate, 0.015)
        self.assertEqual(params.maxErroneousBitsInBorderRate, 0.80)
        self.assertEqual(params.errorCorrectionRate, 0.90)

    def test_preprocessed_detection_finds_blurred_marker(self):
        marker_id = 17
        image = tracker.marker_image(marker_id, pixels=300)
        blurred = cv2.GaussianBlur(image, (0, 0), 3.0)
        deblurrer = tracker.Deblurrer(blurred.shape)

        corners, ids, stage = tracker.detect_deblurred(
            tracker.detector(), deblurrer, blurred, marker_id
        )

        self.assertIsNotNone(ids)
        self.assertIn(marker_id, ids.flatten())
        self.assertIn(stage, ("raw", "unsharp", "deblur s=1.5", "deblur s=3.0", "deblur s=5.0"))
        self.assertGreater(len(corners), 0)

    def test_downsampled_threshold_corners_restore_to_source_scale(self):
        marker_id = 17
        source = cv2.cvtColor(tracker.marker_image(marker_id, pixels=300), cv2.COLOR_GRAY2BGR)
        scale = 0.5
        gray = tracker.downsample_gray(source, scale)
        corners, ids, _ = tracker.detect_deblurred(
            tracker.detector(), tracker.Deblurrer(gray.shape), gray, marker_id, threshold=True
        )
        restored = tracker.restore_corner_scale(corners, scale)

        self.assertIn(marker_id, ids.flatten())
        self.assertGreater(float(np.max(restored[0])), float(np.max(corners[0])))

    def test_stepped_autofocus_only_uses_configured_positions(self):
        capture = self.FakeCapture()
        positions = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
        controller = tracker.SteppedAutofocus(
            capture, positions, settle_frames=0, hold_frames=2
        )
        frame = np.zeros((32, 32, 3), np.uint8)
        for _ in range(10):
            controller.update(frame)

        self.assertTrue(capture.focus_values)
        self.assertTrue(set(capture.focus_values).issubset(set(positions)))
        self.assertTrue(controller.holding)

    def test_known_pose_recovered(self):
        matrix = np.array([[1000., 0, 640], [0, 1000., 360], [0, 0, 1]])
        distortion = np.array([-.05, .01, 0, 0, 0.])
        rvec = np.array([2.8, .2, -.15])
        tvec = np.array([30., -25., 450.])
        points = cv2.projectPoints(tracker.object_points(50), rvec, tvec, matrix, distortion)[0]
        error, recovered_rvec, recovered_tvec = tracker.estimate_pose(points, 50, matrix, distortion)
        np.testing.assert_allclose(recovered_tvec.ravel(), tvec, atol=1e-4)
        np.testing.assert_allclose(cv2.Rodrigues(recovered_rvec)[0], cv2.Rodrigues(rvec)[0], atol=1e-5)
        self.assertLess(error, 1e-5)

    def test_marker_size_scales_translation(self):
        matrix = np.array([[1000., 0, 640], [0, 1000., 360], [0, 0, 1]])
        distortion = np.zeros(5)
        points = cv2.projectPoints(tracker.object_points(50), np.array([2.8, .2, -.15]),
                                  np.array([30., -25., 450.]), matrix, distortion)[0]
        pose = tracker.estimate_pose(points, 100, matrix, distortion)
        np.testing.assert_allclose(pose[2].ravel(), [60, -50, 900], atol=1e-4)

    def test_calibration_resolution_rejected(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'camera.npz'
            np.savez(path, camera_matrix=np.eye(3), dist_coeffs=np.zeros(5), image_size=[1280, 720])
            with self.assertRaises(ValueError):
                tracker.camera_model(path, (640, 480))

    def test_calibration_board_detectable(self):
        found, corners = cv2.findChessboardCornersSB(tracker.chessboard_image(), (9, 6))
        self.assertTrue(found)
        self.assertEqual(len(corners), 54)


if __name__ == '__main__':
    unittest.main()
