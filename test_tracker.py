"""Geometry and detection checks, independent of a physical webcam."""
import unittest
import cv2
import numpy as np
import tracker


class TrackingTests(unittest.TestCase):
    def test_generated_marker_detects_correct_id(self):
        for marker_id in (0, 17, 99):
            corners, ids, _ = tracker.detector().detectMarkers(tracker.marker_image(marker_id))
            self.assertEqual(ids.flatten().tolist(), [marker_id])
            self.assertEqual(len(corners), 1)

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
