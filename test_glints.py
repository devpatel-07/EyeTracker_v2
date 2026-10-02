"""Synthetic contracts; these tests do not establish real gaze accuracy."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from glint_detection import GlintConfig, GlintDetector, draw_glints, file_sha256
from glint_geometry import reconstruct, main, template


def fixture():
    """Forward-create reflections from known surface points, not inverse solver.

    Reflection is reversible: choose a sphere point and camera direction, then
    place a synthetic LED along the reflected outgoing direction. For the pupil,
    construct a transmitted ray using tangential Snell components, place P on
    it, and measure K. The inverse code must recover C and P independently.
    """
    norm = lambda x: np.array(x)/np.linalg.norm(x)
    center = np.array([1., -.8, 35.])
    radius, index = 7.8, 1.336
    rays, leds = [], []
    for normal in ([.1, .12, -1], [-.1, .16, -1], [.2, -.1, -1], [-.18, -.08, -1]):
        normal = norm(normal)
        surface = center+radius*normal
        to_camera = norm(-surface)
        to_light = 2*np.dot(normal, to_camera)*normal-to_camera
        leds.append(surface+30*to_light)
        rays.append(norm(surface))
    normal = norm([.12, -.06, -1])
    q = center+radius*normal
    ray = norm(q)
    tangent = (ray-np.dot(ray, normal)*normal)/index
    transmitted = tangent-np.sqrt(1-np.dot(tangent, tangent))*normal
    pupil = q+3.3*transmitted
    return np.array(rays), np.array(leds), ray, center, pupil, dict(
        radius_mm=radius, pupil_offset_mm=float(np.linalg.norm(pupil-center)),
        refractive_index=index, depth_bounds_mm=[15, 60])


class DetectionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((160, 240), 40, np.uint8)
        self.roi = (20, 20, 200, 120)
        self.ellipse = ((120., 80.), (100., 80.), 0.)
        self.good = SimpleNamespace(allow_model_update=True, state="usable", reason="passed")
        self.detector = GlintDetector()

    def detect(self):
        return self.detector.detect(self.frame, self.roi, quality=self.good, pupil_ellipse=self.ellipse)

    def test_full_frame_coordinates_and_no_mutation(self):
        cv2.circle(self.frame, (83, 65), 4, 255, -1)
        before = self.frame.copy()
        result = self.detect()
        self.assertEqual(len(result["candidates"]), 1)
        np.testing.assert_allclose(result["candidates"][0]["center_px"], [83, 65], atol=.01)
        self.assertIsNone(result["candidates"][0]["led_id"])
        np.testing.assert_array_equal(before, self.frame)

    def test_blink_gates_before_pixel_access(self):
        bad = SimpleNamespace(allow_model_update=False, state="rejected", reason="blink")
        result = self.detector.detect(None, None, quality=bad, pupil_ellipse=None)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["candidates"], [])

    def test_recovery_never_reuses_candidates(self):
        cv2.circle(self.frame, (80, 80), 4, 255, -1)
        self.assertTrue(self.detect()["candidates"])
        self.good.state = "recovering"
        self.assertEqual(self.detect()["candidates"], [])

    def test_no_forced_six_glints(self):
        for x in (70, 100, 140):
            cv2.circle(self.frame, (x, 80), 4, 255, -1)
        self.assertEqual(len(self.detect()["candidates"]), 3)

    def test_saturation_is_warning_not_fake_confidence(self):
        cv2.circle(self.frame, (100, 80), 4, 255, -1)
        item = self.detect()["candidates"][0]
        self.assertTrue(item["warnings"])
        self.assertNotIn("confidence", item)

    def test_crop_edge_rejected(self):
        cv2.circle(self.frame, (20, 70), 4, 255, -1)
        result = self.detect()
        self.assertFalse(result["candidates"])
        self.assertIn("touches_crop_border", result["rejected"][0]["reasons"])

    def test_elongated_merged_blob_rejected(self):
        self.frame[70:74, 70:110] = 255
        self.assertFalse(self.detect()["candidates"])

    def test_weak_brightness_and_dark_frame(self):
        cv2.circle(self.frame, (100, 80), 4, 220, -1)
        self.assertFalse(self.detect()["candidates"])
        self.frame[:] = 0
        self.assertFalse(self.detect()["candidates"])

    def test_low_local_contrast(self):
        self.frame[:] = 195
        cv2.circle(self.frame, (100, 80), 5, 205, -1)
        self.frame[80, 100] = 240
        self.assertFalse(self.detect()["candidates"])

    def test_gray_and_bgr_agree(self):
        cv2.circle(self.frame, (100, 80), 4, 255, -1)
        first = self.detect()["candidates"]
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_GRAY2BGR)
        self.assertEqual(first, self.detect()["candidates"])

    def test_invalid_settings_and_image(self):
        for kwargs in (dict(seed_threshold=199), dict(min_area=0), dict(min_contrast=float('nan'))):
            with self.assertRaises(ValueError):
                GlintConfig(**kwargs)
        self.frame = self.frame.astype(float)
        with self.assertRaises(ValueError):
            self.detect()

    def test_overlay_does_not_change_source(self):
        before = self.frame.copy()
        panel = draw_glints(self.frame, self.roi, self.detect(), self.ellipse)
        self.assertEqual(panel.shape, (120, 200, 3))
        np.testing.assert_array_equal(before, self.frame)

    def test_subpixel_gaussian_center(self):
        yy, xx = np.mgrid[:160, :240]
        self.frame = np.round(40+210*np.exp(-((xx-103.3)**2+(yy-77.6)**2)/18)).astype(np.uint8)
        result = self.detect()
        self.assertEqual(len(result['candidates']), 1)
        np.testing.assert_allclose(result['candidates'][0]['center_px'], [103.3,77.6], atol=.3)


class GeometryTests(unittest.TestCase):
    def test_recovers_forward_generated_geometry(self):
        rays, leds, ray, center, pupil, settings = fixture()
        result = reconstruct(rays, leds, ray, **settings)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(result["corneal_center_mm"], center, atol=1e-5)
        np.testing.assert_allclose(result["pupil_center_mm"], pupil, atol=1e-5)
        np.testing.assert_allclose(result["optical_axis"], (pupil-center)/np.linalg.norm(pupil-center), atol=1e-5)
        self.assertFalse(result["visual_gaze_available"])
        self.assertFalse(result["robot_allowed"])

    def test_two_matched_glints_are_sufficient_for_fixture(self):
        rays, leds, ray, center, _, settings = fixture()
        result = reconstruct(rays[:2], leds[:2], ray, **settings)
        self.assertTrue(result["valid"], result)
        np.testing.assert_allclose(result["corneal_center_mm"], center, atol=1e-5)

    def test_degenerate_planes_rejected(self):
        rays, leds, ray, _, _, settings = fixture()
        result = reconstruct([rays[0], rays[0]], [leds[0], leds[0]], ray, **settings)
        self.assertFalse(result["valid"])

    def test_wrong_correspondence_rejected(self):
        rays, leds, ray, _, _, settings = fixture()
        result = reconstruct(rays, leds[::-1], ray, **settings)
        self.assertFalse(result["valid"])

    def test_small_ray_noise_has_bounded_error(self):
        rays, leds, ray, center, _, settings = fixture()
        noisy = rays+np.random.default_rng(7).normal(0, 1e-5, rays.shape)
        result = reconstruct(noisy, leds, ray, **settings)
        self.assertTrue(result["valid"], result)
        self.assertLess(np.linalg.norm(np.array(result["corneal_center_mm"])-center), .02)

    def test_invalid_anatomy_missing_glints_and_bounds(self):
        rays, leds, ray, _, _, settings = fixture()
        self.assertFalse(reconstruct(rays[:1], leds[:1], ray, **settings)["valid"])
        self.assertFalse(reconstruct(rays, leds, ray, **{**settings, "pupil_offset_mm": 20})["valid"])
        self.assertFalse(reconstruct(rays, leds, ray, **{**settings, "depth_bounds_mm": [45, 60]})["valid"])

    def test_consistent_scale_changes_lengths_not_direction(self):
        rays, leds, ray, center, pupil, settings = fixture()
        settings.update(radius_mm=settings['radius_mm']*2, pupil_offset_mm=settings['pupil_offset_mm']*2,
                        depth_bounds_mm=[30, 120])
        result = reconstruct(rays, leds*2, ray, **settings)
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(result['corneal_center_mm'], center*2, atol=1e-5)
        np.testing.assert_allclose(result['optical_axis'], (pupil-center)/np.linalg.norm(pupil-center), atol=1e-5)

    def test_null_template_is_not_executable_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main(['--write-template',str(root/'config.json')])
            with self.assertRaises(FileExistsError):
                main(['--write-template',str(root/'config.json')])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--detections',str(root/'missing.jsonl'),'--config',str(root/'config.json'),
                      '--matches',str(root/'matches.json'),'--output-dir',str(root/'out')])
            self.assertFalse((root/'out').exists())

    def test_file_workflow_and_rotation(self):
        rays, leds, ray, center, _, settings = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
            np.savez(root/'camera.npz', camera_matrix=camera, dist_coeffs=np.zeros(5),
                     image_size=[640, 480], camera_id='test')
            # Clockwise-rotated image: (u,v) -> (479-v,u). Solver must undo it.
            def pixel(direction):
                u, v = (camera@direction)[:2]/direction[2]
                return [479-float(v), float(u)]
            data = root/'detections.jsonl'
            record = dict(frame_index=0, timestamp_s=0., eye='left',
                          pupil_ellipse=[pixel(ray), [50, 50], 0],
                          quality=dict(allow_model_update=True, state='usable'),
                          glints=dict(candidates=[dict(candidate_id=i+1, center_px=pixel(r)) for i, r in enumerate(rays)]))
            bad_record = {**record, 'frame_index':1, 'quality':dict(allow_model_update=False, state='rejected')}
            data.write_text(json.dumps(record)+'\n'+json.dumps(bad_record)+'\n')
            (root/'metadata.json').write_text(json.dumps(dict(rotation='clockwise', eye='left',
                                                             source_size=[640, 480], processed_size=[480, 640])))
            config = template()
            config.update(calibration_path='camera.npz', camera_id='test',
                corneal_radius_mm=settings['radius_mm'], pupil_offset_mm=settings['pupil_offset_mm'],
                refractive_index=settings['refractive_index'], corneal_distance_bounds_mm=settings['depth_bounds_mm'],
                leds=[dict(id=str(i), position_mm=l.tolist()) for i,l in enumerate(leds)])
            (root/'config.json').write_text(json.dumps(config))
            match = dict(detections_sha256=file_sha256(data), frames={str(f):{str(i+1):str(i) for i in range(len(rays))} for f in [0,1]})
            (root/'matches.json').write_text(json.dumps(match))
            main(['--detections',str(data),'--config',str(root/'config.json'),
                  '--matches',str(root/'matches.json'),'--output-dir',str(root/'out')])
            results=[json.loads(line) for line in (root/'out'/'optical_axes.jsonl').read_text().splitlines()]
            result=results[0]
            self.assertTrue(result['valid'], result)
            self.assertFalse(results[1]['valid'])
            self.assertIsNone(results[1]['optical_axis'])
            np.testing.assert_allclose(result['corneal_center_mm'],center,atol=1e-5)
            # A human cannot accidentally reuse assignments with different data.
            data.write_text(data.read_text()+'\n')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--detections',str(data),'--config',str(root/'config.json'),
                      '--matches',str(root/'matches.json'),'--output-dir',str(root/'other')])


if __name__ == '__main__':
    unittest.main()
