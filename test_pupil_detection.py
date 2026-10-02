"""Run with: python -m unittest test_pupil_detection

Synthetic image checks test the experimental eyelid signal's contracts. They
are not a substitute for annotated recordings or a claim of blink accuracy.
No videos, model checkpoint, or GUI are needed. Tests of real pye3d geometry
run when pye3d is installed and are otherwise reported as skipped.
"""

import copy
import threading
from blink_validation import ReviewWorkspace, atomic_json, reviewer_name
import eye_pipeline as blink_pipeline
from eye_model_estimation import EyeModelEstimate
from blink_validation import (compare_records, format_comparison_report, reviewer_agreement, score_records)
import contextlib
import io
from pathlib import Path
import tempfile
import importlib.util
import json
import math
from dataclasses import FrozenInstanceError, asdict
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from eye_model_estimation import (
    EyeModelEstimator,
    ModelStabilityMonitor,
    Pye3DInputDecision,
    assess_pye3d_input_geometry,
    diagnose_model_output,
    _pye3d_ellipse,
)
from calibration import (CameraCalibration, checkerboard_object_points,
    fit_checkerboard_calibration, save_checkerboard_calibration,
    calibrate_from_recordings, main as calibration_main)
from gaze_validation import evaluate_gaze, fit_direction_rotation, main as gaze_validation_main
from pupil_detection import (
    fit_pupil_ellipse,
    EyelidObservation,
    FrameQualityDecision,
    PupilObservation,
    TemporalQualityTracker,
    assess_pupil_quality,
    detect_eyelid_closure,
)


def eye_image(lid_y=40, pupil_radius=35):
    """Create a synthetic 200-high by 320-wide grayscale eye for tests.

    lid_y is the row of the upper-lid margin (larger means farther down).
    pupil_radius is in pixels; changing it leaves the lid position fixed.
    Returns a uint8 array with brightness values between 0 and 255.
    """
    image = np.full((200, 320), 160, np.uint8)
    cv2.circle(image, (160, 110), pupil_radius, 15, -1)
    image[:lid_y] = 220
    image[lid_y:lid_y + 5] = 35
    return image


def pupil_observation(confidence=0.9, state="no_closure_evidence",
                      ellipse=((160, 110), (70, 70), 0), blink=False):
    """Make one measurement without running a camera or neural network.

    confidence is the pupil score; state is separate eyelid evidence. Passing
    state=None models an older observation with no eyelid analysis. The default
    ellipse is an ordinary, valid oval; ellipse=None means no pupil was found.
    """
    eyelid = None if state is None else EyelidObservation(state=state)
    return PupilObservation(ellipse, blink, confidence, eyelid)


class EyelidDetectionTests(unittest.TestCase):
    def test_uninformative_frames_are_unknown(self):
        for value in (0, 128, 255):
            with self.subTest(value=value):
                result = detect_eyelid_closure(np.full((200, 320), value, np.uint8))
                self.assertEqual(result.state, "unknown")
                self.assertIsNone(result.boundary)

    def test_noise_does_not_assert_closure(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                image = np.random.default_rng(seed).integers(0, 256, (200, 320), dtype=np.uint8)
                self.assertEqual(detect_eyelid_closure(image).state, "unknown")

    def test_isolated_glare_is_not_closure_evidence(self):
        image = eye_image()
        cv2.circle(image, (120, 85), 8, 255, -1)
        cv2.circle(image, (180, 150), 6, 255, -1)
        result = detect_eyelid_closure(image, ((160, 110), (70, 70), 0), 0.9)
        self.assertNotIn(result.state, ("closed_possible", "occlusion_possible"))

    def test_closed_appearance_needs_image_evidence(self):
        image = eye_image(lid_y=155)
        result = detect_eyelid_closure(image)
        self.assertEqual(result.state, "closed_possible")
        self.assertIsNotNone(result.boundary)

    def test_missing_pupil_alone_does_not_mean_closed(self):
        image = np.full((200, 320), 170, np.uint8)
        cv2.circle(image, (160, 105), 25, 10, -1)
        self.assertEqual(detect_eyelid_closure(image).state, "unknown")

    def test_pupil_size_change_does_not_create_closure(self):
        for radius in (15, 25, 40):
            with self.subTest(radius=radius):
                ellipse = ((160, 110), (2 * radius, 2 * radius), 0)
                result = detect_eyelid_closure(eye_image(pupil_radius=radius), ellipse, 0.9)
                self.assertNotIn(result.state, ("occlusion_possible", "closed_possible"))

    def test_edge_over_upper_pupil_is_occlusion_evidence(self):
        ellipse = ((160, 110), (70, 70), 0)
        result = detect_eyelid_closure(eye_image(lid_y=90), ellipse, 0.9)
        self.assertEqual(result.state, "occlusion_possible")

    def test_clipped_pupil_remains_unknown(self):
        result = detect_eyelid_closure(eye_image(lid_y=90), ((5, 110), (70, 70), 0), 0.9)
        self.assertEqual(result.state, "unknown")

    def test_coordinates_color_and_input_are_preserved(self):
        image = eye_image(lid_y=155)
        original = image.copy()
        local = detect_eyelid_closure(image)
        shifted = detect_eyelid_closure(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), roi_origin=(17, 100))
        self.assertEqual(local.state, shifted.state)
        self.assertEqual(local.support, shifted.support)
        np.testing.assert_allclose(np.array(shifted.boundary) - np.array(local.boundary),
                                   np.tile([17, 100], (len(local.boundary), 1)))
        np.testing.assert_array_equal(image, original)

    def test_no_state_is_shared_between_eyes(self):
        first = detect_eyelid_closure(eye_image(lid_y=155))
        detect_eyelid_closure(eye_image(lid_y=40))
        self.assertEqual(first, detect_eyelid_closure(eye_image(lid_y=155)))

    def test_unsupported_inputs(self):
        for image in (np.zeros((8, 8), np.uint8), np.zeros((320, 200), np.uint8)):
            self.assertEqual(detect_eyelid_closure(image).state, "unknown")
        with self.assertRaises(ValueError):
            detect_eyelid_closure(np.zeros((200, 320), np.float32))

    def test_existing_observation_constructor_still_works(self):
        result = PupilObservation(None, True, 0.0)
        self.assertIsNone(result.eyelid)


class FrameQualityTests(unittest.TestCase):
    def test_threshold_must_be_finite_and_between_zero_and_one(self):
        for threshold in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"),
                          None, "not a number", 10**400):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    assess_pupil_quality(pupil_observation(), threshold)

    def test_exact_and_custom_thresholds(self):
        # A measurement at the threshold passes; changing the configured
        # threshold must actually change the decision for an intermediate score.
        cases = ((0.60, 0.60, True), (0.599, 0.60, False),
                 (0.75, 0.80, False), (0.80, 0.80, True),
                 (0.0, 0.0, True), (1.0, 1.0, True))
        for confidence, threshold, allowed in cases:
            with self.subTest(confidence=confidence, threshold=threshold):
                result = assess_pupil_quality(pupil_observation(confidence), threshold)
                self.assertEqual(result.allow_model_update, allowed)
                self.assertEqual(result.reason, "pupil checks passed" if allowed
                                 else "low pupil confidence")

    def test_invalid_confidence_is_rejected(self):
        for confidence in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"),
                           None, "not a number", 10**400):
            with self.subTest(confidence=confidence):
                self.assertEqual(assess_pupil_quality(pupil_observation(confidence)),
                                 FrameQualityDecision(False, "invalid pupil confidence"))

    def test_missing_pupil_has_its_own_reason(self):
        # Absence of a pupil is sufficient to skip a measurement, but does not
        # establish a blink. Its reason remains distinct from eyelid closure.
        for state in (None, "unknown", "closed_possible"):
            with self.subTest(state=state):
                result = assess_pupil_quality(pupil_observation(0.0, state, ellipse=None))
                self.assertEqual(result, FrameQualityDecision(False, "no pupil"))

    def test_malformed_and_nonfinite_geometry_is_rejected(self):
        invalid_ellipses = (
            (), ((1, 2), (3, 4)), ((1,), (3, 4), 0),
            ((1, 2, 3), (3, 4), 0), ((1, 2), (3,), 0),
            ((1, 2), (3, 4, 5), 0), ((1, 2), (0, 4), 0),
            ((1, 2), (3, -4), 0), ((float("nan"), 2), (3, 4), 0),
            ((1, float("inf")), (3, 4), 0),
            ((1, 2), (float("nan"), 4), 0),
            ((1, 2), (3, float("inf")), 0),
            ((1, 2), (3, 4), float("nan")),
            ((1, 2), (3, 4), float("inf")),
            ((1, 2), (3, 4), "not an angle"), ((10**400, 2), (3, 4), 0), 42,
        )
        for ellipse in invalid_ellipses:
            with self.subTest(ellipse=ellipse):
                self.assertEqual(assess_pupil_quality(pupil_observation(ellipse=ellipse)),
                                 FrameQualityDecision(False, "invalid pupil geometry"))

    def test_each_supported_eyelid_state_has_a_distinct_decision(self):
        cases = (
            ("closed_possible", False, "possible eyelid closure"),
            ("occlusion_possible", False, "possible pupil coverage"),
            ("no_closure_evidence", True, "pupil checks passed"),
            ("unknown", True, "pupil checks passed; eyelid uncertain"),
            (None, True, "pupil checks passed; eyelid uncertain"),
            ("future_state", True, "pupil checks passed; eyelid uncertain"),
        )
        for state, allowed, reason in cases:
            with self.subTest(state=state):
                result = assess_pupil_quality(pupil_observation(state=state))
                self.assertEqual(result, FrameQualityDecision(allowed, reason))

    def test_uncertain_eyelid_does_not_bypass_existing_pupil_checks(self):
        for state in (None, "unknown", "future_state"):
            with self.subTest(state=state):
                result = assess_pupil_quality(pupil_observation(0.59, state))
                self.assertEqual(result, FrameQualityDecision(False, "low pupil confidence"))

    def test_pupil_checks_take_priority_over_eyelid_evidence(self):
        cases = (
            (pupil_observation(float("nan"), "closed_possible", ellipse=None),
             "invalid pupil confidence"),
            (pupil_observation(0.1, "closed_possible", ellipse=()),
             "invalid pupil geometry"),
            (pupil_observation(0.1, "closed_possible"), "low pupil confidence"),
        )
        for observation, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(assess_pupil_quality(observation),
                                 FrameQualityDecision(False, reason))

    def test_legacy_blink_flag_does_not_override_the_measurement(self):
        for state in (None, "no_closure_evidence", "closed_possible"):
            with self.subTest(state=state):
                self.assertEqual(assess_pupil_quality(pupil_observation(state=state, blink=True)),
                                 assess_pupil_quality(pupil_observation(state=state, blink=False)))

    def test_alternating_eyes_does_not_change_decisions_or_observations(self):
        left = pupil_observation(state="closed_possible")
        right = pupil_observation(state="unknown")
        originals = (asdict(left), asdict(right))
        expected = (FrameQualityDecision(False, "possible eyelid closure"),
                    FrameQualityDecision(True, "pupil checks passed; eyelid uncertain"))
        for observation, decision in ((left, expected[0]), (right, expected[1]),
                                      (right, expected[1]), (left, expected[0])):
            self.assertEqual(assess_pupil_quality(observation), decision)
        self.assertEqual((asdict(left), asdict(right)), originals)

    def test_decision_is_immutable(self):
        decision = assess_pupil_quality(pupil_observation())
        with self.assertRaises(FrozenInstanceError):
            decision.allow_model_update = False
        with self.assertRaises(FrozenInstanceError):
            decision.reason = "changed"


class Pye3DInputGeometryTests(unittest.TestCase):
    def test_focal_normalization_changes_domain_and_round_trips_full_ellipse(self):
        from eye_model_estimation import _projected_ellipse_to_processed

        ellipse = ((420., 205.), (30., 90.), 37.)
        converted = _pye3d_ellipse(ellipse, 640, 480, 301., 227., 500., 650.)
        restored = _projected_ellipse_to_processed(
            (converted["center"], converted["axes"], converted["angle"]),
            640, 480, 301., 227., 500., 650.)
        np.testing.assert_allclose(restored[0], ellipse[0], atol=1e-10)
        np.testing.assert_allclose(restored[1], ellipse[1], atol=1e-10)
        self.assertAlmostEqual(restored[2], ellipse[2])
        outside_after_scaling = ((600., 227.), (30., 90.), 37.)
        self.assertTrue(assess_pye3d_input_geometry(outside_after_scaling, 640, 480, 301., 227.).allow_model_update)
        self.assertFalse(assess_pye3d_input_geometry(outside_after_scaling, 640, 480, 301., 227., 500., 650.).allow_model_update)
        for fx, fy in ((0,1), (1,-1), (float("nan"),1), (1,None)):
            self.assertFalse(assess_pye3d_input_geometry(ellipse, 640,480,301.,227.,fx,fy).allow_model_update)

    """Exercise the calibrated center domain required by pye3d's binning."""

    def assess(self, ellipse):
        # An off-center principal point proves the check happens after the same
        # shift used by the pye3d adapter. Valid corrected centers are therefore
        # x in [10, 110) and y in [-10, 70) for this synthetic camera.
        return assess_pye3d_input_geometry(
            ellipse,
            width=100,
            height=80,
            principal_x=60,
            principal_y=30,
        )

    def test_half_open_shifted_image_boundaries(self):
        for center in ((10, -10), (10.001, -9.999), (109.999, 69.999)):
            with self.subTest(center=center):
                decision = self.assess((center, (12, 18), 25))
                self.assertTrue(decision.allow_model_update)
                self.assertEqual(decision.reason, "corrected pupil geometry passed")
        for center in ((9.999, 20), (110, 20), (50, -10.001), (50, 70)):
            with self.subTest(center=center):
                decision = self.assess((center, (12, 18), 25))
                self.assertFalse(decision.allow_model_update)
                self.assertEqual(
                    decision.reason,
                    "corrected pupil center outside pye3d image",
                )

    def test_malformed_corrected_geometry_is_rejected_without_mutation(self):
        valid = ((50, 20), (12, 18), 25)
        original = tuple(tuple(value) if isinstance(value, tuple) else value for value in valid)
        self.assertTrue(self.assess(valid).allow_model_update)
        self.assertEqual(valid, original)
        malformed = (
            None,
            ((50, 20), (0, 18), 25),
            ((50, 20), (-1, 18), 25),
            ((float("nan"), 20), (12, 18), 25),
            ((50, 20), (12, 18), float("inf")),
            ((10**400, 20), (12, 18), 25),
        )
        for ellipse in malformed:
            with self.subTest(ellipse=ellipse):
                decision = self.assess(ellipse)
                self.assertFalse(decision.allow_model_update)
                self.assertEqual(decision.reason, "invalid corrected pupil geometry")

    def test_invalid_camera_geometry_is_reported_separately(self):
        ellipse = ((50, 20), (12, 18), 25)
        for camera in (
            (0, 80, 60, 30),
            (100, -1, 60, 30),
            (100, 80, float("nan"), 30),
            (100, 80, 60, float("inf")),
        ):
            with self.subTest(camera=camera):
                decision = assess_pye3d_input_geometry(ellipse, *camera)
                self.assertFalse(decision.allow_model_update)
                self.assertEqual(decision.reason, "invalid pye3d camera geometry")


class TemporalQualityTests(unittest.TestCase):
    """Exercise causal decisions with video timestamps, not real-time sleeps."""

    def assert_quality(self, decision, allowed, state):
        self.assertEqual((decision.allow_model_update, decision.state), (allowed, state))

    def test_static_decision_constructor_keeps_compatible_defaults(self):
        decision = FrameQualityDecision(True, "test")
        self.assertEqual(decision.state, "single_frame")
        self.assertEqual(decision.recovery_elapsed_s, 0.0)

    def test_invalid_configuration_is_rejected(self):
        cases = [
            {"min_confidence": -0.01}, {"min_confidence": 1.01},
            {"recovery_confidence": -0.01}, {"recovery_confidence": 1.01},
            {"min_confidence": 0.8, "recovery_confidence": 0.7},
            {"recovery_duration_s": -0.01}, {"max_gap_s": 0},
            {"max_gap_s": -0.01},
        ]
        for name in ("min_confidence", "recovery_confidence",
                     "recovery_duration_s", "max_gap_s"):
            cases.extend({name: value} for value in
                         (float("nan"), float("inf"), None, "bad", 10**400))
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TemporalQualityTracker(**kwargs)

    def test_first_good_frame_is_usable_even_with_uncertain_eyelids(self):
        for state in (None, "unknown", "no_closure_evidence", "future_state"):
            with self.subTest(state=state):
                result = TemporalQualityTracker().update(pupil_observation(0.60, state), 0)
                self.assert_quality(result, True, "usable")

    def test_every_static_rejection_interrupts_a_usable_sequence_immediately(self):
        observations = (
            pupil_observation(0.0, ellipse=None),
            pupil_observation(float("nan")),
            pupil_observation(ellipse=()),
            pupil_observation(0.599),
            pupil_observation(state="closed_possible"),
            pupil_observation(state="occlusion_possible"),
        )
        for observation in observations:
            with self.subTest(reason=assess_pupil_quality(observation).reason):
                tracker = TemporalQualityTracker()
                self.assert_quality(tracker.update(pupil_observation(), 0), True, "usable")
                rejected = tracker.update(observation, 0.02)
                self.assert_quality(rejected, False, "rejected")
                self.assertEqual(rejected.reason, assess_pupil_quality(observation).reason)
                self.assertEqual(rejected.recovery_elapsed_s, 0.0)
                self.assert_quality(tracker.update(pupil_observation(), 0.04),
                                    False, "recovering")

    def test_recovery_at_30_fps_needs_observed_elapsed_time(self):
        tracker = TemporalQualityTracker()
        tracker.update(pupil_observation(ellipse=None), 0)
        first = tracker.update(pupil_observation(0.70), 1 / 30)
        second = tracker.update(pupil_observation(0.70), 2 / 30)
        third = tracker.update(pupil_observation(0.70), 3 / 30)
        self.assert_quality(first, False, "recovering")
        self.assertEqual(first.recovery_elapsed_s, 0.0)
        self.assert_quality(second, False, "recovering")
        self.assertAlmostEqual(second.recovery_elapsed_s, 1 / 30)
        # Three strong observations span two frame intervals: 66.7 ms.
        self.assert_quality(third, True, "usable")

    def test_weaker_scores_reset_recovery_but_are_allowed_after_recovery(self):
        tracker = TemporalQualityTracker()
        tracker.update(pupil_observation(ellipse=None), 0)
        for timestamp in (0.02, 0.04):
            self.assert_quality(tracker.update(pupil_observation(0.70), timestamp),
                                False, "recovering")
        weak = tracker.update(pupil_observation(0.69), 0.06)
        self.assert_quality(weak, False, "recovering")
        self.assertEqual(weak.recovery_elapsed_s, 0.0)
        restarted = tracker.update(pupil_observation(0.70), 0.08)
        self.assertEqual(restarted.recovery_elapsed_s, 0.0)
        for timestamp in (0.10, 0.12):
            self.assert_quality(tracker.update(pupil_observation(0.70), timestamp),
                                False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(0.70), 0.14), True, "usable")
        # The higher threshold is for recovery only; steady tracking uses 0.60.
        self.assert_quality(tracker.update(pupil_observation(0.60), 0.16), True, "usable")

    def test_bad_frame_resets_an_incomplete_recovery(self):
        tracker = TemporalQualityTracker()
        tracker.update(pupil_observation(ellipse=None), 0)
        tracker.update(pupil_observation(), 0.02)
        tracker.update(pupil_observation(), 0.04)
        self.assert_quality(tracker.update(pupil_observation(state="occlusion_possible"), 0.06),
                            False, "rejected")
        for timestamp in (0.08, 0.10, 0.12):
            self.assert_quality(tracker.update(pupil_observation(), timestamp),
                                False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(), 0.14), True, "usable")

    def test_gap_discards_evidence_from_usable_and_recovering_states(self):
        prefixes = (
            ((pupil_observation(), 0),),
            ((pupil_observation(ellipse=None), 0), (pupil_observation(), 0.02)),
        )
        for prefix in prefixes:
            with self.subTest(prefix_length=len(prefix)):
                tracker = TemporalQualityTracker()
                for observation, timestamp in prefix:
                    tracker.update(observation, timestamp)
                first = tracker.update(pupil_observation(), 0.20)
                self.assert_quality(first, False, "recovering")
                self.assertEqual(first.recovery_elapsed_s, 0.0)
                for timestamp in (0.22, 0.24):
                    self.assert_quality(tracker.update(pupil_observation(), timestamp),
                                        False, "recovering")
                self.assert_quality(tracker.update(pupil_observation(), 0.26), True, "usable")

    def test_exact_gap_and_recovery_boundaries(self):
        # Binary-exact fractions isolate inclusive/exclusive boundary behavior.
        tracker = TemporalQualityTracker(recovery_duration_s=0.125, max_gap_s=0.0625)
        tracker.update(pupil_observation(ellipse=None), 0)
        self.assert_quality(tracker.update(pupil_observation(), 0.0625), False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(), 0.125), False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(), 0.1875), True, "usable")
        self.assert_quality(tracker.update(pupil_observation(), 0.2501), False, "recovering")

    def test_zero_recovery_duration_allows_first_strong_observation(self):
        tracker = TemporalQualityTracker(recovery_duration_s=0)
        tracker.update(pupil_observation(ellipse=None), 0)
        self.assert_quality(tracker.update(pupil_observation(0.69), 0.02), False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(0.70), 0.04), True, "usable")
        self.assert_quality(tracker.update(pupil_observation(0.70), 0.20), True, "usable")

    def test_invalid_timestamps_raise_without_changing_subsequent_decisions(self):
        invalid_times = (-1, 0, 1.01, 1.02, float("nan"), float("inf"),
                         -float("inf"), None, "not a timestamp", 10**400)
        for timestamp in invalid_times:
            with self.subTest(timestamp=timestamp):
                tracker, control = TemporalQualityTracker(), TemporalQualityTracker()
                for current in (tracker, control):
                    current.update(pupil_observation(ellipse=None), 1.0)
                    current.update(pupil_observation(), 1.02)
                # A bad observation would reset recovery if timestamp validation
                # happened too late. Compare future behavior against an untouched tracker.
                with self.assertRaises(ValueError):
                    tracker.update(pupil_observation(ellipse=None), timestamp)
                for valid_time in (1.04, 1.06, 1.08):
                    self.assertEqual(tracker.update(pupil_observation(), valid_time),
                                     control.update(pupil_observation(), valid_time))

    def test_reset_allows_a_new_recording_to_start_at_zero(self):
        tracker = TemporalQualityTracker()
        tracker.update(pupil_observation(ellipse=None), 10)
        tracker.update(pupil_observation(), 10.02)
        tracker.reset()
        result = tracker.update(pupil_observation(0.65, "unknown"), 0)
        self.assert_quality(result, True, "usable")
        self.assertEqual(result.recovery_elapsed_s, 0.0)

    def test_timestamp_origin_does_not_change_state_transitions(self):
        sequence = [pupil_observation(ellipse=None)] + [pupil_observation()] * 5
        expected = None
        for origin in (0, 1000, 1_000_000):
            tracker = TemporalQualityTracker(recovery_duration_s=0.06)
            decisions = [tracker.update(observation, origin + index * 0.02)
                         for index, observation in enumerate(sequence)]
            transitions = [(result.allow_model_update, result.state) for result in decisions]
            if expected is None:
                expected = transitions
            self.assertEqual(transitions, expected)

    def test_future_frames_do_not_change_prefix_decisions(self):
        sequence = [pupil_observation(), pupil_observation(ellipse=None),
                    pupil_observation(), pupil_observation(0.65),
                    pupil_observation(), pupil_observation(), pupil_observation(),
                    pupil_observation(), pupil_observation(state="closed_possible")]
        tracker = TemporalQualityTracker()
        results, snapshots = [], []
        for index, observation in enumerate(sequence):
            decision = tracker.update(observation, index * 0.02)
            results.append(decision)
            snapshots.append(asdict(decision))
        self.assertEqual([asdict(result) for result in results], snapshots)
        for length in range(1, len(sequence) + 1):
            prefix_tracker = TemporalQualityTracker()
            prefix = [prefix_tracker.update(observation, index * 0.02)
                      for index, observation in enumerate(sequence[:length])]
            self.assertEqual(prefix, results[:length])

    def test_two_eye_instances_have_independent_histories(self):
        left = [pupil_observation(), pupil_observation(ellipse=None)] + [pupil_observation()] * 4
        right = [pupil_observation(0.65, "unknown")] * len(left)
        originals = [[asdict(observation) for observation in eye] for eye in (left, right)]
        independent = []
        for eye in (left, right):
            tracker = TemporalQualityTracker()
            independent.append([tracker.update(observation, index * 0.02)
                                for index, observation in enumerate(eye)])
        trackers = (TemporalQualityTracker(), TemporalQualityTracker())
        for index, pair in enumerate(zip(left, right)):
            for side, observation in enumerate(pair):
                self.assertEqual(trackers[side].update(observation, index * 0.02),
                                 independent[side][index])
        self.assertEqual([[asdict(observation) for observation in eye] for eye in (left, right)],
                         originals)

    def test_temporal_acceptance_never_overrides_static_rejection(self):
        tracker = TemporalQualityTracker()
        sequence = ([pupil_observation()] * 4 + [pupil_observation(0.59)]
                    + [pupil_observation()] * 4
                    + [pupil_observation(state="occlusion_possible"),
                       pupil_observation(ellipse=None), pupil_observation(float("nan"))]
                    + [pupil_observation(state="unknown")] * 4)
        for index, observation in enumerate(sequence):
            static = assess_pupil_quality(observation)
            temporal = tracker.update(observation, index * 0.02)
            if temporal.allow_model_update:
                self.assertTrue(static.allow_model_update)
            if not static.allow_model_update:
                self.assert_quality(temporal, False, "rejected")

    def test_post_calibration_rejection_uses_normal_recovery(self):
        tracker = TemporalQualityTracker(recovery_duration_s=0.05)
        rejected = tracker.update(
            pupil_observation(),
            0.0,
            additional_rejection_reason="corrected pupil center outside pye3d image",
        )
        self.assert_quality(rejected, False, "rejected")
        self.assertEqual(
            rejected.reason,
            "corrected pupil center outside pye3d image",
        )
        self.assert_quality(tracker.update(pupil_observation(), 1 / 30),
                            False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(), 2 / 30),
                            False, "recovering")
        self.assert_quality(tracker.update(pupil_observation(), 3 / 30),
                            True, "usable")

    def test_image_level_failure_keeps_reason_priority(self):
        tracker = TemporalQualityTracker()
        decision = tracker.update(
            pupil_observation(ellipse=None),
            0.0,
            additional_rejection_reason="corrected pupil center outside pye3d image",
        )
        self.assert_quality(decision, False, "rejected")
        self.assertEqual(decision.reason, "no pupil")

    def test_invalid_additional_reason_does_not_change_history(self):
        invalid_reasons = ("", "   ", 5, False, object())
        for reason in invalid_reasons:
            with self.subTest(reason=reason):
                tracker, control = TemporalQualityTracker(), TemporalQualityTracker()
                with self.assertRaises(ValueError):
                    tracker.update(
                        pupil_observation(ellipse=None),
                        0.0,
                        additional_rejection_reason=reason,
                    )
                self.assertEqual(
                    tracker.update(pupil_observation(), 0.0),
                    control.update(pupil_observation(), 0.0),
                )


class EyeModelSkipTests(unittest.TestCase):
    def make_estimator_without_pye3d(self):
        """Build the adapter around a spy so these tests need no pye3d install."""
        estimator = object.__new__(EyeModelEstimator)
        estimator.width = estimator.height = 32
        estimator.fx = estimator.fy = 20.0
        estimator.cx = estimator.cy = 16.0
        estimator.min_confidence = 0.60
        estimator.scale = 1.0
        estimator.last_timestamp = 0.0
        estimator.calibration = Mock(
            distort_ellipse=Mock(side_effect=lambda value: value),
            distort_points=Mock(side_effect=lambda value: np.asarray(value)),
        )
        estimator.detector = Mock(update_and_detect=Mock(return_value=None))
        return estimator

    def test_skip_is_empty_and_preserves_model_history(self):
        estimator = self.make_estimator_without_pye3d()
        detector, timestamp = estimator.detector, estimator.last_timestamp
        result = estimator.skip_update(0.91, "rejected", "possible pupil coverage")
        self.assertFalse(result.ready)
        self.assertEqual(result.status,
                         "quality rejected: possible pupil coverage")
        self.assertEqual(result.pupil_confidence, 0.91)
        self.assertEqual(result.update_time_ms, 0.0)
        for name in ("eye_center_mm", "pupil_center_mm", "pupil_diameter_mm",
                     "model_confidence", "projected_eye_sphere",
                     "projected_eye_center", "projected_pupil_center"):
            self.assertIsNone(getattr(result, name), name)
        detector.update_and_detect.assert_not_called()
        self.assertEqual(estimator.last_timestamp, timestamp)

    def test_invalid_rejected_confidence_is_logged_as_nan_without_crashing(self):
        estimator = self.make_estimator_without_pye3d()
        for confidence in (None, "bad", float("nan"), float("inf"), 10**400):
            with self.subTest(confidence=confidence):
                result = estimator.skip_update(confidence, "rejected",
                                               "invalid pupil confidence")
                self.assertTrue(np.isnan(result.pupil_confidence))
                estimator.detector.update_and_detect.assert_not_called()

    def test_next_accepted_measurement_uses_its_real_later_timestamp(self):
        estimator = self.make_estimator_without_pye3d()
        estimator.skip_update(0.9, "rejected", "test rejection")
        frame = np.zeros((32, 32), np.uint8)
        ellipse = ((16, 16), (12, 18), 0)
        result = estimator.update(ellipse, 0.9, 0.25, frame)
        self.assertFalse(result.ready)  # The spy returns None: pye3d is warming.
        self.assertEqual(estimator.last_timestamp, 0.25)
        datum = estimator.detector.update_and_detect.call_args.args[0]
        self.assertEqual(datum["timestamp"], 0.25)

    def test_direct_update_refuses_out_of_bounds_corrected_center(self):
        estimator = self.make_estimator_without_pye3d()
        frame = np.zeros((32, 32), np.uint8)
        # With a centered principal point, this remains x=-0.01 in pye3d input.
        ellipse = ((-0.01, 16), (12, 18), 0)
        decision = estimator.assess_input_geometry(ellipse)
        self.assertFalse(decision.allow_model_update)
        self.assertEqual(
            decision.reason,
            "corrected pupil center outside pye3d image",
        )
        with self.assertRaisesRegex(ValueError, "pye3d input rejected"):
            estimator.update(ellipse, 0.9, 0.25, frame)
        estimator.detector.update_and_detect.assert_not_called()
        self.assertEqual(estimator.last_timestamp, 0.0)


class QualityGateIntegrationTests(unittest.TestCase):
    def test_cli_preserves_configured_headless_export_and_identity_policy(self):
        import eye_pipeline as pipeline
        with (patch.multiple(pipeline, HEADLESS=True, RESULTS_PATH=Path("configured.jsonl"),
                             REQUIRE_CALIBRATION_IDENTITY=True),
              patch("sys.argv", ["eye_pipeline.py"]),
              patch.object(pipeline, "run_pipeline") as run):
            pipeline.main()
            run.assert_called_once_with()
            self.assertTrue(pipeline.HEADLESS)
            self.assertTrue(pipeline.REQUIRE_CALIBRATION_IDENTITY)
            self.assertEqual(pipeline.RESULTS_PATH, Path("configured.jsonl"))

    def test_headless_run_rejects_undecodable_pair_and_releases_captures(self):
        import eye_pipeline as pipeline
        videos = [Mock(read=Mock(return_value=(False, None))) for _ in range(2)]
        calibrations = [Mock(identity_status="matched") for _ in range(2)]
        with (patch.multiple(pipeline, HEADLESS=True, RESULTS_PATH=None,
                             LEFT_VIDEO_PATH="left.mp4", RIGHT_VIDEO_PATH="right.mp4"),
              patch.object(pipeline, "select_eye_video", side_effect=lambda path, side:path),
              patch.object(pipeline, "open_video", side_effect=videos),
              patch.object(pipeline, "video_dimensions", return_value=(32,32)),
              patch.object(pipeline, "matching_video_fps", return_value=30),
              patch.object(pipeline.CameraCalibration, "load", side_effect=calibrations),
              patch.object(pipeline, "load_pupil_detector"),
              patch.object(pipeline, "EyeModelEstimator"),
              self.assertRaisesRegex(ValueError, "no paired video frames")):
            pipeline.run_pipeline()
        for video in videos:
            video.release.assert_called_once_with()

    def test_headless_export_keeps_learning_through_drift_and_records_skips(self):
        import eye_pipeline as pipeline
        from feature_output import frame_record

        count = 28
        frame = np.zeros((32, 32, 3), np.uint8)
        ellipse = ((16, 16), (12, 18), 0)
        observations = []
        for index in range(count):
            observations.extend((
                pupil_observation(state="occlusion_possible" if index == 12 else "unknown", ellipse=ellipse),
                pupil_observation(state="unknown", ellipse=ellipse),
            ))
        videos = [Mock(read=Mock(side_effect=[(True, frame)]*count + [(False, None)])) for _ in range(2)]
        calibrations = [Mock(undistort_ellipse=Mock(side_effect=lambda value: value)) for _ in range(2)]
        models = [diagnostic_estimator() for _ in range(2)]
        outputs = []
        for index in range(count):
            raw = diagnostic_raw_result()
            raw["sphere"]["center"] = [0. if index < 5 else 8., 0., 40.]
            outputs.append(raw)
        models[1].detector.update_and_detect.side_effect = outputs
        stream = io.StringIO()
        metadata = {eye: dict(schema_version=1, recording_id="synthetic_pipeline",
                    camera_id="synthetic-"+eye, coordinate_system="processed_eye_camera",
                    calibration_sha256="a"*64, frame_rotation="none", calibration_rotation="none")
                    for eye in ("left", "right")}
        with (patch.multiple(pipeline, MAX_FRAMES=0, TEXT_OUTPUT=False,
                             LEFT_ROI=(0,0,32,32), RIGHT_ROI=(0,0,32,32)),
              patch.object(pipeline, "ModelStabilityMonitor", side_effect=lambda:
                           ModelStabilityMonitor(startup_seconds=0, window_seconds=.1, min_samples=2)),
              patch.object(pipeline.cv2, "namedWindow") as window,
              patch.object(pipeline, "create_output_frame") as render):
            processed = pipeline.process_frame_loop(*videos, 30, "none", "none", *calibrations,
                Mock(detect=Mock(side_effect=observations)), *models, headless=True,
                record_stream=stream, record_metadata=metadata)
        self.assertEqual(processed, count)
        window.assert_not_called()
        render.assert_not_called()
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(records), 2*count)
        for index, row in enumerate(records):
            self.assertEqual(row["pupil_observation"]["ellipse"], [[16, 16], [12, 18], 0])
            self.assertEqual(row["pupil_observation_coordinate_system"], "rotated_raw_frame_pixels")
            self.assertEqual(row["pupil_observation"]["eyelid"]["state"],
                             observations[index].eyelid.state)
        self.assertEqual(models[1].detector.update_and_detect.call_count, count)
        self.assertTrue(any(row["model_stability"]["status"] == "drift_detected" for row in records[1::2]))
        skipped = [row for row in records if row["model_input"] == "skipped"]
        self.assertEqual(len(skipped), 3)
        for row in skipped:
            self.assertEqual(row["eye"], "left")
            self.assertFalse(row["ready"])
            self.assertIsNone(row["gaze_direction_camera"])
            self.assertIsNone(row["model_diagnostics"])
            self.assertIsNone(row["model_stability"]["recent_center_mm"])
        self.assertEqual(models[0].detector.update_and_detect.call_count, count - len(skipped))
        self.assertEqual(records[-1]["timestamp_s"], (count-1)/30)
        # The actual exporter feeds the target evaluator, not a hand-built
        # substitute. Matching synthetic references should give zero error.
        target = dict(schema_version=1, data_kind="synthetic", direction_source="known fixture normal",
                      recording_id="synthetic_pipeline", coordinate_system="processed_eye_camera",
                      camera_ids={"right":"synthetic-right"}, calibration_sha256={"right":"a"*64},
                      frame_rotation={"right":"none"}, calibration_rotation={"right":"none"},
                      samples=[dict(eye="right", frame_index=index, timestamp_s=index/30,
                                    direction_camera=[0,0,-1], split="validation") for index in range(count)])
        evaluation = evaluate_gaze(records, target)
        self.assertEqual(evaluation["raw_angular_error"]["max_deg"], 0)
        # Invalid optional confidence on a rejected frame is JSON null.
        estimate = models[0].skip_update(float("nan"), "rejected", "invalid confidence")
        record = frame_record("left", 0, 0., estimate, assess_pupil_quality(observations[24]), metadata["left"])
        self.assertIsNone(record["pupil_confidence"])
        json.dumps(record, allow_nan=False)

    def test_rejected_eye_is_skipped_while_other_eye_updates(self):
        # Replace camera/model/window boundaries, then exercise the real paired
        # frame loop. This checks gating without pye3d, a camera, or a GUI.
        import eye_pipeline as pipeline

        frame = np.zeros((32, 32, 3), np.uint8)
        ellipse = ((16, 16), (12, 18), 0)
        left = pupil_observation(state="occlusion_possible", ellipse=ellipse)
        right = pupil_observation(state="unknown", ellipse=ellipse)
        detector = Mock(detect=Mock(side_effect=[left, right]))
        left_video = Mock(read=Mock(return_value=(True, frame)))
        right_video = Mock(read=Mock(return_value=(True, frame)))
        left_calibration = Mock(undistort_ellipse=Mock(side_effect=lambda value: value))
        right_calibration = Mock(undistort_ellipse=Mock(side_effect=lambda value: value))
        left_model, right_model = Mock(), Mock()
        for model in (left_model, right_model):
            model.update.return_value = diagnostic_estimator()._result(diagnostic_raw_result(), .9, .1)
            model.skip_update.return_value = diagnostic_estimator()._empty(.9, "skipped")
        right_model.assess_input_geometry.return_value = Pye3DInputDecision(
            True,
            "corrected pupil geometry passed",
        )
        with (patch.multiple(pipeline, MAX_FRAMES=1, TEXT_OUTPUT=False,
                             LEFT_ROI=(0, 0, 32, 32), RIGHT_ROI=(0, 0, 32, 32),
                             MIN_CONFIDENCE=0.60),
              patch.object(pipeline, "create_output_frame", return_value=frame) as output,
              patch.object(pipeline.cv2, "namedWindow"),
              patch.object(pipeline.cv2, "imshow"),
              patch.object(pipeline.cv2, "waitKey", return_value=-1)):
            pipeline.process_frame_loop(
                left_video, right_video, 30, "none", "none", left_calibration,
                right_calibration, detector, left_model, right_model,
            )
        decisions = [call.kwargs["quality_decision"] for call in output.call_args_list]
        self.assertFalse(decisions[0].allow_model_update)
        self.assertTrue(decisions[1].allow_model_update)
        self.assertEqual([decision.state for decision in decisions], ["rejected", "usable"])
        left_calibration.undistort_ellipse.assert_not_called()
        left_model.assess_input_geometry.assert_not_called()
        left_model.update.assert_not_called()
        left_model.skip_update.assert_called_once_with(
            left.confidence, "rejected", "possible pupil coverage")
        right_calibration.undistort_ellipse.assert_called_once_with(right.ellipse)
        right_model.assess_input_geometry.assert_called_once_with(right.ellipse)
        right_model.skip_update.assert_not_called()
        right_model.update.assert_called_once_with(
            right.ellipse, right.confidence, 0.0, frame)

    def test_paired_loop_keeps_recovery_independent_and_enforces_gate(self):
        import eye_pipeline as pipeline

        left_frames = [np.full((32, 32, 3), index, np.uint8) for index in range(4)]
        right_frames = [np.full((32, 32, 3), index + 10, np.uint8) for index in range(4)]
        left = [pupil_observation(state="occlusion_possible")] + [pupil_observation()] * 3
        right = [pupil_observation(0.65, "unknown")] * 4
        detector = Mock(detect=Mock(side_effect=[item for pair in zip(left, right) for item in pair]))
        videos = [Mock(read=Mock(side_effect=[(True, frame) for frame in frames]))
                  for frames in (left_frames, right_frames)]
        calibrations = tuple(
            Mock(undistort_ellipse=Mock(side_effect=lambda value: value))
            for _ in range(2)
        )
        models = (Mock(), Mock())
        for model in models:
            model.update.return_value = diagnostic_estimator()._result(diagnostic_raw_result(), .9, .1)
            model.skip_update.return_value = diagnostic_estimator()._empty(.9, "skipped")
            model.assess_input_geometry.return_value = Pye3DInputDecision(
                True,
                "corrected pupil geometry passed",
            )
        with (patch.multiple(pipeline, MAX_FRAMES=4, TEXT_OUTPUT=False,
                             LEFT_ROI=(0, 0, 32, 32), RIGHT_ROI=(0, 0, 32, 32),
                             MIN_CONFIDENCE=0.60),
              patch.object(pipeline, "create_output_frame", return_value=left_frames[0]) as output,
              patch.object(pipeline.cv2, "namedWindow"),
              patch.object(pipeline.cv2, "imshow"),
              patch.object(pipeline.cv2, "waitKey", return_value=-1)):
            pipeline.process_frame_loop(*videos, 30, "none", "none", *calibrations,
                                        detector, *models)
        decisions = [call.kwargs["quality_decision"] for call in output.call_args_list]
        self.assertEqual([decision.state for decision in decisions[::2]],
                         ["rejected", "recovering", "recovering", "usable"])
        self.assertTrue(all(decision.allow_model_update for decision in decisions[1::2]))
        # Left rejects once, bypasses geometry during recovery, then
        # submits only the real fourth observation. Right remains independent.
        self.assertEqual(models[0].skip_update.call_count, 3)
        models[0].update.assert_called_once_with(
            left[3].ellipse, left[3].confidence, 3 / 30, left_frames[3])
        self.assertEqual(calibrations[0].undistort_ellipse.call_count, 1)
        self.assertEqual(models[0].assess_input_geometry.call_count, 1)
        models[1].skip_update.assert_not_called()
        self.assertEqual(models[1].update.call_count, 4)
        self.assertEqual(calibrations[1].undistort_ellipse.call_count, 4)
        self.assertEqual(models[1].assess_input_geometry.call_count, 4)
        for index, call in enumerate(models[1].update.call_args_list):
            self.assertEqual(call.args[:3],
                             (right[index].ellipse, right[index].confidence, index / 30))
            self.assertIs(call.args[3], right_frames[index])

    def test_post_calibration_failure_is_a_real_skip_with_recovery(self):
        import eye_pipeline as pipeline

        frames = [np.full((32, 32, 3), index, np.uint8) for index in range(4)]
        pupils = [pupil_observation()] * 8
        detector = Mock(detect=Mock(side_effect=pupils))
        videos = tuple(
            Mock(read=Mock(side_effect=[(True, frame) for frame in frames]))
            for _ in range(2)
        )
        calibrations = tuple(
            Mock(undistort_ellipse=Mock(side_effect=lambda value: value))
            for _ in range(2)
        )
        invalid = Pye3DInputDecision(
            False,
            "corrected pupil center outside pye3d image",
        )
        valid = Pye3DInputDecision(True, "corrected pupil geometry passed")
        left_model = Mock()
        left_model.assess_input_geometry.side_effect = [invalid, valid, valid, valid]
        right_model = Mock()
        right_model.assess_input_geometry.return_value = valid
        for model in (left_model, right_model):
            model.update.return_value = diagnostic_estimator()._result(diagnostic_raw_result(), .9, .1)
            model.skip_update.return_value = diagnostic_estimator()._empty(.9, "skipped")
        with (patch.multiple(pipeline, MAX_FRAMES=4, TEXT_OUTPUT=False,
                             LEFT_ROI=(0, 0, 32, 32), RIGHT_ROI=(0, 0, 32, 32),
                             MIN_CONFIDENCE=0.60),
              patch.object(pipeline, "create_output_frame", return_value=frames[0]) as output,
              patch.object(pipeline.cv2, "namedWindow"),
              patch.object(pipeline.cv2, "imshow"),
              patch.object(pipeline.cv2, "waitKey", return_value=-1)):
            pipeline.process_frame_loop(
                *videos,
                30,
                "none",
                "none",
                *calibrations,
                detector,
                left_model,
                right_model,
            )

        left_decisions = [
            call.kwargs["quality_decision"]
            for call in output.call_args_list[::2]
        ]
        self.assertEqual(
            [decision.state for decision in left_decisions],
            ["rejected", "recovering", "recovering", "usable"],
        )
        self.assertEqual(
            left_decisions[0].reason,
            "corrected pupil center outside pye3d image",
        )
        left_model.skip_update.assert_any_call(
            0.9,
            "rejected",
            "corrected pupil center outside pye3d image",
        )
        self.assertEqual(left_model.skip_update.call_count, 3)
        left_model.update.assert_called_once_with(
            pupils[6].ellipse,
            pupils[6].confidence,
            3 / 30,
            frames[3],
        )
        self.assertEqual(right_model.update.call_count, 4)
        right_model.skip_update.assert_not_called()

    def test_overlay_explains_enforced_action_without_modifying_input(self):
        import feature_output

        frame = np.zeros((220, 320, 3), np.uint8)
        original = frame.copy()
        observation = pupil_observation(state="occlusion_possible")
        decision = assess_pupil_quality(observation)
        estimate = SimpleNamespace(ready=False, status="test estimate",
                                   pupil_confidence=0.9, model_confidence=None)
        with patch.object(feature_output, "_draw_text") as draw_text:
            rendered = feature_output.create_output_frame(
                frame, "Left eye", (0, 0, 320, 200), observation, estimate,
                quality_decision=decision,
            )
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("model input: skipped: possible pupil coverage", texts)
        self.assertIsNot(rendered, frame)
        np.testing.assert_array_equal(frame, original)

    def test_console_log_includes_timestamp_action_state_and_reason(self):
        import feature_output

        decision = FrameQualityDecision(
            False, "possible pupil coverage", state="rejected")
        estimate = SimpleNamespace(
            ready=False, eye_center_mm=None, pupil_center_mm=None,
            pupil_diameter_mm=None, pupil_confidence=0.9,
            model_confidence=None, update_time_ms=0.0,
            status="quality rejected: possible pupil coverage",
        )
        with patch("builtins.print") as print_spy:
            feature_output.print_frame_features("left", 1.25, estimate, decision)
        line = print_spy.call_args.args[0]
        self.assertIn("left t=1.250s", line)
        self.assertIn("model_input=skipped", line)
        self.assertIn("quality_state=rejected", line)
        self.assertIn("quality_reason='possible pupil coverage'", line)


def diagnostic_raw_result():
    """An ordinary in-range pye3d-shaped dictionary in native millimeters."""
    return {
        "sphere": {"center": (0.0, 0.0, 40.0)},
        "circle_3d": {"center": (0.0, 0.0, 30.0), "normal": (0.0, 0.0, -1.0)},
        "diameter_3d": 4.0,
        "phi": -math.pi / 2.0,
        "theta": math.pi / 2.0,
        "model_confidence": 1.0,
    }


def diagnostic_estimator(scale=1.0):
    """Use a detector spy to isolate the project's adapter and logging contract."""
    estimator = object.__new__(EyeModelEstimator)
    estimator.width = estimator.height = 32
    estimator.fx = estimator.fy = 20.0
    estimator.cx = estimator.cy = 16.0
    estimator.min_confidence = 0.60
    estimator.scale = scale
    estimator.last_timestamp = None
    estimator.calibration = Mock(
        identity_status="unverified",
        distort_ellipse=Mock(side_effect=lambda value: value),
        distort_points=Mock(side_effect=lambda value: np.asarray(value)),
    )
    estimator.detector = Mock(update_and_detect=Mock(return_value=diagnostic_raw_result()))
    return estimator


class ModelOutputDiagnosticsTests(unittest.TestCase):
    def test_all_upstream_bounds_are_inclusive(self):
        # These are the exact native ranges in Detector3D._prepare_result.
        cases = (
            ("eye_center_x", -15.0, 15.0),
            ("eye_center_y", -10.0, 10.0),
            ("eye_center_z", 15.0, 75.0),
            ("pupil_diameter", 1.0, 9.0),
            ("gaze_phi", -90.0, 90.0),
            ("gaze_theta", -80.0, 80.0),
        )
        for name, lower, upper in cases:
            for value, expected in ((lower, "passed"), (upper, "passed"),
                                    (lower - 1e-6, "failed"),
                                    (upper + 1e-6, "failed")):
                with self.subTest(check=name, value=value):
                    raw = diagnostic_raw_result()
                    if name.startswith("eye_center_"):
                        center = list(raw["sphere"]["center"])
                        center["xyz".index(name[-1])] = value
                        raw["sphere"]["center"] = center
                    elif name == "pupil_diameter":
                        raw["diameter_3d"] = value
                    elif name == "gaze_phi":
                        raw["phi"] = math.radians(value - 90.0)
                    else:
                        raw["theta"] = math.radians(value + 90.0)
                    result = diagnose_model_output(raw)
                    self.assertEqual(result.range_status, expected)
                    self.assertEqual(result.unavailable_checks, ())
                    self.assertEqual(result.failed_checks, () if expected == "passed" else (name,))

    def test_empty_output_is_unavailable_not_a_pass(self):
        result = diagnose_model_output({})
        self.assertEqual(result.range_status, "unavailable")
        self.assertEqual(result.failed_checks, ())
        self.assertEqual(set(result.unavailable_checks), {
            "eye_center_x", "eye_center_y", "eye_center_z", "pupil_diameter",
            "pupil_normal", "gaze_phi", "gaze_theta",
        })

    def test_malformed_native_geometry_is_unavailable_and_json_safe(self):
        cases = (
            ("sphere", None), ("sphere", []), ("sphere", "invalid"),
            ("sphere", {"center": (0.0, 40.0)}),
            ("sphere", {"center": (0.0, float("nan"), 40.0)}),
            ("sphere", {"center": (0.0, float("inf"), 40.0)}),
            ("sphere", {"center": (0.0, 10**400, 40.0)}),
            ("diameter_3d", None), ("diameter_3d", "invalid"),
            ("diameter_3d", float("nan")), ("diameter_3d", float("inf")),
            ("diameter_3d", 10**400),
        )
        for key, value in cases:
            with self.subTest(field=key, value=value):
                raw = diagnostic_raw_result()
                raw[key] = value
                result = diagnose_model_output(raw)
                self.assertEqual(result.range_status, "unavailable")
                self.assertEqual(result.failed_checks, ())
                # Persisted diagnostics must not leak JSON NaN/Infinity literals.
                json.dumps(asdict(result), allow_nan=False)

    def test_zero_or_nonfinite_normal_does_not_validate_placeholder_angles(self):
        cases = (None, {}, [], "invalid", {"normal": (0.0, 0.0, 0.0)},
                 {"normal": (0.0, float("nan"), 1.0)},
                 {"normal": (0.0, float("inf"), 1.0)},
                 {"normal": (0.0, 1.0)}, {"normal": (0.0, 10**400, 1.0)})
        for circle in cases:
            with self.subTest(circle=circle):
                raw = diagnostic_raw_result()
                raw["circle_3d"] = circle
                raw["phi"] = raw["theta"] = 0.0
                result = diagnose_model_output(raw)
                self.assertEqual(result.range_status, "unavailable")
                self.assertEqual(result.failed_checks, ())
                self.assertEqual(set(result.unavailable_checks),
                                 {"pupil_normal", "gaze_phi", "gaze_theta"})
                self.assertIsNone(result.phi_offset_deg)
                self.assertIsNone(result.theta_offset_deg)
                json.dumps(asdict(result), allow_nan=False)

    def test_missing_or_nonfinite_angles_are_unavailable_individually(self):
        for field, name in (("phi", "gaze_phi"), ("theta", "gaze_theta")):
            for value in (None, "invalid", float("nan"), float("inf"), 10**400):
                with self.subTest(field=field, value=value):
                    raw = diagnostic_raw_result()
                    raw[field] = value
                    result = diagnose_model_output(raw)
                    self.assertEqual(result.range_status, "unavailable")
                    self.assertEqual(result.unavailable_checks, (name,))
                    self.assertEqual(result.failed_checks, ())
                    json.dumps(asdict(result), allow_nan=False)

    def test_failure_and_unavailability_are_both_preserved(self):
        raw = diagnostic_raw_result()
        raw["sphere"]["center"] = (0.0, 11.0, 40.0)
        del raw["theta"]
        result = diagnose_model_output(raw)
        self.assertEqual(result.range_status, "failed")
        self.assertEqual(result.failed_checks, ("eye_center_y",))
        self.assertEqual(result.unavailable_checks, ("gaze_theta",))

    def test_diagnostics_do_not_mutate_raw_result(self):
        raw = diagnostic_raw_result()
        original = copy.deepcopy(raw)
        diagnose_model_output(raw)
        self.assertEqual(raw, original)

    def test_ranges_are_checked_before_configured_eye_radius_scaling(self):
        for scale, native_y, expected_status in ((2.0, 8.0, "passed"), (0.5, 12.0, "failed")):
            with self.subTest(scale=scale, native_y=native_y):
                raw = diagnostic_raw_result()
                raw["sphere"]["center"] = (0.0, native_y, 40.0)
                result = diagnostic_estimator(scale)._result(raw, 0.9, 0.2)
                self.assertTrue(result.ready)
                self.assertEqual(result.eye_center_mm[1], native_y * scale)
                self.assertEqual(result.pupil_diameter_mm, 4.0 * scale)
                self.assertEqual(result.model_diagnostics.native_eye_center_mm[1], native_y)
                self.assertEqual(result.model_diagnostics.native_pupil_diameter_mm, 4.0)
                self.assertEqual(result.model_diagnostics.range_status, expected_status)

    def test_low_model_confidence_does_not_stop_subsequent_updates(self):
        estimator = diagnostic_estimator()
        raw = diagnostic_raw_result()
        raw["sphere"]["center"] = (0.0, 12.0, 40.0)
        raw["model_confidence"] = 0.1
        estimator.detector.update_and_detect.return_value = raw
        frame = np.zeros((32, 32), np.uint8)
        ellipse = ((16, 16), (12, 18), 0)
        for timestamp in (0.0, 0.25):
            result = estimator.update(ellipse, 0.9, timestamp, frame)
            self.assertTrue(result.ready)
            self.assertEqual(result.model_confidence, 0.1)
            self.assertEqual(result.model_diagnostics.failed_checks, ("eye_center_y",))
        self.assertEqual(estimator.detector.update_and_detect.call_count, 2)
        self.assertEqual(estimator.last_timestamp, 0.25)

    def test_skipped_frame_has_no_stale_diagnostics(self):
        estimator = diagnostic_estimator()
        ready = estimator.update(((16, 16), (12, 18), 0), 0.9, 0.0,
                                 np.zeros((32, 32), np.uint8))
        self.assertIsNotNone(ready.model_diagnostics)
        skipped = estimator.skip_update(0.9, "rejected", "possible pupil coverage")
        self.assertIsNone(skipped.model_diagnostics)
        self.assertIsNone(skipped.model_confidence)
        self.assertEqual(skipped.update_time_ms, 0.0)
        self.assertEqual(estimator.detector.update_and_detect.call_count, 1)
        self.assertEqual(estimator.last_timestamp, 0.0)

    def test_malformed_result_fields_retain_diagnostics_without_crashing(self):
        cases = (("sphere", None), ("sphere", []), ("circle_3d", None),
                 ("circle_3d", []), ("diameter_3d", 10**400))
        for key, value in cases:
            with self.subTest(field=key, value=value):
                raw = diagnostic_raw_result()
                raw[key] = value
                result = diagnostic_estimator()._result(raw, 0.9, 0.2)
                self.assertFalse(result.ready)
                self.assertIsNotNone(result.model_diagnostics)
                self.assertEqual(result.model_diagnostics.range_status, "unavailable")

    def test_nonfinite_model_confidence_is_missing_without_losing_geometry(self):
        for value in (None, "invalid", float("nan"), float("inf"), 10**400):
            with self.subTest(value=value):
                raw = diagnostic_raw_result()
                raw["model_confidence"] = value
                result = diagnostic_estimator()._result(raw, 0.9, 0.2)
                self.assertTrue(result.ready)
                self.assertIsNone(result.model_confidence)
                self.assertEqual(result.model_diagnostics.range_status, "passed")

    def test_console_explains_failed_check_and_native_measurement(self):
        import feature_output

        raw = diagnostic_raw_result()
        raw["sphere"]["center"] = (0.0, 12.0, 40.0)
        raw["model_confidence"] = 0.1
        estimate = diagnostic_estimator(0.5)._result(raw, 0.9, 0.2)
        with patch("builtins.print") as printer:
            feature_output.print_frame_features("left", 1.25, estimate)
        line = printer.call_args.args[0]
        self.assertIn("model_checks=failed", line)
        self.assertIn("model_failed_checks=('eye_center_y',)", line)
        self.assertIn("native_eye_center_mm=(0.00, 12.00, 40.00)", line)
        self.assertIn("eye_center_mm=(0.00, 6.00, 20.00)", line)

    def test_overlay_shows_reason_without_claiming_convergence(self):
        import feature_output

        raw = diagnostic_raw_result()
        raw["sphere"]["center"] = (0.0, 12.0, 40.0)
        estimate = diagnostic_estimator()._result(raw, 0.9, 0.2)
        frame = np.zeros((220, 320, 3), np.uint8)
        original = frame.copy()
        observation = PupilObservation(((160, 110), (70, 70), 0), False, 0.9)
        with patch.object(feature_output, "_draw_text") as draw_text:
            rendered = feature_output.create_output_frame(
                frame, "Left eye", (0, 0, 320, 200), observation, estimate)
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn("Left eye: geometry available | model checks failed", texts)
        self.assertIn("outside default range: eye_center_y", texts)
        self.assertFalse(any("converged" in text.lower() for text in texts))
        self.assertIsNot(rendered, frame)
        np.testing.assert_array_equal(frame, original)

    def test_skipped_console_record_does_not_repeat_native_measurements(self):
        import feature_output

        estimate = diagnostic_estimator().skip_update(0.9, "rejected", "test")
        with patch("builtins.print") as printer:
            feature_output.print_frame_features("left", 1.25, estimate)
        line = printer.call_args.args[0]
        self.assertIn("model_checks=unavailable", line)
        self.assertNotIn("native_eye_center_mm", line)



def _known_circle_pixels(center, normal, radius, matrix, distortion=None):
    """Project a sampled 3D circle with independent pinhole camera equations.

    XYZ and radius share arbitrary length units; output is full-image pixels.
    No project code or pye3d projection is used to create this test's input.
    A basis perpendicular to the normal defines the circular pupil boundary.
    Optional coefficients are OpenCV's (k1, k2, p1, p2, k3) distortion order.
    """
    center = np.asarray(center, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    first = np.cross(normal, [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    angle = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    xyz = center + radius * (
        np.cos(angle)[:, None] * first + np.sin(angle)[:, None] * second
    )
    x, y = (xyz[:, :2] / xyz[:, 2:3]).T
    if distortion is not None:
        k1, k2, p1, p2, k3 = distortion
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        distorted_x = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        distorted_y = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x, y = distorted_x, distorted_y
    return np.column_stack((matrix[0, 0] * x + matrix[0, 2],
                            matrix[1, 1] * y + matrix[1, 2]))


def _known_rotate_pixels(points, size, turns):
    """Reference 90-degree pixel-center rotation, independent of calibration.py."""
    points = np.array(points, dtype=float, copy=True)
    width, height = size
    for _ in range(turns % 4):
        points = np.column_stack((height - 1 - points[:, 1], points[:, 0]))
        width, height = height, width
    return points, (width, height)


def _known_ellipse_matrix(ellipse):
    """Compare ellipse shape without depending on equivalent axis/angle labels."""
    _, diameters, angle = ellipse
    angle = np.deg2rad(angle)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return rotation @ np.diag((np.asarray(diameters) / 2)**2) @ rotation.T


class KnownCameraGeometryTests(unittest.TestCase):
    """Known answers exercise camera math without videos, ML weights, or pye3d."""

    def test_projection_recovers_pixels_for_all_rotation_combinations(self):
        # Unequal fx/fy, off-center principal point, and nonzero tangential
        # distortion expose accidental swaps and correcting in the wrong frame.
        size = (640, 480)
        matrix = np.array([[500., 0., 301.], [0., 540., 227.], [0., 0., 1.]])
        coefficients = np.array([-0.08, 0.015, 0.001, -0.002, 0.0])
        center, normal = [3., -2., 36.], [0.3, -0.2, -1.]
        ideal = _known_circle_pixels(center, normal, 2., matrix)
        distorted = _known_circle_pixels(center, normal, 2., matrix, coefficients)
        names = ("none", "clockwise", "180", "counterclockwise")
        for calibration_turns, calibration_name in enumerate(names):
            for frame_turns, frame_name in enumerate(names):
                with self.subTest(calibration=calibration_name, frame=frame_name):
                    calibration = CameraCalibration(matrix, coefficients, size,
                                                    calibration_name, frame_name)
                    expected, expected_size = _known_rotate_pixels(
                        ideal, size, calibration_turns + frame_turns)
                    observed, _ = _known_rotate_pixels(
                        distorted, size, calibration_turns + frame_turns)
                    self.assertEqual(calibration.video_size, expected_size)
                    np.testing.assert_allclose(calibration.undistort_points(observed),
                                               expected, atol=1e-5, rtol=0)
                    np.testing.assert_allclose(calibration.distort_points(expected),
                                               observed, atol=1e-8, rtol=0)

    def test_roi_and_principal_offsets_are_applied_exactly_once(self):
        matrix = np.array([[500., 0., 301.], [0., 500., 227.], [0., 0., 1.]])
        size = (640, 480)
        points = _known_circle_pixels([3., -2., 36.], [0.3, -0.2, -1.],
                                      2., matrix)
        expected = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
        for origin in ((0, 0), (180, 100), (270, 140)):
            with self.subTest(roi_origin=origin):
                # A contour arrives ROI-local, whereas calibration and pye3d
                # must receive full-frame coordinates. Vary only that crop.
                contour = (points - origin).astype(np.float32).reshape(-1, 1, 2)
                _, full = fit_pupil_ellipse(contour, origin)
                calibration = CameraCalibration(matrix, np.zeros(5), size,
                                                "none", "none")
                corrected = calibration.undistort_ellipse(full)
                converted = _pye3d_ellipse(corrected, *size, 301., 227.)
                np.testing.assert_allclose(
                    np.asarray(converted["center"]) - np.asarray(size) / 2,
                    np.asarray(expected[0]) - [301., 227.], atol=5e-5, rtol=0)
                np.testing.assert_allclose(_known_ellipse_matrix(corrected),
                                           _known_ellipse_matrix(expected),
                                           atol=2e-3, rtol=0)

    def test_swapping_axes_preserves_the_projected_ellipse(self):
        matrix = np.array([[500., 0., 301.], [0., 500., 227.], [0., 0., 1.]])
        points = _known_circle_pixels([3., -2., 36.], [0.6, -0.3, -1.],
                                      2., matrix)
        fitted = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
        center, (a, b), angle = fitted
        # These are two OpenCV descriptions of the same geometric ellipse.
        equivalent = (center, (b, a), angle + 90.)
        converted = [_pye3d_ellipse(value, 640, 480, 301., 227.)
                     for value in (fitted, equivalent)]
        for value in converted:
            self.assertLessEqual(*value["axes"])
        np.testing.assert_allclose(converted[0]["center"], converted[1]["center"])
        np.testing.assert_allclose(converted[0]["axes"], converted[1]["axes"])
        self.assertAlmostEqual(converted[0]["angle"], converted[1]["angle"])

    def test_matched_pixel_scaling_preserves_camera_rays(self):
        # This tests supplied, consistently scaled intrinsics; production code
        # deliberately does not infer calibration changes from a resized image.
        rays = []
        for scale in (0.5, 1.0, 2.0):
            matrix = np.array([[500. * scale, 0., 301. * scale],
                               [0., 500. * scale, 227. * scale], [0., 0., 1.]])
            size = (int(640 * scale), int(480 * scale))
            pixels = _known_circle_pixels([3., -2., 36.], [0.3, -0.2, -1.],
                                          2., matrix)
            fitted = cv2.fitEllipse(pixels.astype(np.float32).reshape(-1, 1, 2))
            converted = _pye3d_ellipse(fitted, *size, *matrix[:2, 2])
            rays.append((np.asarray(converted["center"]) - np.asarray(size)/2)
                        / matrix[0, 0])
        np.testing.assert_allclose(rays, np.repeat([rays[1]], 3, axis=0),
                                   atol=1e-7, rtol=0)

    def test_assumed_eye_radius_scales_lengths_but_not_display_pixels(self):
        matrix = np.array([[500., 0., 301.], [0., 500., 227.], [0., 0., 1.]])
        calibration = CameraCalibration(matrix, np.zeros(5), (640, 480),
                                        "none", "none")
        raw = {"sphere": {"center": [2., -3., 40.]},
               "circle_3d": {"center": [2., -3., 30.], "normal": [0., 0., -1.]},
               "diameter_3d": 4., "model_confidence": 1.,
               "phi": -np.pi / 2, "theta": np.pi / 2,
               "projected_sphere": {"center": [345., 202.5],
                                    "axes": [250., 250.], "angle": 0.},
               "location": [353.3333333333333, 190.]}
        results = []
        for scale in (1., 1.25):
            estimator = object.__new__(EyeModelEstimator)
            estimator.calibration = calibration
            estimator.width, estimator.height = 640, 480
            estimator.cx, estimator.cy = 301., 227.
            estimator.fx = estimator.fy = 500.
            estimator.scale = scale
            results.append(estimator._result(raw, 0.99, 0.2))
        first, scaled = results
        self.assertTrue(first.ready)
        self.assertTrue(scaled.ready)
        np.testing.assert_allclose(scaled.eye_center_mm,
                                   np.asarray(first.eye_center_mm) * 1.25)
        np.testing.assert_allclose(scaled.pupil_center_mm,
                                   np.asarray(first.pupil_center_mm) * 1.25)
        self.assertAlmostEqual(scaled.pupil_diameter_mm,
                               first.pupil_diameter_mm * 1.25)
        np.testing.assert_allclose(first.projected_eye_center, [326., 189.5])
        np.testing.assert_allclose(first.projected_pupil_center, [334.3333333333333, 177.])
        self.assertEqual(first.projected_eye_center, scaled.projected_eye_center)
        self.assertEqual(first.projected_pupil_center, scaled.projected_pupil_center)
        self.assertEqual(first.projected_eye_sphere, scaled.projected_eye_sphere)


@unittest.skipUnless(importlib.util.find_spec("pye3d") is not None,
                     "optional pye3d installation is absent")
class KnownPye3DGeometryTests(unittest.TestCase):
    """Exercise the real installed pye3d ellipse adapter and inverse geometry."""

    def test_zero_confidence_configuration_cannot_enable_unaligned_image_search(self):
        calibration = CameraCalibration(np.array([[500.,0.,301.],[0.,650.,227.],[0.,0.,1.]]),
                                        np.zeros(5), (640,480), "none", "none")
        with self.assertRaisesRegex(ValueError, "minimum confidence must be positive"):
            EyeModelEstimator(calibration, 0., 12.)

    def test_unequal_focals_recover_known_geometry_and_display_projection(self):
        from eye_model_estimation import PYE3D_REFERENCE_EYE_RADIUS_MM

        # An intentionally large 30% focal mismatch makes a mean-focal-only
        # adapter fail visibly. Ground truth is independently projected from 3D.
        matrix = np.array([[500., 0., 301.], [0., 650., 227.], [0., 0., 1.]])
        eye_center = np.array([2., -3., 45.])
        for turns, rotation in enumerate(("none", "clockwise", "180", "counterclockwise")):
            calibration = CameraCalibration(matrix, np.zeros(5), (640, 480), rotation, "none")
            estimator = EyeModelEstimator(calibration, .6, PYE3D_REFERENCE_EYE_RADIUS_MM)
            angle = turns * np.pi / 2
            camera_rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
            width, height = calibration.video_size
            frame = np.zeros((height, width), np.uint8)
            for index in range(120):
                normal = np.array([.25 * np.sin(index * 2 * np.pi / 71),
                                   .2 * np.cos(index * 2 * np.pi / 97), -1.])
                normal /= np.linalg.norm(normal)
                center = eye_center + PYE3D_REFERENCE_EYE_RADIUS_MM * normal
                pixels = _known_circle_pixels(center, normal, 2., matrix)
                pixels, _ = _known_rotate_pixels(pixels, (640, 480), turns)
                ellipse = cv2.fitEllipse(pixels.astype(np.float32).reshape(-1, 1, 2))
                result = estimator.update(calibration.undistort_ellipse(ellipse), .99, index/30., frame)
                if index < 100:
                    continue
                with self.subTest(rotation=rotation, frame=index):
                    self.assertTrue(result.ready)
                    np.testing.assert_allclose(result.eye_center_mm, camera_rotation @ eye_center, atol=.01)
                    np.testing.assert_allclose(result.pupil_center_mm, camera_rotation @ center, atol=.01)
                    np.testing.assert_allclose(result.gaze_direction_camera, camera_rotation @ normal, atol=2e-5)
                    # pye3d 'location' is the projected ellipse center. Under
                    # perspective this differs from directly projecting the
                    # physical 3D circle center, so compare to our fitted conic.
                    np.testing.assert_allclose(result.projected_pupil_center, ellipse[0], atol=.02)

    def test_circle_unprojection_matches_one_physical_solution(self):
        from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
        from pye3d.geometry.projections import unproject_ellipse

        # A single projected circle has two possible 3D plane orientations.
        # Verify that the known answer is one solution; choosing the correct
        # branch over time is the eye model's separate responsibility.
        raw_center = np.array([3., -2., 36.])
        raw_normal = np.array([0.35, -0.2, -1.])
        raw_normal /= np.linalg.norm(raw_normal)
        names = ("none", "clockwise", "180", "counterclockwise")
        for scale in (0.5, 1.0, 2.0):
            matrix = np.array([[500. * scale, 0., 301. * scale],
                               [0., 500. * scale, 227. * scale], [0., 0., 1.]])
            size = (int(640 * scale), int(480 * scale))
            raw_pixels = _known_circle_pixels(raw_center, raw_normal, 2., matrix)
            for turns, rotation in enumerate(names):
                with self.subTest(scale=scale, rotation=rotation):
                    calibration = CameraCalibration(matrix, np.zeros(5), size,
                                                    rotation, "none")
                    pixels, processed_size = _known_rotate_pixels(raw_pixels, size, turns)
                    fitted = cv2.fitEllipse(pixels.astype(np.float32).reshape(-1, 1, 2))
                    corrected = calibration.undistort_ellipse(fitted)
                    converted = _pye3d_ellipse(corrected, *processed_size,
                                              *calibration.video_camera_matrix[:2, 2])
                    detector = Detector3D(camera=CameraModel(500. * scale, processed_size),
                                          long_term_mode=DetectorMode.blocking)
                    observation = detector._extract_observation(
                        {"ellipse": converted, "confidence": 0.99, "timestamp": 0.})
                    self.assertFalse(observation.invalid)
                    circles = unproject_ellipse(observation.ellipse, 500. * scale, radius=2.)
                    angle = turns * np.pi / 2
                    camera_rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                                [np.sin(angle), np.cos(angle), 0.],
                                                [0., 0., 1.]])
                    expected_center = camera_rotation @ raw_center
                    expected_normal = camera_rotation @ raw_normal
                    match = min(circles, key=lambda circle:
                                np.linalg.norm(circle.center - expected_center))
                    np.testing.assert_allclose(match.center, expected_center,
                                               atol=2e-4, rtol=0)
                    np.testing.assert_allclose(match.normal, expected_normal,
                                               atol=2e-5, rtol=0)

    def test_temporal_model_recovers_a_known_moving_pupil(self):
        from eye_model_estimation import PYE3D_REFERENCE_EYE_RADIUS_MM

        # Matching pye3d's mathematical model is intentional: this isolates
        # software conventions, not real anatomy. Pupil centers are one model
        # radius from the eye center; varied gaze supplies fitting information.
        matrix = np.array([[500., 0., 301.], [0., 500., 227.], [0., 0., 1.]])
        raw_eye_center = np.array([2., -3., 45.])
        names = ("none", "clockwise", "180", "counterclockwise")
        for turns, rotation in enumerate(names):
            with self.subTest(rotation=rotation):
                calibration = CameraCalibration(matrix, np.zeros(5), (640, 480),
                                                rotation, "none")
                estimator = EyeModelEstimator(calibration, 0.6,
                                              PYE3D_REFERENCE_EYE_RADIUS_MM)
                angle = turns * np.pi / 2
                camera_rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                                            [np.sin(angle), np.cos(angle), 0.],
                                            [0., 0., 1.]])
                width, height = calibration.video_size
                frame = np.zeros((height, width), np.uint8)
                for index in range(120):
                    normal = np.array([0.25 * np.sin(index * 2 * np.pi / 71),
                                       0.20 * np.cos(index * 2 * np.pi / 97), -1.])
                    normal /= np.linalg.norm(normal)
                    center = raw_eye_center + PYE3D_REFERENCE_EYE_RADIUS_MM * normal
                    raw_pixels = _known_circle_pixels(center, normal, 2., matrix)
                    pixels, _ = _known_rotate_pixels(raw_pixels, (640, 480), turns)
                    fitted = cv2.fitEllipse(pixels.astype(np.float32).reshape(-1, 1, 2))
                    result = estimator.update(calibration.undistort_ellipse(fitted),
                                              0.99, index / 30., frame)
                    if index < 100:
                        continue  # Give the temporal fit a varied observation history.
                    self.assertTrue(result.ready)
                    self.assertEqual(result.model_confidence, 1.0)
                    # Tolerances are intentionally much looser than the observed
                    # float32 error (~0.00002 mm), to allow platform differences.
                    np.testing.assert_allclose(result.eye_center_mm,
                                               camera_rotation @ raw_eye_center,
                                               atol=0.01, rtol=0)
                    np.testing.assert_allclose(result.pupil_center_mm,
                                               camera_rotation @ center,
                                               atol=0.01, rtol=0)
                    self.assertAlmostEqual(result.pupil_diameter_mm, 4., delta=0.01)
                    recovered_normal = (np.asarray(result.pupil_center_mm)
                                        - np.asarray(result.eye_center_mm))
                    recovered_normal /= np.linalg.norm(recovered_normal)
                    angular_error = np.rad2deg(np.arccos(np.clip(
                        np.dot(recovered_normal, camera_rotation @ normal), -1., 1.)))
                    self.assertLess(angular_error, 0.01)


class ModelStabilityMonitorTests(unittest.TestCase):
    """Exercise diagnostic behavior, independent of pye3d internals."""

    def test_constant_center_needs_real_post_startup_history(self):
        monitor = ModelStabilityMonitor()
        results = [monitor.update((1, 2, 40), step / 10) for step in range(71)]
        self.assertEqual(results[49].status, "warming_up")
        self.assertEqual(results[50].status, "insufficient_history")
        self.assertEqual(results[69].status, "insufficient_history")
        self.assertEqual(results[70].status, "stable")
        self.assertEqual(results[70].recent_center_mm, (1, 2, 40))
        self.assertEqual(results[70].reference_center_mm, (1, 2, 40))
        self.assertEqual(results[70].reference_timestamp_s, 7)
        self.assertEqual(results[70].displacement_mm, 0)
        self.assertEqual(results[70].recent_spread_mm, 0)
        self.assertGreaterEqual(results[70].observed_duration_s, 2)

    def test_first_transient_cannot_pollute_reference(self):
        monitor = ModelStabilityMonitor()
        for step in range(71):
            center = (100, 100, 100) if step < 50 else (1, 2, 40)
            result = monitor.update(center, step / 10)
        self.assertEqual(result.status, "stable")
        self.assertEqual(result.reference_center_mm, (1, 2, 40))

    def test_fixed_reference_exposes_slow_drift(self):
        monitor = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            baseline = monitor.update((0, 0, 40), step / 10)
        self.assertEqual(baseline.status, "stable")
        for step in range(21, 151):
            result = monitor.update(((step / 10 - 2) * 0.3, 0, 40), step / 10)
        self.assertEqual(result.status, "drift_detected")
        self.assertEqual(result.reference_center_mm, (0, 0, 40))
        self.assertEqual(result.reference_timestamp_s, 2)
        self.assertGreater(result.displacement_mm, 3)
        self.assertLess(result.recent_spread_mm, 1)

    def test_brief_spike_does_not_create_a_drift_warning(self):
        monitor = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            monitor.update((0, 0, 40), step / 10)
        result = monitor.update((100, 100, 40), 2.1)
        self.assertEqual(result.status, "stable")
        self.assertEqual(result.recent_center_mm, (0, 0, 40))
        self.assertEqual(result.displacement_mm, 0)
        self.assertEqual(result.recent_spread_mm, 0)

    def test_variable_window_cannot_establish_reference(self):
        monitor = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            result = monitor.update((5 if step % 2 else -5, 0, 40), step / 10)
        self.assertEqual(result.status, "unstable")
        self.assertIsNone(result.reference_center_mm)
        self.assertIsNone(result.displacement_mm)
        for step in range(21, 51):
            result = monitor.update((0, 0, 40), step / 10)
        self.assertEqual(result.status, "stable")
        self.assertEqual(result.reference_center_mm, (0, 0, 40))

    def test_brief_skips_are_missing_without_stale_values_or_elapsed_credit(self):
        monitor = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            monitor.update((0, 0, 40), step / 10)
        missing = monitor.update(None, 2.1)
        self.assertEqual(missing.status, "missing")
        self.assertIsNone(missing.reference_center_mm)
        self.assertIsNone(missing.recent_center_mm)
        self.assertIsNone(missing.displacement_mm)
        result = monitor.update((0, 0, 40), 2.2)
        self.assertEqual(result.status, "insufficient_history")
        self.assertLess(result.observed_duration_s, 2)
        for step in range(23, 46):
            result = monitor.update((0, 0, 40), step / 10)
        self.assertEqual(result.status, "stable")
        self.assertEqual(result.reference_timestamp_s, 2)

    def test_long_gap_requires_new_warmup_and_new_reference(self):
        monitor = ModelStabilityMonitor()
        for step in range(71):
            monitor.update((0, 0, 40), step / 10)
        gap = monitor.update(None, 8)
        self.assertEqual(gap.status, "data_gap")
        returned = monitor.update((10, 0, 40), 8.1)
        self.assertEqual(returned.status, "data_gap")
        for step in range(82, 152):
            result = monitor.update((10, 0, 40), step / 10)
        self.assertEqual(result.status, "stable")
        self.assertEqual(result.reference_center_mm, (10, 0, 40))
        self.assertEqual(result.displacement_mm, 0)
        self.assertGreater(result.reference_timestamp_s, 15)

    def test_long_gap_without_explicit_skipped_frames_still_resets(self):
        monitor = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            monitor.update((0, 0, 40), step / 10)
        result = monitor.update((5, 0, 40), 4)
        self.assertEqual(result.status, "data_gap")
        self.assertIsNone(result.reference_center_mm)
        self.assertEqual(result.observed_duration_s, 0)

    def test_separate_eyes_and_reset_do_not_share_history(self):
        left = ModelStabilityMonitor(startup_seconds=0)
        right = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            left.update((0, 0, 40), step / 10)
            right_result = right.update((3, 0, 40), step / 10)
        left.reset()
        left_result = left.update((0, 0, 40), 0)
        self.assertEqual(left_result.status, "insufficient_history")
        self.assertEqual(right_result.reference_center_mm, (3, 0, 40))
        self.assertEqual(right.update((3, 0, 40), 2.1).status, "stable")

    def test_invalid_times_cannot_mutate_history(self):
        monitor = ModelStabilityMonitor(startup_seconds=0)
        control = ModelStabilityMonitor(startup_seconds=0)
        for step in range(21):
            monitor.update((0, 0, 40), step / 10)
            control.update((0, 0, 40), step / 10)
        for invalid in (float("nan"), float("inf"), -1, 2, 1.9, None, "no"):
            with self.subTest(timestamp=invalid), self.assertRaises(ValueError):
                monitor.update((99, 0, 40), invalid)
        self.assertEqual(monitor.update((0, 0, 40), 2.1),
                         control.update((0, 0, 40), 2.1))

    def test_invalid_or_behind_camera_centers_are_missing(self):
        invalid_centers = (None, (), (1, 2), (1, 2, 3, 4), (0, 0, 0),
                           (0, 0, -1), (float("nan"), 0, 40),
                           (0, float("inf"), 40), "abc")
        for center in invalid_centers:
            with self.subTest(center=center):
                monitor = ModelStabilityMonitor(startup_seconds=0)
                result = monitor.update(center, 0)
                self.assertEqual(result.status, "missing")
                self.assertIsNone(result.recent_center_mm)
                self.assertEqual(result.sample_count, 0)

    def test_hard_memory_limit_cannot_fake_observation_duration(self):
        monitor = ModelStabilityMonitor(startup_seconds=0, min_samples=20,
                                        max_samples=20)
        for step in range(500):
            result = monitor.update((0, 0, 40), step / 100)
        self.assertEqual(result.status, "insufficient_history")
        self.assertEqual(result.sample_count, 20)
        self.assertLess(result.observed_duration_s, 0.2)

    def test_configuration_validation(self):
        for options in (
            {"startup_seconds": -1}, {"window_seconds": 0},
            {"max_gap_seconds": float("nan")}, {"max_drift_mm": 0},
            {"max_spread_mm": "bad"}, {"min_samples": 1},
            {"min_samples": 2.5}, {"min_samples": True},
            {"max_samples": 19}, {"max_samples": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ModelStabilityMonitor(**options)


class PerCameraCalibrationWorkflowTests(unittest.TestCase):
    @staticmethod
    def fixture():
        matrix = np.array([[700., 0., 319.5], [0., 720., 239.5], [0., 0., 1.]])
        distortion = np.array([-.08, .025, .001, -.0007, -.005])
        objects = checkerboard_object_points((9, 6), 20.)
        views, poses = [], []
        for index in range(15):
            rotation = np.array([.32 * np.sin(index * 1.7), .4 * np.cos(index * .9), .1 * np.sin(index)])
            translation = np.array([-80. + 50. * np.sin(index * 1.3),
                                    -50. + 40. * np.cos(index * .8),
                                    600. + 100. * np.sin(index * .7)])
            corners, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
            views.append(corners)
            poses.append((rotation, translation))
        return matrix, distortion, views, poses

    def test_known_intrinsics_recovered_from_diverse_synthetic_corners(self):
        matrix, distortion, views, _ = self.fixture()
        result = fit_checkerboard_calibration(views, (640, 480), (9, 6), 20., "left-eye-camera")
        np.testing.assert_allclose(result["camera_matrix"], matrix, atol=.01)
        np.testing.assert_allclose(result["dist_coeffs"].reshape(-1), distortion, atol=.001)
        self.assertLess(result["metadata"]["rms_reprojection_error_px"], .001)
        self.assertEqual(len(result["metadata"]["per_view_rms_error_px"]), 15)
        self.assertIn("unvalidated", result["metadata"]["accuracy_status"])

    def test_identity_round_trip_legacy_policy_and_mismatch(self):
        matrix, distortion, views, _ = self.fixture()
        result = fit_checkerboard_calibration(views, (640, 480), (9, 6), 20., "left-eye-camera")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "left.npz"
            save_checkerboard_calibration(result, path)
            matched = CameraCalibration.load(path, expected_camera_id="left-eye-camera", require_identity=True)
            self.assertEqual(matched.identity_status, "matched")
            self.assertEqual(matched.camera_id, "left-eye-camera")
            self.assertEqual(matched.metadata["view_count"], 15)
            self.assertEqual(CameraCalibration.load(path).identity_status, "declared")
            with self.assertRaisesRegex(ValueError, "does not match"):
                CameraCalibration.load(path, expected_camera_id="right-eye-camera")
            with self.assertRaises(FileExistsError):
                save_checkerboard_calibration(result, path)
            legacy = Path(directory) / "legacy.npz"
            np.savez(legacy, camera_matrix=matrix, dist_coeffs=distortion, image_size=(640, 480))
            self.assertEqual(CameraCalibration.load(legacy, expected_camera_id="left-eye-camera").identity_status, "unverified")
            with self.assertRaisesRegex(ValueError, "identity"):
                CameraCalibration.load(legacy, expected_camera_id="left-eye-camera", require_identity=True)

    def test_invalid_intrinsics_distortion_and_size_are_rejected(self):
        matrix, distortion, _, _ = self.fixture()
        negative_focal = matrix.copy()
        negative_focal[0, 0] = -1
        skew = matrix.copy()
        skew[0, 1] = 1
        bad_bottom = matrix.copy()
        bad_bottom[2, 2] = 2
        cases = [(negative_focal, distortion, (640, 480)),
                 (skew, distortion, (640, 480)),
                 (bad_bottom, distortion, (640, 480)),
                 (matrix, np.zeros(6), (640, 480)),
                 (matrix, np.zeros((2, 2)), (640, 480)),
                 (matrix, [0, 0, 0, float("nan")], (640, 480)),
                 (matrix, distortion, (640.5, 480)),
                 (matrix, distortion, (640, 0))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            for bad_matrix, bad_distortion, bad_size in cases:
                with self.subTest(matrix=bad_matrix, distortion=bad_distortion, size=bad_size):
                    np.savez(path, camera_matrix=bad_matrix, dist_coeffs=bad_distortion, image_size=bad_size)
                    with self.assertRaises(ValueError):
                        CameraCalibration.load(path)

    def test_repeated_and_insufficient_views_are_rejected(self):
        _, _, views, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "repeats"):
            fit_checkerboard_calibration([views[0]] * 12, (640, 480), (9, 6), 20., "camera")
        with self.assertRaisesRegex(ValueError, "at least 10"):
            fit_checkerboard_calibration(views[:3], (640, 480), (9, 6), 20., "camera")
        bad = views[0].copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_checkerboard_calibration([bad] + views[1:], (640, 480), (9, 6), 20., "camera")

    def test_many_frontoparallel_views_remain_degenerate(self):
        matrix, _, _, _ = self.fixture()
        objects = checkerboard_object_points((9, 6), 20.)
        views = [cv2.projectPoints(objects, np.zeros(3), np.array([-80.+index*2, -50., 600.+index*10]),
                                  matrix, np.zeros(5))[0] for index in range(12)]
        with self.assertRaisesRegex(ValueError, "tilt"):
            fit_checkerboard_calibration(views, (640, 480), (9, 6), 20., "camera")

    def test_distortion_cannot_make_frontoparallel_poses_look_diverse(self):
        # Regression: these distorted but untilted views formerly produced an
        # excellent RMS and fx=8728 pixels, although the generating fx was 600.
        # Rechecking pose constraints after distortion correction rejects them.
        objects = checkerboard_object_points((9, 6), 10.)
        matrix = np.array([[600., 0., 320.], [0., 605., 240.], [0., 0., 1.]])
        for radial in (0., -.1, -.2, .1):
            distortion = np.array([radial, .02, .001, -.001, 0.])
            views = [cv2.projectPoints(
                objects, np.zeros(3),
                np.array([-50.+10*(i % 4), -40.+15*(i // 4), 250.+10*(i % 3)]),
                matrix, distortion,
            )[0] for i in range(12)]
            with self.subTest(radial=radial), self.assertRaisesRegex(ValueError, "tilt"):
                fit_checkerboard_calibration(views, (640, 480), (9, 6), 10., "camera")

    def test_saving_inconsistent_metadata_does_not_create_an_artifact(self):
        matrix, distortion, views, _ = self.fixture()
        result = fit_checkerboard_calibration(views, (640, 480), (9, 6), 20., "left-eye-camera")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            for metadata in ([], {"camera_id": "wrong-camera"}):
                with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                    save_checkerboard_calibration(dict(result, metadata=metadata), path)
                self.assertFalse(path.exists())

    def test_corner_noise_does_not_substitute_for_board_tilt(self):
        # Tiny random errors can restore full homography rank for a physically
        # untilted board. The separate normal-span policy catches this case.
        objects = checkerboard_object_points((9, 6), 10.)
        matrix = np.array([[600., 0., 320.], [0., 605., 240.], [0., 0., 1.]])
        distortion = np.array([-.1, .02, .001, -.001, 0.])
        rng = np.random.default_rng(12)
        views = []
        for index in range(12):
            points = cv2.projectPoints(
                objects, np.zeros(3),
                np.array([-50.+10*(index % 4), -40.+15*(index // 4), 250.+10*(index % 3)]),
                matrix, distortion,
            )[0]
            views.append(points + rng.normal(0, .1, points.shape).astype(np.float32))
        with self.assertRaisesRegex(ValueError, "tilt"):
            fit_checkerboard_calibration(views, (640, 480), (9, 6), 10., "camera")

    def test_cli_detects_rendered_boards_and_records_provenance(self):
        matrix, _, _, poses = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "boards"
            folder.mkdir()
            # Render an independent visual fixture with ideal pinhole projection.
            # Supersampling reduces pixel quantization; detection receives pixels,
            # not the exact corner coordinates used by the mathematical test.
            render_matrix = matrix.copy()
            render_matrix[:2] *= 3
            for index, (rotation, translation) in enumerate(poses):
                image = np.full((1440, 1920), 180, np.uint8)
                for row in range(-1, 6):
                    for column in range(-1, 9):
                        square = np.array([[column, row, 0], [column+1, row, 0],
                                           [column+1, row+1, 0], [column, row+1, 0]], np.float32) * 20
                        pixels, _ = cv2.projectPoints(square, rotation, translation, render_matrix, np.zeros(5))
                        cv2.fillConvexPoly(image, np.rint(pixels.reshape(-1, 2)).astype(np.int32),
                                          255 if (row+column) % 2 else 0, lineType=cv2.LINE_AA)
                image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
                self.assertTrue(cv2.imwrite(str(folder / f"view_{index:02d}.png"), image))
            output = Path(directory) / "camera.npz"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                result = calibration_main(["calibrate", "--images", str(folder), "--camera-id", "synthetic-test-camera",
                                           "--board-cols", "9", "--board-rows", "6", "--square-size-mm", "20",
                                           "--output", str(output)])
            self.assertEqual(result, 0)
            report = json.loads(printed.getvalue())
            self.assertEqual(report["collection_counts"]["examined"], 15)
            self.assertGreaterEqual(report["view_count"], 10)
            self.assertEqual(len(report["input_files"]), 15)
            self.assertTrue(all(len(entry["sha256"]) == 64 for entry in report["input_files"]))
            recovered = CameraCalibration.load(output, expected_camera_id="synthetic-test-camera")
            np.testing.assert_allclose(recovered.raw_camera_matrix[:2, :2], matrix[:2, :2], atol=10.)
            self.assertLess(report["rms_reprojection_error_px"], .6)

    def test_recording_collection_rejects_mixed_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            cv2.imwrite(str(Path(directory) / "a.png"), np.zeros((100, 100), np.uint8))
            cv2.imwrite(str(Path(directory) / "b.png"), np.zeros((120, 100), np.uint8))
            with self.assertRaisesRegex(ValueError, "mixed"):
                calibrate_from_recordings(images=directory, camera_id="camera", board_shape=(9, 6), square_size_mm=20.)


class GazeTargetValidationTests(unittest.TestCase):
    @staticmethod
    def fixture():
        """Generate distinct training/test rays with a known seven-degree offset."""
        angle = np.deg2rad(7.0)
        rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ])
        rays = [
            [x, y, -1.0]
            for x, y in [(-.2, -.2), (.2, -.2), (-.2, .2), (.2, .2),
                         (-.1, -.15), (.13, -.1), (-.12, .15), (.14, .1)]
        ]
        records = []
        samples = []
        for frame, ray in enumerate(rays):
            ray = np.asarray(ray) / np.linalg.norm(ray)
            records.append({
                "schema_version": 1,
                "recording_id": "synthetic_known_rotation",
                "coordinate_system": "processed_eye_camera",
                "eye": "left", "frame_index": frame, "timestamp_s": frame / 60.0,
                "camera_id": "synthetic-left", "calibration_sha256": "a" * 64,
                "frame_rotation": "none", "calibration_rotation": "none",
                "gaze_direction_camera": ray.tolist(), "model_input": "accepted", "ready": True,
                "model_diagnostics": {"range_checks_passed": False},
            })
            samples.append({
                "eye": "left", "frame_index": frame, "timestamp_s": frame / 60.0,
                "direction_camera": (rotation @ ray).tolist(),
                "split": "calibration" if frame < 4 else "validation",
            })
        targets = {
            "schema_version": 1, "data_kind": "synthetic",
            "direction_source": "Independent synthetic rigid rotation fixture",
            "recording_id": "synthetic_known_rotation", "coordinate_system": "processed_eye_camera",
            "camera_ids": {"left": "synthetic-left"}, "calibration_sha256": {"left": "a" * 64},
            "frame_rotation": {"left": "none"}, "calibration_rotation": {"left": "none"},
            "samples": samples,
        }
        return records, targets, rotation

    def test_recovers_rotation_on_disjoint_later_targets(self):
        records, targets, rotation = self.fixture()
        result = evaluate_gaze(records, targets, fit_rotation=True)
        np.testing.assert_allclose(result["eyes"]["left"]["rotation_predicted_to_reference"], rotation, atol=1e-12)
        self.assertGreater(result["raw_angular_error"]["mean_deg"], 6.0)
        self.assertLess(result["calibrated_angular_error"]["max_deg"], 1e-10)
        self.assertEqual(result["raw_angular_error"]["count"], 4)
        self.assertEqual(result["eyes"]["left"]["rotation_fit_sample_count"], 4)
        self.assertFalse(result["real_reference_supplied"])
        self.assertIsNone(result["accuracy_verdict"])
        self.assertIn("real gaze accuracy remains unmeasured", result["interpretation"])

    def test_validation_targets_do_not_change_rotation_fit(self):
        records, targets, rotation = self.fixture()
        for sample in targets["samples"][4:]:
            sample["direction_camera"] = [1.0, 0.0, 0.0]
        result = evaluate_gaze(records, targets, fit_rotation=True)
        np.testing.assert_allclose(result["eyes"]["left"]["rotation_predicted_to_reference"], rotation, atol=1e-12)
        self.assertGreater(result["calibrated_angular_error"]["mean_deg"], 80.0)

    def test_each_eye_gets_its_own_independent_rotation(self):
        records, targets, left_rotation = self.fixture()
        right_rotation = np.diag([-1.0, -1.0, 1.0])
        right_records = [dict(record, eye="right", camera_id="synthetic-right") for record in records]
        right_samples = []
        for record, sample in zip(right_records, targets["samples"]):
            right_samples.append(dict(
                sample, eye="right",
                direction_camera=(right_rotation @ record["gaze_direction_camera"]).tolist(),
            ))
        records.extend(right_records)
        targets["samples"].extend(right_samples)
        targets["camera_ids"]["right"] = "synthetic-right"
        for field in ("calibration_sha256", "frame_rotation", "calibration_rotation"):
            targets[field]["right"] = targets[field]["left"]
        result = evaluate_gaze(records, targets, fit_rotation=True)
        np.testing.assert_allclose(result["eyes"]["left"]["rotation_predicted_to_reference"], left_rotation, atol=1e-12)
        np.testing.assert_allclose(result["eyes"]["right"]["rotation_predicted_to_reference"], right_rotation, atol=1e-12)
        self.assertEqual(result["raw_angular_error"]["count"], 8)
        self.assertLess(result["calibrated_angular_error"]["max_deg"], 1e-10)

    def test_skipped_and_startup_frames_reduce_coverage_without_interpolation(self):
        records, targets, _ = self.fixture()
        records[4].update(model_input="skipped", ready=False, gaze_direction_camera=None)
        records[5].update(ready=False, gaze_direction_camera=None)
        records[6]["gaze_direction_camera"] = None
        result = evaluate_gaze(records, targets)
        self.assertEqual(result["validation_coverage"]["counts"], {
            "missing_gaze": 1, "not_ready": 1, "skipped": 1, "usable": 1,
        })
        self.assertEqual(result["validation_coverage"]["usable_fraction"], .25)
        self.assertEqual(result["raw_angular_error"]["count"], 1)
        self.assertIsNone(result["calibrated_angular_error"])

    def test_no_usable_validation_gives_null_metrics(self):
        records, targets, _ = self.fixture()
        for record in records[4:]:
            record.update(model_input="skipped", ready=False, gaze_direction_camera=None)
        result = evaluate_gaze(records, targets)
        self.assertEqual(result["raw_angular_error"]["count"], 0)
        self.assertIsNone(result["raw_angular_error"]["mean_deg"])
        self.assertEqual(result["validation_coverage"]["usable_fraction"], 0.0)

    def test_identity_and_coordinate_mismatches_are_rejected(self):
        for field, value in [
            ("recording_id", "different-recording"), ("camera_id", None),
            ("calibration_sha256", "b" * 64), ("frame_rotation", "cw90"),
            ("calibration_rotation", "ccw90"), ("coordinate_system", "world"),
        ]:
            with self.subTest(field=field):
                records, targets, _ = self.fixture()
                records[0][field] = value
                with self.assertRaises(ValueError):
                    evaluate_gaze(records, targets)

    def test_exact_frame_and_timestamp_alignment_is_required(self):
        records, targets, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "absent"):
            evaluate_gaze(records[:-1], targets)
        targets["samples"][-1]["timestamp_s"] += .001
        with self.assertRaisesRegex(ValueError, "timestamp does not match"):
            evaluate_gaze(records, targets)

    def test_duplicate_records_targets_and_temporal_leakage_are_rejected(self):
        records, targets, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate exported"):
            evaluate_gaze(records + [records[0]], targets)
        duplicate = copy.deepcopy(targets)
        duplicate["samples"].append(dict(targets["samples"][0], split="validation"))
        with self.assertRaisesRegex(ValueError, "Duplicate target"):
            evaluate_gaze(records, duplicate)
        targets["samples"][0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "before all validation"):
            evaluate_gaze(records, targets, fit_rotation=True)

    def test_invalid_and_zero_directions_fail(self):
        for direction in (
            [0, 0, 0], [1, 0, float("nan")], [1, float("inf"), 0], [1, 2],
            [True, 0, 1], [1, np.bool_(False), 1], ["1", 0, 1], [1, "0.5", 1],
        ):
            with self.subTest(direction=direction):
                records, targets, _ = self.fixture()
                records[0]["gaze_direction_camera"] = direction
                with self.assertRaises(ValueError):
                    evaluate_gaze(records, targets)
                records, targets, _ = self.fixture()
                targets["samples"][0]["direction_camera"] = direction
                with self.assertRaises(ValueError):
                    evaluate_gaze(records, targets)

    def test_export_schema_version_must_be_integer_one(self):
        for version in (None, True, 1.0, "1", 2):
            records, targets, _ = self.fixture()
            if version is None:
                del records[0]["schema_version"]
            else:
                records[0]["schema_version"] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "schema_version"):
                evaluate_gaze(records, targets)

    def test_rotation_names_are_valid_even_when_metadata_matches(self):
        for field in ("frame_rotation", "calibration_rotation"):
            for invalid in ("cw90", "ccw90", 180, "unsupported"):
                records, targets, _ = self.fixture()
                targets[field]["left"] = invalid
                for record in records:
                    record[field] = invalid
                with self.subTest(field=field, value=invalid), self.assertRaises(ValueError):
                    evaluate_gaze(records, targets)
        for valid in ("none", "clockwise", "180", "counterclockwise"):
            records, targets, _ = self.fixture()
            for field in ("frame_rotation", "calibration_rotation"):
                targets[field]["left"] = valid
                for record in records:
                    record[field] = valid
            with self.subTest(valid=valid):
                result = evaluate_gaze(records, targets)
                self.assertEqual(result["raw_angular_error"]["count"], 4)

    def test_skipped_frame_cannot_export_stale_geometry(self):
        records, targets, _ = self.fixture()
        records[4].update(model_input="skipped", ready=False)
        with self.assertRaisesRegex(ValueError, "stale"):
            evaluate_gaze(records, targets)

    def test_rotation_needs_three_usable_noncollinear_pairs(self):
        records, targets, _ = self.fixture()
        for record in records[:2]:
            record.update(model_input="skipped", ready=False, gaze_direction_camera=None)
        with self.assertRaisesRegex(ValueError, "at least three"):
            evaluate_gaze(records, targets, fit_rotation=True)
        with self.assertRaisesRegex(ValueError, "collinear"):
            fit_direction_rotation([[0, 0, -1]] * 3, [[0, 0, -1]] * 3)
        with self.assertRaisesRegex(ValueError, "collinear"):
            fit_direction_rotation([[0, 0, -1], [1, 0, -1], [0, 0, -1]], [[0, 0, -1], [1, 0, -1], [0, 0, -1]])

    def test_rotation_cannot_be_a_reflection(self):
        predicted = np.eye(3)
        references = np.diag([-1.0, 1.0, 1.0])
        rotation = fit_direction_rotation(predicted, references)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=12)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertGreater(np.linalg.norm((rotation @ predicted.T).T - references), 1.0)

    def test_empty_inputs_and_unknown_provenance_fail(self):
        records, targets, _ = self.fixture()
        with self.assertRaises(ValueError):
            evaluate_gaze([], targets)
        for field, value in [("samples", []), ("data_kind", "unknown"), ("direction_source", "")]:
            changed = copy.deepcopy(targets)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evaluate_gaze(records, changed)

    def test_cli_writes_report_without_changing_inputs(self):
        records, targets, _ = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path = root / "records.jsonl"
            targets_path = root / "targets.json"
            report_path = root / "report.json"
            records_text = "".join(json.dumps(record) + "\n" for record in records)
            targets_text = json.dumps(targets)
            records_path.write_text(records_text)
            targets_path.write_text(targets_text)
            result = gaze_validation_main([
                "--records", str(records_path), "--targets", str(targets_path),
                "--output", str(report_path), "--fit-rotation",
            ])
            self.assertEqual(result, 0)
            self.assertEqual(records_path.read_text(), records_text)
            self.assertEqual(targets_path.read_text(), targets_text)
            report = json.loads(report_path.read_text())
            self.assertLess(report["calibrated_angular_error"]["max_deg"], 1e-10)


class BlinkValidationScoringTests(unittest.TestCase):
    def records(self, decisions=("accepted",), mode="filtered", eye="left"):
        return [dict(
            eye=eye, frame_index=index, timestamp_s=index / 30,
            model_input=decision, ready=decision == "accepted", filter_mode=mode,
            video_sha256="a" * 64, frame_rotation="none", roi=[0, 0, 20, 10],
            recording_id="recording", calibration_sha256="b" * 64,
            model_sha256="c" * 64, min_confidence=0.6, mask_threshold=0.5,
            recovery_confidence=0.7, recovery_duration_s=0.05,
            source_fingerprint="source", quality={"reason": "pupil checks passed"},
            model_diagnostics={"native_eye_center_mm": [index, 0, 30]},
        ) for index, decision in enumerate(decisions)]

    def labels(self, values=("usable",), eye="left", reviewer="Reviewer A"):
        return [dict(eye=eye, frame_index=index, label=value, reason="Human judgment",
                     reviewer=reviewer, split="development", reviewed=True,
                     video_sha256="a" * 64, frame_rotation="none", roi=[0, 0, 20, 10])
                for index, value in enumerate(values)]

    def test_mistake_list_matches_rates_and_source_frames(self):
        records = self.records(("accepted", "skipped", "accepted", "skipped"))
        records[0]["pupil_confidence"] = .9
        labels = self.labels(("unusable", "usable", "uncertain", "usable"))
        labels[3]["reviewed"] = False
        result = score_records(records, labels)
        mistakes = result["mistakes"]
        self.assertEqual([m["frame_index"] for m in mistakes], [0, 1])
        self.assertEqual([m["kind"] for m in mistakes], ["bad_frame_admitted", "good_frame_discarded"])
        self.assertEqual(mistakes[0]["reviewer_note"], "Human judgment")
        self.assertEqual(mistakes[0]["pupil_confidence"], .9)
        self.assertIsNone(mistakes[1]["pupil_confidence"])
        self.assertEqual(len(mistakes), result["aggregate"]["bad_frames_admitted"]["count"]
                         + result["aggregate"]["good_frames_discarded"]["count"])

    def test_raw_observations_are_checked_as_paired_inputs(self):
        baseline=self.records(mode="baseline");filtered=self.records()
        for records in (baseline,filtered):
            records[0]["pupil_observation"]={"ellipse":[[10,12],[5,7],0]}
            records[0]["pupil_observation_coordinate_system"]="rotated_raw_frame_pixels"
        compare_records(baseline,filtered,[])
        filtered[0]["pupil_observation"]["ellipse"][0][0]=11
        with self.assertRaisesRegex(ValueError,"pupil_observation"):
            compare_records(baseline,filtered,[])

    def test_timing_may_differ_but_tracking_inputs_must_match(self):
        baseline=self.records(mode="baseline");filtered=self.records()
        baseline[0]["frame_pair_timing_ms"]={"detection_ms":12.}
        filtered[0]["frame_pair_timing_ms"]={"detection_ms":15.}
        for rows in (baseline,filtered):
            rows[0]["roi_mode"]="tracked"
            rows[0]["roi_tracking"]={"detection_roi":[0,0,20,10]}
        compare_records(baseline,filtered,[])
        filtered[0]["roi_tracking"]={"detection_roi":[1,0,19,10]}
        with self.assertRaisesRegex(ValueError,"roi_tracking"):
            compare_records(baseline,filtered,[])

    def test_mistake_report_preserves_raw_observation_and_old_run_absence(self):
        records=self.records(("skipped",))
        self.assertIsNone(score_records(records,self.labels())["mistakes"][0]["pupil_observation"])
        observation={"ellipse":[[10,12],[5,7],0],"boundary_rejection":"image rejection"}
        records[0]["pupil_observation"]=observation
        records[0]["pupil_observation_coordinate_system"]="rotated_raw_frame_pixels"
        result=score_records(records,self.labels())["mistakes"][0]
        self.assertEqual(result["pupil_observation"],observation)
        self.assertEqual(result["pupil_observation_coordinate_system"],"rotated_raw_frame_pixels")

    def test_robot_decision_is_output_not_comparison_input(self):
        baseline=self.records(mode="baseline");filtered=self.records(mode="filtered")
        baseline[0]["robot_command"]={"allowed":False,"reason":"baseline"}
        filtered[0]["robot_command"]={"allowed":False,"reason":"offline"}
        self.assertEqual(compare_records(baseline,filtered,self.labels())["baseline"]["aggregate"]["scored_frames"],1)

    def test_counts_and_denominators_match_labels(self):
        records = self.records(("accepted", "skipped", "accepted", "skipped"))
        labels = self.labels(("unusable", "unusable", "usable", "usable"))
        result = score_records(records, labels)
        self.assertEqual(result["aggregate"]["bad_frames_admitted"],
                         {"count": 1, "total": 2, "percent": 50.0})
        self.assertEqual(result["aggregate"]["good_frames_discarded"],
                         {"count": 1, "total": 2, "percent": 50.0})
        self.assertIsNone(result["by_eye"]["right"]["bad_frames_admitted"]["percent"])

    def test_suggestions_and_uncertain_are_excluded(self):
        labels = self.labels(("unusable", "uncertain", "usable"))
        labels[0]["reviewed"] = False
        result = score_records(self.records(("accepted",) * 3), labels)
        self.assertEqual(result["excluded"], {"without_label": 0, "unreviewed": 1, "uncertain_reviewed": 1})
        self.assertEqual(result["aggregate"]["scored_frames"], 1)
        self.assertIsNone(result["aggregate"]["bad_frames_admitted"]["percent"])

    def test_empty_and_all_uncertain_labels_are_not_perfect_scores(self):
        for labels in ([], self.labels(("uncertain",))):
            result = score_records(self.records(), labels)
            self.assertEqual(result["evidence_status"], "no_scorable_human_labels")
            self.assertIsNone(result["aggregate"]["good_frames_discarded"]["percent"])

    def test_unlabeled_export_frames_are_explicitly_counted(self):
        result = score_records(self.records(("accepted",) * 3), self.labels())
        self.assertEqual(result["excluded"]["without_label"], 2)
        self.assertEqual(result["aggregate"]["scored_frames"], 1)
        empty = score_records(self.records(("accepted",) * 3), [])
        self.assertEqual(empty["excluded"]["without_label"], 3)
        self.assertIsNone(empty["aggregate"]["good_frames_discarded"]["percent"])

    def test_no_export_is_an_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "contains no frames"):
            score_records([], [])

    def test_label_binding_and_missing_frames_rejected(self):
        for field, wrong in (("video_sha256", "d" * 64), ("roi", [1, 0, 20, 10]),
                             ("frame_rotation", "clockwise")):
            labels = self.labels()
            labels[0][field] = wrong
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                score_records(self.records(), labels)
        labels = self.labels()
        labels[0]["frame_index"] = 9
        with self.assertRaisesRegex(ValueError, "Missing export"):
            score_records(self.records(), labels)

    def test_duplicate_records_labels_and_mixed_reviewers_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate export"):
            score_records(self.records() * 2, [])
        with self.assertRaisesRegex(ValueError, "Duplicate label"):
            score_records(self.records(), self.labels() * 2)
        labels = self.labels(("usable", "usable"))
        labels[1]["reviewer"] = "Reviewer B"
        with self.assertRaisesRegex(ValueError, "one reviewer"):
            score_records(self.records(("accepted",) * 2), labels)

    def test_timestamps_are_finite_strictly_increasing(self):
        for wrong in (math.nan, math.inf, -1, 0, True):
            records = self.records(("accepted",) * 2)
            records[1]["timestamp_s"] = wrong
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                score_records(records, [])

    def test_unknown_decision_and_malformed_label_rejected(self):
        records = self.records()
        records[0]["model_input"] = "maybe"
        with self.assertRaises(ValueError):
            score_records(records, [])
        for field, wrong in (("label", "open"), ("reviewed", "yes"), ("frame_index", True),
                             ("split", "test"), ("reason", None), ("roi", [0, 0, -1, 2])):
            labels = self.labels()
            labels[0][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                score_records(self.records(), labels)

    def test_recovery_counts_delay_and_fully_missed_interval_separately(self):
        labels = self.labels(("unusable", "usable", "usable", "unusable", "usable"))
        records = self.records(("skipped", "skipped", "accepted", "accepted", "accepted"))
        recovery = score_records(records, labels)["aggregate"]["recovery"]
        self.assertEqual(recovery["event_count"], 2)
        self.assertEqual(recovery["missed_prior_bad_events"], 1)
        self.assertAlmostEqual(recovery["mean_delay_s"], 1 / 30)
        self.assertEqual(recovery["events"][1]["delay_s"], 0)
        self.assertTrue(recovery["events"][1]["fully_missed_bad_interval"])

    def test_recovery_censor_and_label_gap_do_not_invent_delay(self):
        labels = self.labels(("unusable", "usable", "usable"))
        records = self.records(("skipped",) * 3)
        recovery = score_records(records, labels)["aggregate"]["recovery"]
        self.assertEqual(recovery["censored_count"], 1)
        self.assertIsNone(recovery["mean_delay_s"])
        self.assertEqual(recovery["events"][0]["last_observed_usable_frame"], 2)
        recovery = score_records(records, [labels[0], labels[2]])["aggregate"]["recovery"]
        self.assertEqual(recovery["event_count"], 0)

    def test_opposite_transition_and_uncertain_gap_are_not_recovery(self):
        records = self.records(("accepted",) * 3)
        for values in (("usable", "unusable", "unusable"), ("unusable", "uncertain", "usable")):
            recovery = score_records(records, self.labels(values))["aggregate"]["recovery"]
            self.assertEqual(recovery["event_count"], 0)

    def test_geometry_and_recovery_reasons_remain_separate(self):
        records = self.records(("skipped", "skipped", "skipped"))
        for record, reason in zip(records, ("invalid pupil geometry", "possible eyelid closure", "waiting for stable pupil measurements")):
            record["quality"]["reason"] = reason
        groups = score_records(records, self.labels(("usable",) * 3))["aggregate"]["reason_groups"]
        self.assertEqual(groups, {"usable / skipped / geometry": 1,
                                  "usable / skipped / closure_or_coverage": 1,
                                  "usable / skipped / recovery": 1})

    def test_development_and_held_out_remain_separate(self):
        labels = self.labels(("usable", "unusable"))
        labels[1]["split"] = "held_out"
        records = self.records(("accepted",) * 2)
        self.assertEqual(score_records(records, labels)["aggregate"]["bad_frames_admitted"]["total"], 0)
        self.assertEqual(score_records(records, labels, split="held_out")["aggregate"]["bad_frames_admitted"]["count"], 1)

    def test_comparison_requires_identical_inputs_and_correct_modes(self):
        baseline = self.records(mode="baseline")
        for field, wrong in (("recording_id", "other"), ("source_fingerprint", "other"),
                             ("mask_threshold", 0.4), ("recovery_duration_s", 0.1),
                             ("future_configuration_option", "new"), ("filter_mode", "baseline")):
            filtered = self.records()
            filtered[0][field] = wrong
            with self.subTest(field=field), self.assertRaises(ValueError):
                compare_records(baseline, filtered, [])
        with self.assertRaisesRegex(ValueError, "identical eye/frame"):
            compare_records(baseline, self.records(("accepted",) * 2), [])

    def test_comparison_uses_exact_adjacent_common_ready_frames(self):
        baseline = self.records(("accepted",) * 5, mode="baseline")
        filtered = self.records(("accepted", "skipped", "accepted", "accepted", "accepted"))
        filtered[3]["model_diagnostics"]["native_eye_center_mm"] = [2.5, 0, 30]
        filtered[4]["model_diagnostics"]["native_eye_center_mm"] = [3, 0, 30]
        result = compare_records(baseline, filtered, [])
        model = result["model_output"]
        self.assertEqual(model["common_ready"], 4)
        self.assertEqual(model["adjacent_common_ready_center_steps"], 2)
        self.assertEqual(model["baseline_center_step_native_mm"]["mean"], 1)
        self.assertEqual(model["filtered_center_step_native_mm"]["mean"], 0.5)
        self.assertIn("do not establish gaze accuracy", format_comparison_report(result))

    def test_nonfinite_centers_excluded_from_both_paired_step_sets(self):
        baseline = self.records(("accepted",) * 2, mode="baseline")
        filtered = self.records(("accepted",) * 2)
        filtered[0]["model_diagnostics"]["native_eye_center_mm"] = [math.nan, 0, 30]
        model = compare_records(baseline, filtered, [])["model_output"]
        self.assertEqual(model["adjacent_common_ready_center_steps"], 0)
        self.assertIsNone(model["baseline_center_step_native_mm"]["mean"])

    def test_runtime_is_descriptive_and_validated(self):
        baseline, filtered = self.records(mode="baseline"), self.records()
        result = compare_records(baseline, filtered, [], run_summaries={
            "baseline": {"elapsed_seconds": 2}, "filtered": {"wall_time_seconds": 3}})
        self.assertEqual(result["runtime"]["filtered_elapsed_seconds"], 3)
        self.assertIn("not a runtime benchmark", result["runtime"]["note"])
        with self.assertRaises(ValueError):
            compare_records(baseline, filtered, [], run_summaries={
                "baseline": {"elapsed_seconds": 0}, "filtered": {"elapsed_seconds": 1}})

    def test_reviewer_agreement_excludes_suggestions_and_retains_disagreements(self):
        a = self.labels(("usable", "unusable", "uncertain"))
        b = self.labels(("usable", "usable", "usable"), reviewer="Reviewer B")
        b[2]["reviewed"] = False
        pair = reviewer_agreement(a + b)["pairs"][0]
        self.assertEqual(pair["common_reviewed_frames"], 2)
        self.assertEqual(pair["agreement"]["percent"], 50)
        self.assertEqual(pair["disagreements"][0]["frame_index"], 1)

    def test_reviewer_agreement_requires_same_task_binding(self):
        a, b = self.labels(), self.labels(reviewer="Reviewer B")
        b[0]["roi"] = [1, 0, 20, 10]
        with self.assertRaisesRegex(ValueError, "bindings/splits differ"):
            reviewer_agreement(a + b)

    def test_eye_histories_are_independent(self):
        left = self.records(("skipped", "accepted"))
        right = self.records(("accepted", "accepted"), eye="right")
        labels = self.labels(("unusable", "usable")) + self.labels(("unusable", "usable"), eye="right")
        result = score_records(left + right, labels)
        self.assertEqual(result["by_eye"]["left"]["recovery"]["detected_prior_bad_events"], 1)
        self.assertEqual(result["by_eye"]["right"]["recovery"]["missed_prior_bad_events"], 1)


class BlinkPipelineModeTests(unittest.TestCase):
    @staticmethod
    def pupil(*, confidence=0.9, state="no_closure_evidence", ellipse=((12., 12.), (8., 10.), 0.)):
        return PupilObservation(ellipse, False, confidence, EyelidObservation(state=state))

    @staticmethod
    def estimate():
        return EyeModelEstimate(ready=False, eye_center_mm=None, pupil_center_mm=None,
            pupil_diameter_mm=None, pupil_confidence=0.9, model_confidence=None,
            update_time_ms=0., projected_eye_sphere=None, projected_eye_center=None,
            projected_pupil_center=None, status="warmup")

    def test_baseline_bypasses_only_eyelid_evidence_and_does_not_mutate_observation(self):
        tracker = Mock()
        for state in ("closed_possible", "occlusion_possible"):
            pupil = self.pupil(state=state)
            with self.subTest(state=state), patch.object(blink_pipeline, "FILTER_MODE", "baseline"):
                adapted = blink_pipeline._filter_observation(pupil)
                self.assertIsNone(adapted.eyelid)
                self.assertEqual(pupil.eyelid.state, state)
                quality = blink_pipeline._quality_for_mode(pupil, 0., None, tracker)
                self.assertTrue(quality.allow_model_update)
                self.assertEqual(quality.state, "usable")
        tracker.update.assert_not_called()

    def test_baseline_still_rejects_missing_invalid_and_low_confidence_pupils(self):
        cases = (self.pupil(ellipse=None), self.pupil(confidence=0.59),
                 self.pupil(confidence=float("nan")), self.pupil(confidence=1.1),
                 self.pupil(ellipse=((12., 12.), (0., 10.), 0.)),
                 self.pupil(ellipse=((float("inf"), 12.), (8., 10.), 0.)))
        with patch.object(blink_pipeline, "FILTER_MODE", "baseline"):
            for pupil in cases:
                with self.subTest(pupil=pupil):
                    quality = blink_pipeline._quality_for_mode(pupil, 0., None, Mock())
                    self.assertFalse(quality.allow_model_update)
                    self.assertEqual(quality.state, "rejected")

    def test_geometry_protection_is_retained_after_baseline_eyelid_bypass(self):
        calibration, model = Mock(), Mock()
        corrected = ((-100., 12.), (8., 10.), 0.)
        calibration.undistort_ellipse.return_value = corrected
        model.assess_input_geometry.return_value = SimpleNamespace(
            allow_model_update=False, reason="ellipse center outside pye3d image bounds")
        pupil = self.pupil(state="closed_possible")
        with patch.object(blink_pipeline, "FILTER_MODE", "baseline"):
            ellipse, reason = blink_pipeline._prepare_corrected_model_input(pupil, calibration, model)
            quality = blink_pipeline._quality_for_mode(pupil, 0., reason, Mock())
        self.assertEqual(ellipse, corrected)
        self.assertFalse(quality.allow_model_update)
        self.assertIn("bounds", quality.reason)
        calibration.undistort_ellipse.assert_called_once_with(pupil.ellipse)
        model.assess_input_geometry.assert_called_once_with(corrected)

    def test_rejected_pupil_is_not_sent_to_lens_correction_in_either_mode(self):
        for mode in ("baseline", "filtered"):
            calibration, model = Mock(), Mock()
            with self.subTest(mode=mode), patch.object(blink_pipeline, "FILTER_MODE", mode):
                result = blink_pipeline._prepare_corrected_model_input(self.pupil(confidence=0.1), calibration, model)
            self.assertEqual(result, (None, None))
            calibration.undistort_ellipse.assert_not_called()
            model.assess_input_geometry.assert_not_called()

    def test_filtered_mode_keeps_temporal_recovery_and_geometry_rejection(self):
        tracker = TemporalQualityTracker(0.6, 0.7, 0.05, max_gap_s=0.1)
        with patch.object(blink_pipeline, "FILTER_MODE", "filtered"):
            first = blink_pipeline._quality_for_mode(self.pupil(), 0., "invalid geometry", tracker)
            second = blink_pipeline._quality_for_mode(self.pupil(), 1 / 30, None, tracker)
            third = blink_pipeline._quality_for_mode(self.pupil(), 2 / 30, None, tracker)
            fourth = blink_pipeline._quality_for_mode(self.pupil(), 3 / 30, None, tracker)
        self.assertFalse(first.allow_model_update)
        self.assertEqual(first.reason, "invalid geometry")
        self.assertEqual(second.state, "recovering")
        self.assertFalse(third.allow_model_update)
        self.assertTrue(fourth.allow_model_update)

    def test_actual_loop_modes_export_skips_and_keep_the_other_eye_independent(self):
        for mode, expected_left_updates in (("baseline", 2), ("filtered", 0)):
            with self.subTest(mode=mode):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                captures = [Mock(), Mock()]
                for capture in captures:
                    capture.read.side_effect = [(True, frame.copy()), (True, frame.copy()), (False, None)]
                calibrations = [Mock(), Mock()]
                for calibration in calibrations:
                    calibration.undistort_ellipse.side_effect = lambda ellipse: ellipse
                models = [Mock(), Mock()]
                for model in models:
                    model.assess_input_geometry.return_value = SimpleNamespace(allow_model_update=True, reason="valid")
                    model.update.return_value = self.estimate()
                    model.skip_update.return_value = self.estimate()
                detector = Mock()
                detector.detect.side_effect = [self.pupil(state="closed_possible"), self.pupil(),
                                               self.pupil(), self.pupil()]
                stream = io.StringIO()
                with patch.multiple(blink_pipeline, FILTER_MODE=mode, MAX_FRAMES=0, TEXT_OUTPUT=False,
                                    PROGRESS_EVERY=0, LEFT_ROI=(0, 0, 32, 32), RIGHT_ROI=(0, 0, 32, 32)):
                    count = blink_pipeline.process_frame_loop(
                        *captures, 30., "none", "none", *calibrations, detector, *models,
                        headless=True, record_stream=stream,
                        record_metadata={eye: {"filter_mode": mode} for eye in ("left", "right")})
                self.assertEqual(count, 2)
                self.assertEqual(models[0].update.call_count, expected_left_updates)
                self.assertEqual(models[0].skip_update.call_count, 2 - expected_left_updates)
                self.assertEqual(models[1].update.call_count, 2)
                rows = [json.loads(line) for line in stream.getvalue().splitlines()]
                self.assertEqual(len(rows), 4)
                self.assertEqual([(row["eye"], row["frame_index"]) for row in rows],
                                 [("left", 0), ("right", 0), ("left", 1), ("right", 1)])
                self.assertTrue(all(row["filter_mode"] == mode for row in rows))
                self.assertAlmostEqual(rows[-1]["timestamp_s"], 1 / 30)
                if mode == "filtered":
                    self.assertEqual(rows[0]["model_input"], "skipped")
                    self.assertEqual(rows[2]["quality"]["state"], "recovering")

    def test_invalid_filter_settings_fail_before_processing(self):
        invalid = ({"MIN_CONFIDENCE": float("nan")}, {"RECOVERY_CONFIDENCE": float("inf")},
                   {"RECOVERY_DURATION_S": float("nan")}, {"MIN_CONFIDENCE": 0.},
                   {"RECOVERY_CONFIDENCE": 0.5}, {"RECOVERY_DURATION_S": -1.},
                   {"FILTER_MODE": "unknown"}, {"ROI_MODE": "unknown"}, {"MAX_FRAMES": -1}, {"PROGRESS_EVERY": -1})
        for changes in invalid:
            with self.subTest(changes=changes), patch.multiple(blink_pipeline, **changes):
                with self.assertRaises(ValueError):
                    blink_pipeline._validate_filter_settings()

    def test_cli_nonfinite_setting_is_reported_before_run(self):
        # main mutates module defaults; restore every such global after checking
        # argparse's user-facing failure path rather than polluting other tests.
        names = ("LEFT_VIDEO_PATH", "RIGHT_VIDEO_PATH", "LEFT_CALIBRATION_PATH", "RIGHT_CALIBRATION_PATH",
                 "LEFT_CAMERA_ID", "RIGHT_CAMERA_ID", "REQUIRE_CALIBRATION_IDENTITY", "HEADLESS",
                 "RESULTS_PATH", "MAX_FRAMES", "DEVICE", "FILTER_MODE", "MIN_CONFIDENCE", "ROI_MODE",
                 "RECOVERY_CONFIDENCE", "RECOVERY_DURATION_S", "RUN_SUMMARY_PATH", "PROGRESS_EVERY",
                 "LEFT_FRAME_ROTATION", "RIGHT_FRAME_ROTATION", "LEFT_CALIBRATION_ROTATION", "RIGHT_CALIBRATION_ROTATION")
        saved = {name: getattr(blink_pipeline, name) for name in names}
        with patch.multiple(blink_pipeline, **saved), patch("sys.argv", ["eye_pipeline.py", "--min-confidence", "nan"]), \
                patch.object(blink_pipeline, "run_pipeline") as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                blink_pipeline.main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_source_fingerprint_changes_with_core_algorithm_source(self):
        names = ("eye_pipeline.py", "pupil_detection.py", "pupil_quality.py", "pupil_roi.py", "calibration.py", "eye_model_estimation.py",
                 "feature_output.py", "video_preparation.py")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in names:
                (folder / name).write_text("# fixture\n")
            with patch.object(blink_pipeline, "HERE", folder):
                first = blink_pipeline._source_fingerprint()
                self.assertEqual(first, blink_pipeline._source_fingerprint())
                (folder / "pupil_detection.py").write_text("# changed algorithm\n")
                self.assertNotEqual(first, blink_pipeline._source_fingerprint())
                self.assertEqual(len(first), 64)
                second = blink_pipeline._source_fingerprint()
                (folder / "pupil_quality.py").write_text("# changed quality gate\n")
                self.assertNotEqual(second, blink_pipeline._source_fingerprint())


class BlinkReviewStorageTests(unittest.TestCase):
    def workspace(self, root):
        workspace = object.__new__(ReviewWorkspace)
        workspace.root = Path(root)
        (workspace.root / "labels").mkdir(exist_ok=True)
        (workspace.root / "runs").mkdir(exist_ok=True)
        workspace.lock = threading.RLock()
        workspace.frame_count, workspace.fps = 20, 30.
        workspace.recording_id = "test-recording"
        workspace.split = "development"
        workspace.worker = None
        workspace.videos = {eye:dict(video_sha256=("a" if eye=="left" else "b")*64,
                                   frame_rotation="none",roi=[0,0,20,10]) for eye in ("left","right")}
        return workspace

    def body(self, **changes):
        result = dict(reviewer="student", eye="left",frame_index=0,label="uncertain",
                      reason="Independent reviewer note",revision=0)
        result.update(changes)
        return result

    def test_labels_resume_with_identity_and_no_other_reviewers_leaking(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = self.workspace(folder)
            self.assertEqual(workspace.labels_for("student")["labels"],[])
            saved = workspace.save_label(self.body())
            second_instance = self.workspace(folder)
            self.assertEqual(second_instance.labels_for("student"),saved)
            self.assertEqual(second_instance.labels_for("different")["labels"],[])
            self.assertEqual(saved["labels"][0]["video_sha256"],"a"*64)
            self.assertTrue(saved["labels"][0]["reviewed"])
            self.assertEqual(saved["labels"][0]["label"],"uncertain")

    def test_stale_revision_cannot_overwrite_another_tab_or_process(self):
        with tempfile.TemporaryDirectory() as folder:
            first,second = self.workspace(folder),self.workspace(folder)
            saved=first.save_label(self.body())
            with self.assertRaisesRegex(ValueError,"another tab"):
                second.save_label(self.body(label="usable"))
            self.assertEqual(first.labels_for("student"),saved)
            revised=second.save_label(self.body(label="usable",revision=1))
            self.assertEqual(revised["revision"],2)
            self.assertEqual(revised["labels"][0]["label"],"usable")

    def test_clear_only_removes_selected_reviewer_and_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace=self.workspace(folder)
            workspace.save_label(self.body())
            workspace.save_label(self.body(reviewer="second",label="usable"))
            workspace.save_label(self.body(frame_index=1,revision=1))
            cleared=workspace.save_label(self.body(label=None,revision=2))
            self.assertEqual([r["frame_index"] for r in cleared["labels"]],[1])
            self.assertEqual(len(workspace.labels_for("second")["labels"]),1)

    def test_invalid_inputs_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace=self.workspace(folder)
            saved=workspace.save_label(self.body())
            for update in (dict(frame_index=True),dict(frame_index=-1),dict(frame_index=20),
                           dict(eye="other"),dict(label="blink"),dict(reason="x"*501),
                           dict(reviewer="../outside"),dict(revision=True)):
                with self.subTest(update=update),self.assertRaises(ValueError):
                    workspace.save_label(self.body(**update))
            self.assertEqual(workspace.labels_for("student"),saved)

    def test_atomic_write_failure_preserves_last_complete_document(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"data.json"
            atomic_json(path,{"saved":True})
            with self.assertRaises(ValueError):
                atomic_json(path,{"bad":float("nan")})
            self.assertEqual(json.loads(path.read_text()),{"saved":True})
            self.assertEqual(list(Path(folder).glob("*.tmp")),[])

    def test_invalid_roi_mode_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace=self.workspace(folder)
            with self.assertRaisesRegex(ValueError,"ROI mode"):
                workspace.start_run(dict(min_confidence=.7,recovery_confidence=.7,
                                         recovery_duration=.05,max_frames=0,roi_mode="typo"))
            self.assertEqual(list((Path(folder)/"runs").iterdir()),[])

    def test_held_out_setting_change_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace=self.workspace(folder)
            workspace.split="held_out"
            workspace.config=SimpleNamespace(_source_fingerprint=lambda:"source-a")
            original=dict(min_confidence=.6,recovery_confidence=.7,recovery_duration=.05,max_frames=0)
            atomic_json(Path(folder)/"evaluation_plan.json",dict(settings=original,source_fingerprint="source-a"))
            with self.assertRaisesRegex(ValueError,"frozen"):
                workspace.start_run(dict(original,recovery_duration=.1))
            with self.assertRaisesRegex(ValueError,"complete recording"):
                workspace.start_run(dict(original,max_frames=10))
            workspace.config=SimpleNamespace(_source_fingerprint=lambda:"source-b")
            with self.assertRaisesRegex(ValueError,"frozen"):
                workspace.start_run(original)
            self.assertEqual(list((Path(folder)/"runs").iterdir()),[])



class BoundaryQualityTests(unittest.TestCase):
    """Isolate crop clipping, coverage and valid dark-pupil controls."""

    def fixture(self):
        image = np.full((200, 320), 150, np.uint8)
        cv2.ellipse(image, ((160, 100), (80, 70), 20), 20, -1)
        mask = (image < 50).astype(np.uint8)
        contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        return image, contour, cv2.fitEllipse(contour)

    def test_clear_rotated_pupil_and_internal_glints_pass(self):
        from pupil_detection import pupil_boundary_quality
        image, contour, ellipse = self.fixture()
        cv2.circle(image, (157, 103), 3, 255, -1)
        original = image.copy()
        self.assertIsNone(pupil_boundary_quality(image, contour, ellipse))
        np.testing.assert_array_equal(image, original)

    def test_contour_on_any_border_is_rejected(self):
        from pupil_detection import pupil_boundary_quality
        image, contour, ellipse = self.fixture()
        for point in ((0, 100), (319, 100), (160, 0), (160, 199)):
            clipped = contour.copy()
            clipped[0, 0] = point
            self.assertIn('clipped', pupil_boundary_quality(image, clipped, ellipse))

    def test_dark_upper_surrounding_arc_is_coverage(self):
        from pupil_detection import pupil_boundary_quality
        image, contour, ellipse = self.fixture()
        image[:90] = 20
        self.assertIn('coverage', pupil_boundary_quality(image, contour, ellipse))

    def test_brightness_asymmetry_alone_is_warning(self):
        from pupil_detection import pupil_boundary_quality
        image, contour, ellipse = self.fixture()
        upper_pupil = (image < 50) & (np.indices(image.shape)[0] < 90)
        image[upper_pupil] = 65
        from pupil_detection import pupil_boundary_evidence
        self.assertIsNone(pupil_boundary_quality(image, contour, ellipse, policy="experimental"))
        self.assertIsNotNone(pupil_boundary_quality(image, contour, ellipse))
        self.assertEqual(pupil_boundary_evidence(image, contour, ellipse).status, "warning")

    def test_uniform_dark_image_does_not_imply_upper_coverage(self):
        from pupil_detection import pupil_boundary_quality
        image, contour, ellipse = self.fixture()
        image[:] = 20
        self.assertIsNone(pupil_boundary_quality(image, contour, ellipse))

    def test_boundary_gate_is_bypassed_only_in_baseline(self):
        from dataclasses import replace
        observation = replace(pupil_observation(), boundary_rejection='pupil contour clipped by ROI boundary')
        self.assertFalse(assess_pupil_quality(observation).allow_model_update)
        with patch.object(blink_pipeline, 'FILTER_MODE', 'baseline'):
            self.assertTrue(assess_pupil_quality(blink_pipeline._filter_observation(observation)).allow_model_update)
        with patch.object(blink_pipeline, 'FILTER_MODE', 'filtered'):
            self.assertFalse(assess_pupil_quality(blink_pipeline._filter_observation(observation)).allow_model_update)



class RobotCommandGateTests(unittest.TestCase):
    def fixture(self, **options):
        from pupil_quality import RobotCommandGate, BoundaryEvidence
        from dataclasses import replace
        clock = Mock(return_value=10.)
        gate = RobotCommandGate(live=True, mapping_verified=True, clock=clock, **options)
        pupil = replace(pupil_observation(), boundary_evidence=BoundaryEvidence('clear', 'clear'))
        quality = FrameQualityDecision(True, 'passed', state='usable')
        estimate = SimpleNamespace(ready=True, gaze_direction_camera=(0.,0.,1.),
            calibration_identity_status='matched', model_diagnostics=SimpleNamespace(range_status='passed'),
            model_stability=SimpleNamespace(status='stable'))
        eyes = ((pupil,quality,estimate),(pupil,quality,copy.deepcopy(estimate)))
        args = dict(filter_mode='filtered',capture_times=(9.99,9.99),target_robot=(1,2,3))
        return gate,clock,eyes,args

    def test_valid_pair_dispatches_with_expiry(self):
        gate,clock,eyes,args = self.fixture();send=Mock();invalidate=Mock()
        decision=gate.dispatch(eyes,send=send,invalidate=invalidate,**args)
        self.assertTrue(decision.allowed);send.assert_called_once_with(decision);invalidate.assert_not_called()
        self.assertAlmostEqual(decision.expires_at_monotonic_s,10.09)

    def test_offline_and_baseline_never_send(self):
        for case in ('offline','baseline','mapping'):
            gate,clock,eyes,args=self.fixture()
            if case=='offline':gate.live=False
            elif case=='mapping':gate.mapping_verified=False
            else:args['filter_mode']='baseline'
            send=Mock();invalidate=Mock();result=gate.dispatch(eyes,send=send,invalidate=invalidate,**args)
            self.assertFalse(result.allowed);send.assert_not_called();invalidate.assert_called_once()

    def test_rejected_or_recovering_either_eye_blocks(self):
        for index in (0,1):
            for state in ('rejected','recovering'):
                gate,clock,eyes,args=self.fixture();eyes=list(eyes)
                pupil,_,estimate=eyes[index];eyes[index]=(pupil,FrameQualityDecision(False,state,state),estimate)
                self.assertFalse(gate.evaluate(eyes,**args).allowed)

    def test_unknown_or_warning_image_blocks_robot_but_not_learning(self):
        from dataclasses import replace
        from pupil_quality import BoundaryEvidence
        for state in ('unknown','warning'):
            gate,clock,eyes,args=self.fixture();pupil,quality,estimate=eyes[0]
            pupil=replace(pupil,boundary_evidence=BoundaryEvidence(state,state))
            self.assertTrue(assess_pupil_quality(pupil).allow_model_update)
            self.assertFalse(gate.evaluate(((pupil,quality,estimate),eyes[1]),**args).allowed)

    def test_missing_eyelid_or_model_checks_block(self):
        from dataclasses import replace
        for field in ('eyelid','geometry','ranges','stability','calibration','direction'):
            gate,clock,eyes,args=self.fixture();pupil,quality,estimate=eyes[0]
            if field=='eyelid':pupil=replace(pupil,eyelid=None)
            elif field=='geometry':estimate.ready=False
            elif field=='ranges':estimate.model_diagnostics.range_status='failed'
            elif field=='stability':estimate.model_stability.status='warming_up'
            elif field=='calibration':estimate.calibration_identity_status='unverified'
            else:estimate.gaze_direction_camera=(float('nan'),0,1)
            self.assertFalse(gate.evaluate(((pupil,quality,estimate),eyes[1]),**args).allowed)

    def test_stale_missing_future_unsynchronized_timestamps_block(self):
        for times in (None,(9.,9.),(11.,11.),(9.91,9.99),(float('nan'),9.99)):
            gate,clock,eyes,args=self.fixture();args['capture_times']=times
            self.assertFalse(gate.evaluate(eyes,**args).allowed)

    def test_send_time_recheck_and_replay_prevention(self):
        gate,clock,eyes,args=self.fixture();self.assertTrue(gate.evaluate(eyes,**args).allowed)
        clock.return_value=10.2;send=Mock();invalidate=Mock()
        gate.dispatch(eyes,send=send,invalidate=invalidate,**args);send.assert_not_called()
        clock.return_value=10.;gate.dispatch(eyes,send=send,invalidate=invalidate,**args)
        result=gate.dispatch(eyes,send=send,invalidate=invalidate,**args)
        self.assertFalse(result.allowed);self.assertEqual(send.call_count,1)

    def test_sender_failure_invalidates_and_cannot_replay(self):
        gate,clock,eyes,args=self.fixture();invalidate=Mock()
        with self.assertRaises(RuntimeError):
            gate.dispatch(eyes,send=Mock(side_effect=RuntimeError('offline')),invalidate=invalidate,**args)
        invalidate.assert_called_once();self.assertFalse(gate.evaluate(eyes,**args).allowed)

    def test_missing_or_nonfinite_robot_target_blocks(self):
        for target in (None,(1,2),(1,float('inf'),3)):
            gate,clock,eyes,args=self.fixture();args['target_robot']=target
            self.assertFalse(gate.evaluate(eyes,**args).allowed)


class EarlyImageGateTests(unittest.TestCase):
    def test_rejection_and_recovery_precede_calibration(self):
        tracker=TemporalQualityTracker();calibration=Mock();model=Mock()
        with patch.object(blink_pipeline,'FILTER_MODE','filtered'):
            for i,obs in enumerate((pupil_observation(ellipse=None),pupil_observation(),pupil_observation())):
                ellipse,decision=blink_pipeline.prepare_eye_update(obs,i/30,calibration,model,tracker)
                self.assertFalse(decision.allow_model_update);self.assertIsNone(ellipse)
        calibration.undistort_ellipse.assert_not_called();model.assess_input_geometry.assert_not_called()
        model.update.assert_not_called()

    def test_robot_decision_occurs_after_both_model_results(self):
        from pupil_quality import RobotCommandDecision
        events=[];frame=np.zeros((32,32,3),np.uint8)
        videos=[Mock(read=Mock(return_value=(True,frame))) for _ in range(2)]
        observations=[pupil_observation(ellipse=None),pupil_observation()]
        def detect(*args):events.append('detect');return observations.pop(0)
        detector=Mock(detect=detect)
        calibrations=[Mock(undistort_ellipse=lambda e:e) for _ in range(2)]
        models=[]
        for side in ('left','right'):
            model=Mock()
            model.assess_input_geometry.return_value=Pye3DInputDecision(True,'passed')
            def update(*args,side=side):
                events.append(side+'_update')
                return diagnostic_estimator()._result(diagnostic_raw_result(),.9,.1)
            def skip(*args,side=side):
                events.append(side+'_skip')
                return diagnostic_estimator()._empty(.9,'skipped')
            model.update.side_effect=update;model.skip_update.side_effect=skip;models.append(model)
        def gate(eyes,**kwargs):
            events.append('robot_gate')
            self.assertFalse(eyes[0][1].allow_model_update)
            return RobotCommandDecision(False,'offline')
        with patch.multiple(blink_pipeline,MAX_FRAMES=1,TEXT_OUTPUT=False,FILTER_MODE='filtered'), \
             patch.object(blink_pipeline,'RobotCommandGate') as factory:
            factory.return_value.evaluate.side_effect=gate
            blink_pipeline.process_frame_loop(*videos,30,'none','none',*calibrations,detector,*models,headless=True)
        self.assertEqual(events,['detect','detect','left_skip','right_update','robot_gate'])
        models[0].update.assert_not_called()

    def test_geometry_failure_restarts_recovery_without_duplicate_timestamp(self):
        tracker=TemporalQualityTracker();calibration=Mock();model=Mock()
        model.assess_input_geometry.return_value=Pye3DInputDecision(False,'invalid geometry')
        with patch.object(blink_pipeline,'FILTER_MODE','filtered'):
            _,decision=blink_pipeline.prepare_eye_update(pupil_observation(),0.,calibration,model,tracker)
            self.assertEqual(decision.state,'rejected')
            _,decision=blink_pipeline.prepare_eye_update(pupil_observation(),1/30,calibration,model,tracker)
            self.assertEqual(decision.state,'recovering')
        self.assertEqual(calibration.undistort_ellipse.call_count,1)


class BorderEvidenceTests(unittest.TestCase):
    def fixture(self, dark_corridor):
        image=np.full((120,180),160,np.uint8)
        cv2.ellipse(image,((90,35),(60,60),0),20,-1)
        mask=(image<50).astype(np.uint8)
        contour=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0][0]
        ellipse=cv2.fitEllipse(contour)
        if dark_corridor:image[:12,75:106]=20
        return image,contour,ellipse

    def test_raw_border_connection_catches_retreated_contour(self):
        from pupil_detection import pupil_boundary_evidence
        image,contour,ellipse=self.fixture(True)
        self.assertGreater(contour[:,:,1].min(),0)
        evidence=pupil_boundary_evidence(image,contour,ellipse)
        self.assertEqual(evidence.status,'rejected');self.assertIn('boundary',evidence.reason)

    def test_clear_iris_separation_near_border_does_not_reject(self):
        from pupil_detection import pupil_boundary_evidence
        image,contour,ellipse=self.fixture(False)
        self.assertNotEqual(pupil_boundary_evidence(image,contour,ellipse).status,'rejected')

    def test_missing_samples_are_reported_without_zero_padding(self):
        from pupil_detection import pupil_boundary_evidence
        image,contour,ellipse=self.fixture(False)
        evidence=pupil_boundary_evidence(image,contour,((90,25),(60,60),0))
        self.assertLess(evidence.valid_fraction,1.)
        self.assertTrue(evidence.lower_contrast is None or np.isfinite(evidence.lower_contrast))


class RawObservationExportTests(unittest.TestCase):
    """Rejected input remains inspectable without turning into model output."""
    def record(self, pupil, include=True):
        from feature_output import frame_record
        decision=assess_pupil_quality(pupil)
        estimate=diagnostic_estimator()._empty(.9,"skipped")
        return frame_record("left",3,.1,estimate,decision,{},
                            **({"pupil":pupil} if include else {}))

    def test_rejected_boundary_retains_measurement_without_ready_geometry(self):
        from dataclasses import replace
        pupil=replace(pupil_observation(),boundary_rejection="possible upper pupil contamination")
        record=self.record(pupil)
        self.assertEqual(record["model_input"],"skipped")
        self.assertFalse(record["ready"])
        self.assertIsNone(record["gaze_direction_camera"])
        self.assertEqual(record["pupil_observation"]["boundary_rejection"],pupil.boundary_rejection)
        self.assertIsNotNone(record["pupil_observation"]["ellipse"])
        json.dumps(record,allow_nan=False)

    def test_missing_and_nonfinite_inputs_are_json_safe_without_mutation(self):
        from dataclasses import replace
        for pupil in (pupil_observation(ellipse=None),
                      replace(pupil_observation(),confidence=float("nan")),
                      replace(pupil_observation(),ellipse=((float("inf"),10),(5,7),0))):
            record=self.record(pupil)
            self.assertEqual(record["model_input"],"skipped")
            json.dumps(record,allow_nan=False)
        self.assertTrue(math.isinf(pupil.ellipse[0][0]))
        self.assertIsNone(record["pupil_observation"]["ellipse"][0][0])

    def test_old_caller_can_omit_raw_observation(self):
        record=self.record(pupil_observation(),include=False)
        self.assertNotIn("pupil_observation",record)


class PupilRoiTests(unittest.TestCase):
    def observation(self, center=(320,240), confidence=.95):
        return PupilObservation((center,(60,80),30),False,confidence)

    def tracker(self):
        from pupil_roi import PupilRoiTracker
        return PupilRoiTracker((0,0,640,480))

    def test_rotated_ellipse_bounds_use_diameters(self):
        from pupil_roi import ellipse_bounds, ellipse_fits
        bounds=ellipse_bounds(((100,100),(80,20),90))
        np.testing.assert_allclose(bounds,(90,60,110,140))
        self.assertTrue(ellipse_fits(((100,100),(80,20),90),(89,59,23,83),1))
        self.assertFalse(ellipse_fits(((100,100),(80,20),90),(90,60,20,80)))

    def test_first_search_broad_then_narrow_with_broad_context(self):
        tracker=self.tracker();detector=Mock(detect=Mock(return_value=self.observation()))
        frame=np.zeros((480,640),np.uint8)
        _,first=tracker.detect(detector,frame,0)
        pupil,second=tracker.detect(detector,frame,1/30)
        self.assertEqual(first['area_fraction'],1)
        self.assertLess(second['area_fraction'],1)
        self.assertEqual(detector.detect.call_args.kwargs['evidence_roi'],(0,0,640,480))
        self.assertEqual(pupil.ellipse,self.observation().ellipse)

    def test_crop_failure_retries_once_without_reusing_previous_pupil(self):
        tracker=self.tracker();frame=np.zeros((480,640),np.uint8)
        missing=PupilObservation(None,True,0.)
        detector=Mock(detect=Mock(side_effect=[self.observation(),missing,missing]))
        tracker.detect(detector,frame,0)
        pupil,info=tracker.detect(detector,frame,1/30)
        self.assertIsNone(pupil.ellipse)
        self.assertEqual(info['network_calls'],2)
        self.assertEqual(info['detection_roi'],[0,0,640,480])
        self.assertIsNone(tracker.last)

    def test_border_measurement_triggers_broad_retry(self):
        tracker=self.tracker();frame=np.zeros((480,640),np.uint8)
        detector=Mock(detect=Mock(return_value=self.observation()))
        tracker.detect(detector,frame,0)
        far=self.observation((50,50));detector.detect.side_effect=[far,self.observation()]
        _,info=tracker.detect(detector,frame,1/30)
        self.assertIn('edge',info['fallback_reason'])
        self.assertEqual(info['network_calls'],2)

    def test_blink_recovery_remains_broad_until_good_time_has_elapsed(self):
        tracker=self.tracker();frame=np.zeros((480,640),np.uint8)
        detector=Mock(detect=Mock(return_value=PupilObservation(None,True,0.)))
        tracker.detect(detector,frame,0)
        detector.detect.return_value=self.observation()
        for t in (1/30,2/30):
            _,info=tracker.detect(detector,frame,t)
            self.assertEqual(info['area_fraction'],1)
        _,info=tracker.detect(detector,frame,3/30)
        self.assertLess(info['area_fraction'],1)

    def test_timestamp_gap_and_periodic_refresh_reacquire_broadly(self):
        frame=np.zeros((480,640),np.uint8);detector=Mock(detect=Mock(return_value=self.observation()))
        for timestamp in (.2,-1):
            tracker=self.tracker();tracker.detect(detector,frame,0)
            _,info=tracker.detect(detector,frame,timestamp)
            self.assertEqual(info['area_fraction'],1)
        tracker=self.tracker()
        for i in range(16):pupil,info=tracker.detect(detector,frame,i/30)
        self.assertEqual(info['area_fraction'],1)

    def test_tracking_is_per_eye_and_does_not_depend_on_filter_mode(self):
        frame=np.zeros((480,640),np.uint8);detector=Mock(detect=Mock(return_value=self.observation()))
        left,right=self.tracker(),self.tracker()
        left.detect(detector,frame,0)
        _,info=right.detect(detector,frame,1/30)
        self.assertEqual(info['area_fraction'],1)
        records=[]
        for mode in ('baseline','filtered'):
            tracker=self.tracker()
            with patch.object(blink_pipeline,'FILTER_MODE',mode):
                records.append([tracker.detect(detector,frame,i/30)[1] for i in range(4)])
        self.assertEqual(*records)

    def test_roi_and_timestamp_validation(self):
        from pupil_roi import PupilRoiTracker
        with self.assertRaises(ValueError):PupilRoiTracker((-1,0,100,100))
        tracker=self.tracker()
        with self.assertRaises(ValueError):tracker.detect(Mock(),np.zeros((32,32)),0)
        with self.assertRaises(ValueError):tracker.detect(Mock(),np.zeros((480,640)),float('nan'))

    def test_reference_area_prevents_crop_size_from_changing_contour_limit(self):
        from pupil_detection import select_pupil_contour
        mask=np.zeros((100,100),np.uint8);cv2.circle(mask,(50,50),40,255,-1)
        self.assertIsNone(select_pupil_contour(mask))
        self.assertIsNotNone(select_pupil_contour(mask,reference_area=320*200))

    def test_detector_translates_measurement_once_and_preserves_context(self):
        from pupil_detection import PupilDetector
        from pupil_quality import BoundaryEvidence,EyelidObservation
        import torch
        detector=PupilDetector(torch.nn.Identity(),'cpu',.5)
        frame=np.zeros((200,320),np.uint8);prob=np.zeros((80,100),np.float32)
        cv2.ellipse(prob,((50,40),(30,20),0),1,-1)
        with patch('pupil_detection.generate_probability_map',return_value=prob), \
             patch('pupil_detection.detect_eyelid_closure',return_value=EyelidObservation()) as lid, \
             patch('pupil_detection.pupil_boundary_evidence',return_value=BoundaryEvidence()), \
             patch('pupil_detection.existing_boundary_quality',return_value=None):
            pupil=detector.detect(frame,(100,60,100,80),evidence_roi=(0,0,320,200))
        np.testing.assert_allclose(pupil.ellipse[0],(150,100),atol=.2)
        self.assertEqual(lid.call_args.args[0].shape,(200,320))
        np.testing.assert_allclose(lid.call_args.args[1][0],pupil.ellipse[0])
        with self.assertRaises(ValueError):detector.detect(frame,(100,60,100,80),evidence_roi=(120,60,80,80))

if __name__ == '__main__':
    unittest.main()
