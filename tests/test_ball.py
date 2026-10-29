from __future__ import annotations

import sys
import unittest
from pathlib import Path
from collections import deque
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import BallModel, BallObjectDetector, BallTracker, ModelBank, YoloBallModel, ball_candidates
from engine import draw_frame, summarize
from court import LENGTH, WIDTH
from ball_motion import ActiveBallSelector, MotionBallDetector


class BallTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((360, 640, 3), np.uint8)

    def candidate(self, x, y, intensity=220):
        return {"x": x, "y": y, "intensity": intensity}

    def test_components_reject_long_lines_and_keep_ball(self):
        intensity = np.zeros((360, 640), np.uint8)
        intensity[100:103, 200:203] = 140
        intensity[10, 10:60] = 255
        candidates = ball_candidates(intensity)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0], self.candidate(201, 101, 140))

    def test_unexpected_heatmap_shape_is_an_error(self):
        with self.assertRaises(ValueError):
            ball_candidates(np.zeros((180, 320), np.uint8))

    def test_single_scale_detector_preserves_its_candidates(self):
        detector = BallObjectDetector.__new__(BallObjectDetector)
        detector.size, detector.multiscale = 640, False
        values = [self.candidate(100, 100)]
        with patch.object(detector, "_candidates", return_value=values) as detect:
            self.assertIs(detector.candidates(self.image), values)
            detect.assert_called_once_with(self.image, 640)

    def test_multiscale_fuses_duplicate_centers_before_temporal_confirmation(self):
        detector = BallObjectDetector.__new__(BallObjectDetector)
        detector.size, detector.multiscale = 640, True
        first = [self.candidate(100, 100, 190), self.candidate(200, 200, 230)]
        second = [self.candidate(101, 100, 220), self.candidate(300, 300, 200)]
        with patch.object(detector, "_candidates", side_effect=[first, second]) as detect:
            values = detector.candidates(self.image)
            self.assertEqual([call.args[1] for call in detect.call_args_list], [640, 960])
        self.assertEqual(values, [
            self.candidate(200, 200, 230), self.candidate(101, 100, 220), self.candidate(300, 300, 200),
        ])

    def test_model_bank_keeps_light_and_multiscale_backends_distinct(self):
        bank = ModelBank()
        with patch("inference.YoloBallModel") as factory:
            bank.ball_detector("yolo_multiscale")
            factory.assert_called_once_with(multiscale=True)
            bank.ball_detector("yolo")
            self.assertEqual(factory.call_count, 2)
            factory.assert_called_with(multiscale=False)

    def test_velocity_selects_forward_ball_not_stronger_old_position(self):
        tracker = BallTracker()
        tracker.update(self.image, 0, [self.candidate(100, 100)])
        tracker.update(self.image, 0.1, [self.candidate(160, 100)])
        point = tracker.update(self.image, 0.2, [
            self.candidate(160, 100, 250), self.candidate(220, 100, 200),
        ])
        self.assertAlmostEqual(point["x"], 220)

    def test_rejects_distant_component_then_reacquires_after_gap(self):
        tracker = BallTracker()
        tracker.update(self.image, 0, [self.candidate(100, 100)])
        self.assertIsNone(tracker.update(self.image, 0.1, [self.candidate(500, 300)]))
        self.assertIsNotNone(tracker.update(self.image, 0.4, [self.candidate(500, 300)]))

    def test_no_neural_candidate_does_not_fall_back_to_old_ball(self):
        tracker = BallTracker()
        tracker.update(self.image, 0, [self.candidate(100, 100)])
        self.assertIsNone(tracker.update(self.image, 1 / 30, []))
        self.assertIsNone(tracker.update(self.image, 2 / 30, None))

    def test_blank_image_cannot_supply_optical_flow(self):
        tracker = BallTracker()
        tracker.update(self.image, 0, [self.candidate(100, 100)])
        self.assertIsNone(tracker.update(self.image, 1 / 30, None))

    def test_verified_translation_supplies_intermediate_measurement(self):
        rng = np.random.default_rng(42)
        image = np.zeros_like(self.image)
        image[90:111, 90:111] = rng.integers(0, 256, (21, 21, 3), dtype=np.uint8)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, 2]]), (640, 360))
        tracker = BallTracker()
        tracker.update(image, 0, [self.candidate(100, 100)])
        point = tracker.update(moved, 1 / 30, None)
        self.assertIsNotNone(point)
        np.testing.assert_allclose([point["x"], point["y"]], [103, 102], atol=0.3)
        self.assertEqual(point["method"], "bidirectional optical flow")
        self.assertIsNone(point["intensity"])

    def test_flow_is_not_extended_without_neural_confirmation(self):
        tracker = BallTracker()
        tracker.update(self.image, 0, [self.candidate(100, 100)])
        self.assertIsNone(tracker.update(self.image, 0.1, None))

    def test_duplicate_timestamp_is_rejected(self):
        tracker = BallTracker()
        tracker.update(self.image, 0, [])
        with self.assertRaises(ValueError):
            tracker.update(self.image, 0, [])

    def test_ball_coverage_uses_all_frames_and_reports_methods(self):
        sample = {"camera_valid": True, "players": [
            {"side": "north", "pitch": [5, -1], "confidence": 0.8},
        ], "ball": None}
        frames = [
            {"ball": {"method": "TrackNet + temporal association"}},
            {"ball": {"method": "bidirectional optical flow"}},
            {"ball": None},
        ]
        summary = summarize([sample], "complete", 0.1, 1, frames)
        self.assertAlmostEqual(summary["ball_coverage"], 2 / 3)
        self.assertEqual(summary["ball_samples"], 3)
        self.assertEqual(summary["ball_methods"]["bidirectional optical flow"], 1)

    def test_minimap_shows_player_behind_baseline_without_clamping(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        sample = {"camera_valid": True, "matrix": np.eye(3).tolist(), "players": [
            {"side": "north", "pitch": [WIDTH / 2, -2], "box": [100, 100, 120, 150]},
        ], "ball": None, "time": 0}
        draw_frame(frame, sample, sample, 0, [sample], 0, 0)
        scale = round(1280 * 0.13) / (WIDTH + 8)
        x = 1280 - round(1280 * 0.13) - 22 + round((WIDTH / 2 + 4) * scale)
        y = 57 + round(5 * scale)
        np.testing.assert_array_equal(frame[y, x], [210, 224, 92])


class MotionBallTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(108)
        gray = rng.integers(30, 120, (360, 640), dtype=np.uint8)
        self.background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_static_spare_ball_is_not_a_moving_ball_candidate(self):
        image = self.background.copy()
        cv2.circle(image, (250, 140), 4, (0, 255, 255), -1)
        detector = MotionBallDetector()
        self.assertEqual(detector.candidates(image), [])
        self.assertEqual(detector.candidates(image), [])

    def test_moving_yellow_ball_is_retained(self):
        before, after = self.background.copy(), self.background.copy()
        cv2.circle(before, (250, 140), 4, (0, 255, 255), -1)
        cv2.circle(after, (270, 150), 4, (0, 255, 255), -1)
        detector = MotionBallDetector()
        detector.candidates(before)
        candidates = detector.candidates(after)
        self.assertEqual(len(candidates), 1)
        np.testing.assert_allclose([candidates[0]["x"], candidates[0]["y"]], [270, 150], atol=0.1)

    def test_camera_pan_does_not_make_static_spare_ball_active(self):
        before = self.background.copy()
        cv2.circle(before, (250, 140), 4, (0, 255, 255), -1)
        after = cv2.warpAffine(before, np.float32([[1, 0, 7], [0, 1, 3]]), (640, 360))
        detector = MotionBallDetector()
        detector.candidates(before)
        self.assertEqual(detector.candidates(after), [])


class YoloBallTests(unittest.TestCase):
    def moving_model(self, sequence):
        model = YoloBallModel.__new__(YoloBallModel)
        model.detector = Mock()
        model.detector.candidates.side_effect = sequence
        model.reset()
        model.motion = Mock()
        model.motion.candidates.return_value = []
        model.motion.affine = np.float64([[1, 0, 0], [0, 1, 0]])
        return model

    def test_stationary_yolo_candidate_is_not_accepted_as_live_ball(self):
        model = YoloBallModel.__new__(YoloBallModel)
        model.detector = Mock()
        model.detector.candidates.return_value = [{"x": 100, "y": 120, "intensity": 240}]
        model.reset()
        image = np.random.default_rng(42).integers(0, 150, (360, 640, 3), dtype=np.uint8)
        frames = deque([image], maxlen=3)
        points = [model.detect(frames, index / 30) for index in range(12)]
        self.assertTrue(all(point is None for point in points))

    def test_renderer_never_connects_different_ball_identities(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        sample = {"camera_valid": False, "players": [], "ball": None, "time": 0}
        series = [
            {"time": 0, "ball": {"x": 100, "y": 300, "track_id": 1, "continuous": False}},
            {"time": 1 / 30, "ball": {"x": 300, "y": 300, "track_id": 2, "continuous": False}},
        ]
        draw_frame(frame, sample, sample, 0, [sample], 0, 1 / 30, ball_samples=series, frame_index=1)
        np.testing.assert_array_equal(frame[300, 200], [0, 0, 0])

    def test_yolo_reacquires_stronger_ball_instead_of_locking_on_a_spare(self):
        spare = {"x": 100, "y": 120, "intensity": 0.4 * 255}
        sequence = [[spare.copy()], [spare.copy()]] + [
            [spare.copy(), {"x": x, "y": 250, "intensity": 0.8 * 255}] for x in (500, 510, 520)
        ]
        model = self.moving_model(sequence)
        frames = deque([np.zeros((360, 640, 3), np.uint8)], maxlen=3)
        for index in range(4):
            self.assertIsNone(model.detect(frames, index / 30))
        point = model.detect(frames, 4 / 30)
        self.assertAlmostEqual(point["x"], 520)
        self.assertEqual(len(model.tracker.history), 1)

    def test_yolo_uses_continuity_to_break_near_equal_detector_scores(self):
        sequence = [[{"x": x, "y": 120, "intensity": 0.74 * 255}] for x in (100, 110, 120)]
        sequence.append([{"x": 130, "y": 120, "intensity": 0.74 * 255},
                         {"x": 500, "y": 250, "intensity": 0.95 * 255}])
        model = self.moving_model(sequence)
        frames = deque([np.zeros((360, 640, 3), np.uint8)], maxlen=3)
        for index in range(3):
            model.detect(frames, index / 30)
        point = model.detect(frames, 3 / 30)
        self.assertAlmostEqual(point["x"], 130)
        self.assertTrue(point["continuous"])

    def test_letterbox_rgb_decode_and_nms(self):
        detector = BallObjectDetector.__new__(BallObjectDetector)
        detector.size, detector.confidence, detector.input_name = 640, 0.15, "images"
        detector.multiscale = False
        detector.net = Mock()
        detector.net.run.return_value = [np.array([[
            [320, 321, 320], [320, 320, 80], [4, 4, 4], [4, 4, 4], [0.8, 0.7, 0.9],
        ]], np.float32)]
        image = np.full((720, 1280, 3), [10, 20, 30], np.uint8)
        candidates = detector.candidates(image)
        self.assertEqual(len(candidates), 1)
        np.testing.assert_allclose([candidates[0]["x"], candidates[0]["y"]], [320, 180])
        tensor = detector.net.run.call_args.args[1]["images"]
        np.testing.assert_allclose(tensor[0, :, 320, 320], np.array([30, 20, 10]) / 255, atol=1e-7)

    def test_yolo_runs_detector_on_every_frame_even_between_tracknet_slots(self):
        model = self.moving_model([
            [{"x": x, "y": 120, "intensity": 0.8 * 255}] for x in (100, 110, 120)
        ])
        frames = deque([np.zeros((360, 640, 3), np.uint8)], maxlen=3)
        first = model.detect(frames, 0, infer=False)
        second = model.detect(frames, 1 / 30, infer=False)
        third = model.detect(frames, 2 / 30, infer=False)
        self.assertEqual(model.detector.candidates.call_count, 3)
        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(third["method"], "YOLOv8n + temporal association")
        self.assertIsNone(third["intensity"])
        self.assertAlmostEqual(third["detector_score"], 0.8)

    def test_unknown_backend_is_not_silently_substituted(self):
        with self.assertRaises(ValueError):
            ModelBank().ball_detector("invented")


class ActiveBallTests(unittest.TestCase):
    def test_new_static_candidate_after_a_gap_cannot_inherit_a_moving_track(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(5):
            point = selector.update(
                [{"x": 100 + 5 * index, "y": 120 - 3 * index, "intensity": 220}],
                index / 30, affine,
            )
        self.assertIsNotNone(point)
        for index in range(5, 8):
            self.assertIsNone(selector.update([], index / 30, affine))
        for index in range(8, 12):
            point = selector.update(
                [{"x": 150, "y": 190, "intensity": 240}], index / 30, affine,
            )
            self.assertIsNone(point)

    def test_fast_tracknet_cadence_keeps_its_identity_across_a_gap(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        identity = None
        for index in (0, 3, 6, 9, 13):
            point = selector.update(
                [{"x": 70 + 12 * index, "y": 90, "intensity": 220}], index / 30, affine,
            )
            if index >= 6:
                self.assertIsNotNone(point)
                if identity is None:
                    identity = point["track_id"]
                self.assertEqual(point["track_id"], identity)

    def test_large_direction_change_requires_new_motion_confirmation(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(4):
            point = selector.update(
                [{"x": 100 + 15 * index, "y": 100, "intensity": 220}], index / 30, affine,
            )
        identity = point["track_id"]
        for index, position in enumerate(((110, 125), (90, 140), (70, 155)), start=4):
            point = selector.update(
                [{"x": position[0], "y": position[1], "intensity": 220}], index / 30, affine,
            )
            if index < 6:
                self.assertIsNone(point)
        self.assertIsNotNone(point)
        self.assertNotEqual(point["track_id"], identity)

    def test_old_outlier_does_not_hide_a_new_coherent_arc(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        positions = [
            (100, 110), (100, 85), (102, 85), (104, 85.5), (106, 86), None,
            (110, 89), (112, 91), (114, 94), (116, 97), (118, 101),
        ]
        for index, position in enumerate(positions):
            candidates = [] if position is None else [
                {"x": position[0], "y": position[1], "intensity": 220},
            ]
            point = selector.update(candidates, index / 30, affine)
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point["x"], 118)
        self.assertAlmostEqual(point["y"], 101)

    def test_tracknet_also_requires_confirmed_motion(self):
        model = BallModel.__new__(BallModel)
        maps = []
        for x in (100, 120, 140):
            image = np.zeros((360, 640), np.uint8)
            image[100:103, x:x + 3] = 220
            maps.append(image)
        model.heatmap = Mock(side_effect=maps)
        model.reset()
        model.motion = Mock()
        model.motion.candidates.return_value = []
        model.motion.affine = np.float64([[1, 0, 0], [0, 1, 0]])
        frames = deque(maxlen=3)
        for index in range(9):
            frames.append(np.zeros((360, 640, 3), np.uint8))
            point = model.detect(frames, index / 30, infer=index % 3 == 2)
            if index < 8:
                self.assertIsNone(point)
        self.assertIsNotNone(point)
        self.assertEqual(model.heatmap.call_count, 3)
        self.assertEqual(point["method"], "TrackNet + temporal association")
        self.assertIn("track_id", point)

    def test_camera_pan_does_not_confirm_a_stationary_spare(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 7], [0, 1, 3]])
        for index in range(15):
            point = selector.update([{"x": 100 + 7 * index, "y": 100 + 3 * index, "intensity": 250}],
                                    index / 30, affine)
            self.assertIsNone(point)

    def test_camera_loss_requires_a_new_confirmation(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(3):
            point = selector.update([{"x": 100 + 10 * index, "y": 100, "intensity": 220}],
                                    index / 30, affine)
        self.assertIsNotNone(point)
        previous_id = point["track_id"]
        self.assertIsNone(selector.update([{"x": 130, "y": 100, "intensity": 220}], .1, None))
        for index in range(4, 6):
            point = selector.update([{"x": 100 + 10 * index, "y": 100, "intensity": 220}],
                                    index / 30, affine)
        self.assertIsNotNone(point)
        self.assertNotEqual(point["track_id"], previous_id)

    def test_expired_spare_track_cannot_inherit_the_active_ball_motion(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(11):
            candidates = [{"x": 100 + 6 * index, "y": 100, "intensity": 220}]
            if index < 3:
                candidates.append({"x": 160, "y": 100, "intensity": 250})
            point = selector.update(candidates, index / 30, affine)
            if index == 9:
                self.assertIsNotNone(point)
        self.assertIsNone(point)
        point = selector.update([{"x": 166, "y": 100, "intensity": 220}], 11 / 30, affine)
        self.assertIsNotNone(point)

    def test_stationary_memory_moves_with_camera_and_expires(self):
        selector = ActiveBallSelector()
        still = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(3):
            selector.update([{"x": 160, "y": 100, "intensity": 250}], index / 30, still)
        pan = np.float64([[1, 0, 5], [0, 1, 2]])
        selector.update([], .1, pan)
        np.testing.assert_allclose(selector.stationary[0][1], [165, 102])
        selector.update([], 5.2, still)
        self.assertEqual(selector.stationary, [])

    def test_stationary_evidence_is_independent_of_live_track_assignment(self):
        selector = ActiveBallSelector()
        affine = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(11):
            candidates = [{"x": 100 + 6 * index, "y": 100, "intensity": 220}]
            if index < 2:
                candidates.append({"x": 160, "y": 100, "intensity": 250})
            point = selector.update(candidates, index / 30, affine)
        self.assertIsNone(point)
        self.assertTrue(selector.stationary)

    def test_camera_loss_discards_stationary_memory(self):
        selector = ActiveBallSelector()
        still = np.float64([[1, 0, 0], [0, 1, 0]])
        for index in range(3):
            selector.update([{"x": 160, "y": 100, "intensity": 250}], index / 30, still)
        self.assertTrue(selector.stationary)
        selector.update([], .1, None)
        self.assertEqual(selector.stationary, [])


if __name__ == "__main__":
    unittest.main()
