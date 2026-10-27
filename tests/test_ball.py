from __future__ import annotations

import sys
import unittest
from pathlib import Path
from collections import deque
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import BallObjectDetector, BallTracker, ModelBank, YoloBallModel, ball_candidates
from engine import draw_frame, summarize
from court import LENGTH, WIDTH
from ball_motion import MotionBallDetector


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
    def test_yolo_reacquires_stronger_ball_instead_of_locking_on_a_spare(self):
        model = YoloBallModel.__new__(YoloBallModel)
        model.detector = Mock()
        spare = {"x": 100, "y": 120, "intensity": 0.4 * 255}
        active = {"x": 500, "y": 250, "intensity": 0.8 * 255}
        model.detector.candidates.side_effect = [[spare.copy()], [spare.copy()], [spare.copy(), active]]
        model.reset()
        frames = deque([np.zeros((360, 640, 3), np.uint8)], maxlen=3)
        model.detect(frames, 0)
        model.detect(frames, 1 / 30)
        point = model.detect(frames, 2 / 30)
        self.assertAlmostEqual(point["x"], 500)
        self.assertEqual(len(model.tracker.history), 1)

    def test_yolo_uses_continuity_to_break_near_equal_detector_scores(self):
        model = YoloBallModel.__new__(YoloBallModel)
        model.detector = Mock()
        model.detector.candidates.side_effect = [
            [{"x": 100, "y": 120, "intensity": 0.8 * 255}],
            [{"x": 105, "y": 120, "intensity": 0.74 * 255},
             {"x": 500, "y": 250, "intensity": 0.8 * 255}],
        ]
        model.reset()
        frames = deque([np.zeros((360, 640, 3), np.uint8)], maxlen=3)
        model.detect(frames, 0)
        point = model.detect(frames, 1 / 30)
        self.assertAlmostEqual(point["x"], 105)

    def test_letterbox_rgb_decode_and_nms(self):
        detector = BallObjectDetector.__new__(BallObjectDetector)
        detector.size, detector.confidence, detector.input_name = 640, 0.15, "images"
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
        model = YoloBallModel.__new__(YoloBallModel)
        model.detector = Mock()
        model.detector.candidates.return_value = [{"x": 100, "y": 120, "intensity": 0.8 * 255}]
        model.reset()
        frames = deque([np.zeros((360, 640, 3), np.uint8)], maxlen=3)
        first = model.detect(frames, 0, infer=False)
        second = model.detect(frames, 1 / 30, infer=False)
        self.assertEqual(model.detector.candidates.call_count, 2)
        self.assertEqual(second["method"], "YOLOv8n + temporal association")
        self.assertIsNone(first["intensity"])
        self.assertAlmostEqual(first["detector_score"], 0.8)

    def test_unknown_backend_is_not_silently_substituted(self):
        with self.assertRaises(ValueError):
            ModelBank().ball_detector("invented")


if __name__ == "__main__":
    unittest.main()
