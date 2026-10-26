from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from court import LENGTH, WIDTH, WORLD_CORNERS, CourtCamera, court_matrix, project, validate_corners
from engine import summarize
from inference import Detection, PlayerSelector

QUAD = [[0.25, 0.15], [0.75, 0.15], [0.92, 0.85], [0.08, 0.85]]


class CourtTests(unittest.TestCase):
    def test_corners_round_trip(self):
        matrix = court_matrix(QUAD, (1280, 720))
        actual = project(project(WORLD_CORNERS, matrix), np.linalg.inv(matrix))
        np.testing.assert_allclose(actual, WORLD_CORNERS, atol=1e-4)

    def test_crossed_corners_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_corners([QUAD[0], QUAD[2], QUAD[1], QUAD[3]])

    def test_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_corners([[float("nan"), 0], *QUAD[1:]])

    def test_small_court_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_corners([[0.1, 0.1], [0.15, 0.1], [0.15, 0.15], [0.1, 0.15]])

    def test_camera_rejects_missing_texture(self):
        image = np.full((720, 1280, 3), 80, np.uint8)
        camera = CourtCamera(image, court_matrix(QUAD, (1280, 720)))
        state = camera.update(image, [])
        self.assertFalse(state.valid)
        self.assertIn("Textura", state.reason)


class PlayerTests(unittest.TestCase):
    def setUp(self):
        self.matrix = court_matrix(QUAD, (1280, 720))

    def detection(self, position, score=0.9):
        x, y = project([position], self.matrix)[0]
        return Detection((float(x - 15), float(y - 70), float(x + 15), float(y)), score)

    def test_selects_one_person_per_half(self):
        selector = PlayerSelector()
        players = selector.select(
            [self.detection([WIDTH / 2, -1]), self.detection([WIDTH / 2, LENGTH + 1])],
            self.matrix, 0,
        )
        self.assertEqual({p["side"] for p in players}, {"north", "south"})

    def test_does_not_fabricate_hidden_players(self):
        selector = PlayerSelector()
        selector.select([self.detection([5, -1])], self.matrix, 0)
        self.assertEqual(selector.select([], self.matrix, 0.2), [])

    def test_provisional_selection_does_not_change_tracking_state(self):
        selector = PlayerSelector()
        selector.select([self.detection([5, -1])], self.matrix, 0, commit=False)
        self.assertEqual(selector.history, {})
        self.assertEqual(selector.last_boxes, {})

    def test_bystander_outside_court_is_rejected(self):
        selector = PlayerSelector()
        self.assertEqual(selector.select([self.detection([25, 5])], self.matrix, 0), [])

    def test_reports_captured_score_not_accuracy(self):
        result = PlayerSelector().select([self.detection([5, -1], 0.7)], self.matrix, 0)
        self.assertAlmostEqual(result[0]["confidence"], 0.7)
        self.assertTrue(result[0]["observed"])


class SummaryTests(unittest.TestCase):
    def test_coverage_counts_missing_samples(self):
        sample = {
            "camera_valid": True, "players": [
                {"side": "north", "pitch": [5, -1], "confidence": 0.8}
            ], "ball": None,
        }
        missing = {"camera_valid": False, "players": [], "ball": None}
        result = summarize([sample, missing], "players", 2, 1)
        self.assertEqual(result["players"]["north"]["coverage"], 0.5)
        self.assertEqual(result["camera_coverage"], 0.5)
        self.assertIsNone(result["ball_coverage"])
        self.assertTrue(result["warnings"])

    def test_no_detections_is_an_error_not_a_successful_empty_report(self):
        with self.assertRaises(ValueError):
            summarize([{"camera_valid": True, "players": [], "ball": None}], "players", 2, 1)


if __name__ == "__main__":
    unittest.main()
