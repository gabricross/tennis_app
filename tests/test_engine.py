from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from court import (
    CALIBRATION_REFERENCES, COURT_LINES, LENGTH, WIDTH, WORLD_CORNERS, CourtCamera,
    calibration_preview, court_matrix, line_support, project, validate_corners,
)
from engine import summarize
from inference import Detection, PlayerSelector

QUAD = [[0.25, 0.15], [0.75, 0.15], [0.92, 0.85], [0.08, 0.85]]


class CourtTests(unittest.TestCase):
    def test_corners_round_trip(self):
        matrix = court_matrix(QUAD, (1280, 720))
        actual = project(project(WORLD_CORNERS, matrix), np.linalg.inv(matrix))
        np.testing.assert_allclose(actual, WORLD_CORNERS, atol=1e-4)

    def test_partial_reference_cannot_project_full_court_through_horizon(self):
        with self.assertRaisesRegex(ValueError, "detrás de la cámara"):
            court_matrix([[0.4, 0.3], [0.6, 0.3], [0.9, 0.8], [0.1, 0.8]], (640, 360), "service_left")

    def test_calibration_preview_shows_full_lines_not_just_reference_rectangle(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        matrix = court_matrix(QUAD, (1280, 720))
        for line in COURT_LINES:
            a, b = np.rint(project(line, matrix)).astype(int)
            cv2.line(image, tuple(a), tuple(b), (255, 255, 255), 3)
        result = calibration_preview(image, QUAD)
        self.assertEqual(len(result["lines"]), len(COURT_LINES))
        self.assertEqual(result["visible_corners"], 4)
        self.assertTrue(result["usable"])
        self.assertFalse(result["partial"])
        reference = project(CALIBRATION_REFERENCES["service_left"], matrix) / [1280, 720]
        partial = calibration_preview(image, reference.tolist(), "service_left")
        self.assertTrue(partial["partial"])
        self.assertTrue(partial["usable"])
        np.testing.assert_allclose(result["lines"], partial["lines"], atol=1e-5)
        self.assertGreater(result["support"], 0.95)

    def test_calibration_preview_does_not_approve_an_image_without_lines(self):
        result = calibration_preview(np.zeros((360, 640, 3), np.uint8), QUAD)
        self.assertFalse(result["usable"])
        self.assertEqual(result["support"], 0)

    def test_pale_court_surface_is_not_counted_as_white_line_support(self):
        image = np.full((360, 640, 3), [170, 170, 215], np.uint8)
        result = calibration_preview(image, QUAD)
        self.assertFalse(result["usable"])
        self.assertEqual(result["support"], 0)

    def test_real_white_lines_remain_visible_on_a_pale_surface(self):
        image = np.full((360, 640, 3), [170, 170, 215], np.uint8)
        matrix = court_matrix(QUAD, (640, 360))
        for line in COURT_LINES:
            a, b = np.rint(project(line, matrix)).astype(int)
            cv2.line(image, tuple(a), tuple(b), (250, 250, 250), 3)
        self.assertGreater(line_support(image, matrix), 0.95)

    def test_bright_colored_shadow_stripes_are_not_white_paint(self):
        image = np.full((360, 640, 3), [90, 90, 120], np.uint8)
        matrix = court_matrix(QUAD, (640, 360))
        for line in COURT_LINES:
            a, b = np.rint(project(line, matrix)).astype(int)
            cv2.line(image, tuple(a), tuple(b), (170, 170, 215), 3)
        self.assertEqual(line_support(image, matrix), 0)

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

    def test_initial_camera_features_exclude_moving_people(self):
        rng = np.random.default_rng(24)
        image = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
        box = [400, 250, 700, 550]
        camera = CourtCamera(image, court_matrix(QUAD, (1280, 720)), boxes=[box])
        self.assertIsNotNone(camera.features)
        features = camera.features.reshape(-1, 2) / camera.scale
        inside_player = (
            (features[:, 0] >= box[0]) & (features[:, 0] <= box[2])
            & (features[:, 1] >= box[1]) & (features[:, 1] <= box[3])
        )
        self.assertFalse(np.any(inside_player))

    def test_partial_camera_handles_projective_motion(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        initial = court_matrix(QUAD, (640, 360), "service_left")
        motion = np.array([[1, 0.025, 3], [0.015, 1, 2], [0.00015, -0.0001, 1]], np.float64)
        moved = cv2.warpPerspective(image, motion, (640, 360))
        camera = CourtCamera(image, initial, allow_partial=True)
        state = camera.update(moved, [])
        self.assertTrue(state.valid, state.reason)
        actual = project(CALIBRATION_REFERENCES["service_left"], state.matrix)
        expected = project(CALIBRATION_REFERENCES["service_left"], motion @ initial)
        np.testing.assert_allclose(actual, expected, atol=1)

    def test_partial_camera_recovers_only_with_static_correspondences(self):
        rng = np.random.default_rng(52)
        image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        matrix = court_matrix(QUAD, (640, 360), "service_left")
        motion = np.array([[1, 0, 7], [0, 1, 4], [0, 0, 1]], np.float64)
        camera = CourtCamera(image, matrix, allow_partial=True)
        camera.lost = True
        moved = cv2.warpPerspective(image, motion, (640, 360))
        state = camera.update(moved, [])
        self.assertTrue(state.valid, state.reason)
        self.assertIn("recuperado", state.reason)
        actual = project(CALIBRATION_REFERENCES["service_left"], state.matrix)
        expected = project(CALIBRATION_REFERENCES["service_left"], motion @ matrix)
        np.testing.assert_allclose(actual, expected, atol=1)

    def test_partial_camera_does_not_recover_on_unrelated_scene(self):
        rng = np.random.default_rng(53)
        image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        unrelated = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        camera = CourtCamera(image, court_matrix(QUAD, (640, 360)), allow_partial=True)
        camera.lost = True
        state = camera.update(unrelated, [])
        self.assertFalse(state.valid)

    def test_full_court_camera_can_recover_verified_original_scene(self):
        rng = np.random.default_rng(92)
        image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        matrix = court_matrix(QUAD, (640, 360))
        camera = CourtCamera(image, matrix)
        camera.lost = True
        state = camera.update(image, [])
        self.assertTrue(state.valid, state.reason)
        np.testing.assert_allclose(
            project(WORLD_CORNERS, state.matrix), project(WORLD_CORNERS, matrix), atol=1,
        )

    def test_partial_reference_round_trip_uses_service_coordinates(self):
        for reference in ("service_boxes", "service_left", "service_right"):
            with self.subTest(reference=reference):
                matrix = court_matrix(QUAD, (1280, 720), reference)
                actual = project(CALIBRATION_REFERENCES[reference], matrix)
                np.testing.assert_allclose(actual, np.array(QUAD) * [1280, 720], atol=1e-3)
                restored = project(actual, np.linalg.inv(matrix))
                np.testing.assert_allclose(restored, CALIBRATION_REFERENCES[reference], atol=1e-4)

    def test_broader_service_reference_works_without_weakening_small_quad_rejection(self):
        image = np.zeros((720, 1280, 3), np.uint8)
        matrix = court_matrix([[0.35, 0.35], [0.65, 0.35], [0.85, 0.65], [0.15, 0.65]], (1280, 720))
        for line in COURT_LINES:
            a, b = np.rint(project(line, matrix)).astype(int)
            cv2.line(image, tuple(a), tuple(b), (255, 255, 255), 3)
        left = project(CALIBRATION_REFERENCES["service_left"], matrix) / [1280, 720]
        with self.assertRaisesRegex(ValueError, "poco espacio"):
            calibration_preview(image, left.tolist(), "service_left")
        both = project(CALIBRATION_REFERENCES["service_boxes"], matrix) / [1280, 720]
        result = calibration_preview(image, both.tolist(), "service_boxes")
        self.assertTrue(result["usable"])
        self.assertTrue(result["partial"])

    def test_service_midpoint_projects_to_net_ground_not_net_tape(self):
        matrix = court_matrix(QUAD, (1280, 720), "service_left")
        midpoint = [(1.37 + WIDTH / 2) / 2, LENGTH / 2]
        image_point = project([midpoint], matrix)
        restored = project(image_point, np.linalg.inv(matrix))[0]
        self.assertAlmostEqual(float(restored[1]), LENGTH / 2, places=4)

    def test_unknown_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "desconocida"):
            court_matrix(QUAD, (1280, 720), "invented")

    def test_old_calibration_is_unchanged(self):
        np.testing.assert_array_equal(
            court_matrix(QUAD, (1280, 720)),
            court_matrix(QUAD, (1280, 720), "doubles"),
        )

    def test_partial_line_support_ignores_offscreen_lines_only_when_enabled(self):
        pixels = np.array([[473, 449], [785, 446], [842, 574], [211, 575]])
        matrix = court_matrix((pixels / [1280, 720]).tolist(), (1280, 720), "service_left")
        image = np.zeros((720, 1280, 3), np.uint8)
        for line in COURT_LINES:
            a, b = np.rint(project(line, matrix)).astype(int)
            visible, start, end = cv2.clipLine((0, 0, 1280, 720), tuple(a), tuple(b))
            if visible:
                cv2.line(image, start, end, (255, 255, 255), 3)
        self.assertEqual(line_support(image, matrix), 0)
        self.assertGreater(line_support(image, matrix, allow_partial=True), 0.95)


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
