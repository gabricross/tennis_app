from __future__ import annotations

import unittest

import cv2
import numpy as np

from court import project
from tools.benchmark_player_reference import compare, ground_matrix


class PlayerReferenceTests(unittest.TestCase):
    def test_metric_ground_matrix_matches_independent_3d_projection(self):
        rotation, _ = cv2.Rodrigues(np.array([0.6, 0.1, 0.2]))
        camera = {
            "K": [[800, 0, 640], [0, 810, 360], [0, 0, 1]],
            "R": rotation.tolist(), "t": [[5], [120], [350]], "resolution": [1280, 720],
        }
        intrinsic, matrix = ground_matrix(camera, (640, 360), -3)
        metres = np.array([[0, 0], [1.25, 2.5], [-1, 1]])
        world_cm = np.column_stack((metres[:, 0] * 100, [-3] * 3, metres[:, 1] * 100))
        expected, _ = cv2.projectPoints(
            world_cm, rotation, np.array(camera["t"], float), intrinsic, None,
        )
        np.testing.assert_allclose(project(metres, matrix), expected.reshape(-1, 2), atol=1e-5)
        np.testing.assert_allclose(project(expected.reshape(-1, 2), np.linalg.inv(matrix)), metres, atol=1e-6)

    def test_reference_comparison_uses_matching_times_and_known_speed(self):
        samples, reference = [], []
        for index in range(21):
            x = 1 + index * 0.5
            reference.append(np.array([x, 2]))
            samples.append({
                "time": index / 10, "matrix": [[100, 0, 0], [0, 100, 0], [0, 0, 1]],
                "camera_valid": True,
                "players": [{"side": "north", "pitch": [x, 2], "pitch_raw": [x, 2],
                             "box": [100 * x - 10, 100, 100 * x + 10, 200]}],
            })
        result = compare(samples, reference)
        self.assertAlmostEqual(result["position_error_m"]["mae"], 0)
        self.assertAlmostEqual(result["speed_error_kmh"]["mae"], 0)
        self.assertAlmostEqual(result["matched_distance"]["difference_m"], 0)
        self.assertAlmostEqual(result["matched_distance"]["reference_polyline_m"], 9)

    def test_missing_reference_is_not_zero_error_or_bridged_distance(self):
        samples = [
            {"time": i / 10, "matrix": None, "camera_valid": False, "players": []}
            for i in range(6)
        ]
        result = compare(samples, [None] * 6)
        self.assertIsNone(result["position_error_m"]["mae"])
        self.assertIsNone(result["speed_error_kmh"]["mae"])
        self.assertEqual(result["matched_distance"]["observed_seconds"], 0)
        self.assertIsNone(result["matched_distance"]["reference_polyline_m"])
        self.assertIsNone(result["matched_distance"]["difference_m"])


if __name__ == "__main__":
    unittest.main()
