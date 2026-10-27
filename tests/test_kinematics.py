from __future__ import annotations

import unittest

import numpy as np

from court import CALIBRATION_REFERENCES, LENGTH, SERVICE_BOTTOM, SERVICE_TOP, court_matrix, project
from kinematics import BALL_SPEED, player_kinematics


def observation(t, x, *, valid=True):
    return {
        "time": t, "camera_valid": valid, "line_support": 1,
        "matrix": [[100, 0, 0], [0, 100, 0], [0, 0, 1]],
        "players": [{"side": "north", "pitch": [x, 2], "pitch_raw": [x, 2],
                     "box": [100 * x - 10, 100, 100 * x + 10, 200]}] if valid else [],
    }


class KinematicsTests(unittest.TestCase):
    def test_service_lines_are_640cm_from_net_not_baseline(self):
        self.assertAlmostEqual(LENGTH / 2 - SERVICE_TOP, 6.4)
        self.assertAlmostEqual(SERVICE_BOTTOM - LENGTH / 2, 6.4)
        self.assertAlmostEqual(SERVICE_BOTTOM - SERVICE_TOP, 12.8)
        quad = [[.2, .2], [.7, .2], [.8, .8], [.1, .8]]
        matrix = court_matrix(quad, (1280, 720), "service_left")
        np.testing.assert_allclose(project(CALIBRATION_REFERENCES["service_left"], matrix),
                                   np.array(quad) * [1280, 720], atol=.001)

    def test_known_five_metre_per_second_trajectory(self):
        samples = [observation(i / 10, 1 + .5 * i) for i in range(21)]
        result = player_kinematics(samples)["north"]
        self.assertAlmostEqual(result["mean_speed_kmh"], 18, places=5)
        self.assertAlmostEqual(result["peak_speed_kmh"], 18, places=5)
        self.assertAlmostEqual(result["distance_m"], result["observed_seconds"] * 5, places=5)
        self.assertEqual(result["rejected_observations"], 0)

    def test_stationary_detector_jitter_does_not_accumulate_distance(self):
        samples = [observation(i / 10, 5 + (.08 if i % 2 else -.08)) for i in range(30)]
        result = player_kinematics(samples)["north"]
        self.assertEqual(result["distance_m"], 0)
        self.assertIsNone(result["peak_speed_kmh"])

    def test_missing_camera_does_not_create_travel_across_gap(self):
        samples = ([observation(i / 10, 1) for i in range(7)] +
                   [observation(.7, 0, valid=False)] +
                   [observation(i / 10, 8) for i in range(8, 16)])
        result = player_kinematics(samples)["north"]
        self.assertEqual(result["distance_m"], 0)
        self.assertLess(result["observed_seconds"], 1.5)

    def test_outlier_is_excluded_not_clipped_into_a_speed(self):
        samples = [observation(i / 10, 5 if i != 8 else 25) for i in range(20)]
        result = player_kinematics(samples)["north"]
        self.assertGreater(result["rejected_observations"], 0)
        self.assertEqual(result["distance_m"], 0)

    def test_insufficient_observations_are_null_not_zero_measurements(self):
        samples = [observation(0, 1), observation(.1, 1)]
        result = player_kinematics(samples)["north"]
        self.assertIsNone(result["distance_m"])
        self.assertIsNone(result["mean_speed_kmh"])
        self.assertFalse(BALL_SPEED["available"])
        self.assertIsNone(BALL_SPEED["speed_kmh"])

    def test_subresolution_wobble_does_not_create_peak_without_travel(self):
        samples = []
        for index, x in enumerate([5, 5, 5, 5.25, 5.25, 4.75, 4.75, 4.75]):
            sample = observation(index / 10, x)
            sample["matrix"] = [[10, 0, 0], [0, 10, 0], [0, 0, 1]]
            sample["players"][0]["box"] = [10 * x - 1, 10, 10 * x + 1, 20]
            samples.append(sample)
        result = player_kinematics(samples)["north"]
        self.assertEqual(result["distance_m"], 0)
        self.assertIsNone(result["peak_speed_kmh"])

    def test_camera_break_between_valid_samples_starts_a_new_segment(self):
        samples = [observation(i / 10, 1 if i < 7 else 2) for i in range(15)]
        samples[7]["camera_continuous"] = False
        result = player_kinematics(samples)["north"]
        self.assertEqual(result["distance_m"], 0)


if __name__ == "__main__":
    unittest.main()
