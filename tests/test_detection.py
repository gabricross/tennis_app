from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detection import Detection, PersonDetector


class DetectionTests(unittest.TestCase):
    def make_detector(self, output=None):
        network = Mock()
        network.forward.return_value = np.zeros((1, 8400, 85), np.float32) if output is None else output
        with patch("detection.cv2.dnn.readNetFromONNX", return_value=network):
            detector = PersonDetector(Path("unused-test-model.onnx"))
        return detector, network

    def test_no_people_is_empty_not_an_invalid_shape(self):
        detector, network = self.make_detector()
        self.assertEqual(detector.detect(np.zeros((360, 640, 3), np.uint8)), [])
        self.assertEqual(network.setInput.call_args.args[0].shape, (1, 3, 640, 640))

    def test_decodes_coco_person_and_unletterboxes(self):
        output = np.zeros((1, 8400, 85), np.float32)
        output[0, 0, :6] = [20, 20, np.log(10), np.log(20), 0.9, 0.8]
        detector, _ = self.make_detector(output)
        result = detector.detect(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(len(result), 1)
        np.testing.assert_allclose(result[0].box, (240, 160, 400, 480), atol=0.001)
        self.assertAlmostEqual(result[0].confidence, 0.72, places=5)

    def test_unexpected_model_output_is_an_explicit_error(self):
        detector, _ = self.make_detector(np.zeros((1, 10, 8), np.float32))
        with self.assertRaisesRegex(ValueError, "Unexpected YOLOX"):
            detector.detect(np.zeros((360, 640, 3), np.uint8))

    def test_empty_frame_is_rejected(self):
        detector, _ = self.make_detector()
        with self.assertRaises(ValueError):
            detector.detect(np.zeros((0, 10, 3), np.uint8))

    def test_feet_are_box_bottom_midpoint(self):
        self.assertEqual(Detection((10, 20, 50, 90), 0.8).foot, (30, 90))

    def test_invalid_threshold_does_not_load_a_model(self):
        with self.assertRaises(ValueError):
            PersonDetector(Path("unused.onnx"), confidence=0)


if __name__ == "__main__":
    unittest.main()
