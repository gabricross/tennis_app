from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import probe, validate_signature


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.video = Path(self.temporary.name) / "upload"
        self.video.write_bytes((24).to_bytes(4, "big") + b"ftypisom" + b"\0" * 64)

    def tearDown(self):
        self.temporary.cleanup()

    def test_invalid_signature_is_rejected(self):
        self.video.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            validate_signature(self.video)

    def test_resolution_is_checked_before_reading_pixels(self):
        def frames():
            yield {"size": (100000, 100000), "duration": 10, "fps": 30}
            raise AssertionError("Pixels must not be requested for an oversized video.")
        with patch("media.imageio_ffmpeg.read_frames", return_value=frames()):
            with self.assertRaises(ValueError):
                probe(self.video)

    def test_nonfinite_duration_is_rejected(self):
        def frames():
            yield {"size": (640, 360), "duration": float("inf"), "fps": 30}
            raise AssertionError("Pixels must not be requested for invalid duration.")
        with patch("media.imageio_ffmpeg.read_frames", return_value=frames()):
            with self.assertRaises(ValueError):
                probe(self.video)

    def test_variable_frame_rate_is_explicit_not_silently_assumed(self):
        def frames():
            yield {"size": (640, 360), "duration": 2, "fps": 0, "codec": "vp8"}
            yield bytes(640 * 360 * 3)
        with patch("media.imageio_ffmpeg.read_frames", return_value=frames()):
            metadata = probe(self.video)
        self.assertIsNone(metadata["fps"])
        self.assertEqual(metadata["normalization_fps"], 30)
        self.assertTrue((self.video.parent / "poster.jpg").exists())


if __name__ == "__main__":
    unittest.main()
