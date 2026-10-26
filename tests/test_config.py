from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_explicit_home_works_without_localappdata(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.pop("LOCALAPPDATA", None)
            environment["TENNISLAB_HOME"] = temporary
            result = subprocess.run(
                [sys.executable, "-c",
                 "import config; print(config.HOME); assert config.DATA.is_dir(); assert config.MODELS.is_dir()"],
                cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
            )
            self.assertEqual(Path(result.stdout.strip()), Path(temporary).resolve())

    def test_application_imports_without_models_or_sibling_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ, TENNISLAB_HOME=temporary)
            result = subprocess.run(
                [sys.executable, "-c",
                 "import sys, app, inference; print(inference.PersonDetector.__module__); "
                 "assert 'futbol.vision' not in sys.modules"],
                cwd=ROOT, env=environment, capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), "detection")


if __name__ == "__main__":
    unittest.main()
