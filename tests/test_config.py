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

    def test_selected_missing_ball_model_never_falls_back_to_another_backend(self):
        code = """
from unittest.mock import Mock, patch
from fastapi import HTTPException
import app
from config import MODELS

values = dict(clip_id="fixture", start=0, duration=1, mode="complete",
              corners=[[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]])
(MODELS / "people.onnx").touch()
manager = Mock()
with patch.object(app, "require_clip", return_value={"metadata":{"duration":2}}), patch.object(app, "manager", manager):
    for selected, available in [("yolo","ball.onnx"), ("tracknet","ball_yolo.onnx"), ("yolo_wasb","ball_yolo.onnx")]:
        for path in MODELS.glob("ball*.onnx"):
            path.unlink()
        (MODELS / available).touch()
        request = app.AnalysisRequest(**values, **({"ball_backend":selected} if selected!="tracknet" else {}))
        assert request.ball_backend == selected
        try:
            app.start_job(request)
        except HTTPException as error:
            assert error.status_code == 503
            assert "seleccionado" in error.detail
        else:
            raise AssertionError("A different installed ball model must not be used silently.")
    manager.submit.assert_not_called()
    request = app.AnalysisRequest(**{**values, "mode":"players"})
    app.start_job(request)
    manager.submit.assert_called_once()
"""
        with tempfile.TemporaryDirectory() as temporary:
            subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT,
                env=dict(os.environ, TENNISLAB_HOME=temporary),
                capture_output=True, text=True, check=True,
            )


if __name__ == "__main__":
    unittest.main()
