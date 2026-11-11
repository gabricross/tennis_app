from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_HOME = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "TennisLab"
HOME = Path(os.environ.get("TENNISLAB_HOME", str(DEFAULT_HOME))).expanduser().resolve()
MODELS = HOME / "models"
DATA = HOME / "data"
DATABASE = DATA / "library.sqlite3"
SHOWCASE_FILE = DATA / "showcase.json"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 120
MAX_SOURCE_SECONDS = 1800
MAX_PENDING_JOBS = 3
FPS = 30
SAMPLE_STRIDE = 3
HOST = "127.0.0.1"
PORT = int(os.environ.get("TENNISLAB_PORT", "8768"))

for directory in (MODELS, DATA):
    directory.mkdir(parents=True, exist_ok=True)
