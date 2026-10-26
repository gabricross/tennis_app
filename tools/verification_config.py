from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import EXAMPLE_VIDEO, HOME, PORT
from court import validate_corners

DEMO_CORNERS = [
    [344 / 1280, 131 / 720], [936 / 1280, 131 / 720],
    [1147 / 1280, 569 / 720], [124 / 1280, 568 / 720],
]


@dataclass(frozen=True)
class VerificationConfig:
    video: Path
    base_url: str
    host: str
    port: int
    corners: list[list[float]]
    start: float
    output: Path
    players_only: bool
    reference: str


def parse_arguments() -> VerificationConfig:
    parser = argparse.ArgumentParser(description="Verify Tennis Lab using your own local tennis clip.")
    parser.add_argument("--video", type=Path, default=EXAMPLE_VIDEO)
    parser.add_argument("--url", default=f"http://127.0.0.1:{PORT}")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--players-only", action="store_true", help="Validate a light installation without court or ball models.")
    parser.add_argument("--calibration-reference", choices=("doubles", "service_left", "service_right"), default="doubles")
    parser.add_argument("--corners", type=float, nargs=8, metavar="XY",
                        help="Four normalized outer corners, clockwise from the far-left corner.")
    arguments = parser.parse_args()
    if not arguments.video.is_file():
        parser.error("Provide an existing local clip with --video. No example video is shipped.")
    parsed = urlsplit(arguments.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
        parser.error("Verification only supports an HTTP server on this computer.")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        parser.error("--url must be a plain local server URL without credentials or a path.")
    if arguments.start < 0:
        parser.error("--start must be nonnegative.")
    corners = DEMO_CORNERS if arguments.corners is None else [
        arguments.corners[index:index + 2] for index in range(0, 8, 2)
    ]
    try:
        validate_corners(corners)
    except ValueError as error:
        parser.error(str(error))
    port = parsed.port or 80
    return VerificationConfig(
        arguments.video.resolve(), f"http://{parsed.hostname}:{port}", parsed.hostname,
        port, corners, arguments.start, HOME / "verification", arguments.players_only,
        arguments.calibration_reference,
    )
