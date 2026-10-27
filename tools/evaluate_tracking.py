from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ball_motion import MotionBallDetector
from config import HOME
from inference import BallTracker


def localization(series: list[dict], labels: list, frames: list[int], size: tuple[int, int],
                 tolerance: float, *, held: bool = False) -> dict:
    matches, emitted, visible = 0, 0, 0
    errors = []
    for frame, label in zip(frames, labels, strict=True):
        if label is None:
            continue
        visible += 1
        observation = next((s for s in reversed(series) if s["frame"] <= frame), None) if held else next(
            (s for s in series if s["frame"] == frame), None
        )
        point = observation["ball"] if observation is not None else None
        if point is None:
            continue
        emitted += 1
        position = np.array([point["x"] / size[0] * 640, point["y"] / size[1] * 360])
        distance = float(np.linalg.norm(position - label))
        errors.append(distance)
        matches += distance <= tolerance
    return {
        "visible_labeled_frames": visible, "emitted_on_visible_frames": emitted,
        "localized_within_tolerance": matches, "wrong_location": emitted - matches,
        "missing_on_visible_frames": visible - emitted, "pixel_errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Review localization, not just output coverage.")
    parser.add_argument("--results", type=Path, default=HOME / "tracking_benchmark" / "results")
    parser.add_argument("--labels", type=Path, default=Path(__file__).with_name("tracking_labels.json"))
    parser.add_argument("--score-only", action="store_true", help="Score existing outputs without running another experiment.")
    args = parser.parse_args()
    truth = json.loads(args.labels.read_text(encoding="utf-8"))
    report = {"note": truth["note"], "tolerance_model_pixels": truth["tolerance_model_pixels"], "clips": {}}
    if args.score_only:
        for identifier, labels in truth["labels"].items():
            path = args.results / identifier / "analysis.json"
            if not path.exists():
                raise FileNotFoundError(path)
            result = json.loads(path.read_text(encoding="utf-8"))
            report["clips"][identifier] = localization(
                result["ball_samples"], labels, truth["frames"], (result["width"], result["height"]),
                truth["tolerance_model_pixels"],
            )
        report["totals"] = {
            key: sum(clip[key] for clip in report["clips"].values())
            for key in ("visible_labeled_frames", "emitted_on_visible_frames", "localized_within_tolerance",
                        "wrong_location", "missing_on_visible_frames")
        }
        (args.results / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    cv2.setNumThreads(2)
    for identifier, labels in truth["labels"].items():
        directory = args.results / identifier
        source = directory / "original.mp4"
        if not source.exists():
            source = HOME / "tracking_benchmark" / identifier / "review" / "original.mp4"
        if not source.exists():
            raise FileNotFoundError(source)
        capture = cv2.VideoCapture(str(source))
        width, height = int(capture.get(3)), int(capture.get(4))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        detector, tracker = MotionBallDetector(), BallTracker()
        fast = []
        started = time.perf_counter()
        for index in range(total):
            success, image = capture.read()
            if not success:
                raise RuntimeError(f"Truncated video: {identifier}")
            candidates = detector.candidates(image)
            point = tracker.update(image, index / 30, candidates)
            if point is not None:
                point["method"] = "experimental color + camera-compensated motion"
            fast.append({"frame": index, "time": index / 30, "ball": point})
        capture.release()
        duration = time.perf_counter() - started
        record = {"color_motion": localization(
            fast, labels, truth["frames"], (width, height), truth["tolerance_model_pixels"]),
            "color_motion_seconds_decode_and_track": duration,
            "color_motion_coverage": sum(s["ball"] is not None for s in fast) / total}
        if (directory / "analysis.json").exists():
            result = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
            series_to_score = [("neural_temporal", result["ball_samples"], False)]
            if (directory / "legacy.json").exists():
                legacy = json.loads((directory / "legacy.json").read_text(encoding="utf-8"))
                series_to_score.append(("legacy_hold", legacy, True))
            for name, series, held in series_to_score:
                record[name] = localization(series, labels, truth["frames"], (width, height),
                                            truth["tolerance_model_pixels"], held=held)
        report["clips"][identifier] = record
        output = args.results / "motion_experiment"
        output.mkdir(exist_ok=True)
        (output / f"{identifier}.json").write_text(json.dumps(fast), encoding="utf-8")
    (args.results / "quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
