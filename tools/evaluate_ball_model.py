from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import HOME
from inference import BallObjectDetector
from tools.evaluate_tracking import localization


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen alternative detector against independently reviewed ball frames.")
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()
    root = HOME / "tracking_benchmark"
    truth = json.loads(Path(__file__).with_name("tracking_labels.json").read_text(encoding="utf-8"))
    detector = BallObjectDetector(root / "ball_yolo.onnx", size=args.size)
    report = {"note": truth["note"], "size": args.size, "confidence_threshold": 0.15, "clips": {}}
    for identifier, labels in truth["labels"].items():
        capture = cv2.VideoCapture(str(root / identifier / "review" / "original.mp4"))
        width, height = int(capture.get(3)), int(capture.get(4))
        series, elapsed = [], []
        for index in range(120):
            success, image = capture.read()
            if not success:
                raise RuntimeError(f"Truncated {identifier}")
            if index not in truth["frames"]:
                continue
            started = time.perf_counter()
            candidates = detector.candidates(image)
            elapsed.append(time.perf_counter() - started)
            point = max(candidates, key=lambda p: p["intensity"]) if candidates else None
            if point:
                point = {**point, "x": point["x"] / 640 * width, "y": point["y"] / 360 * height}
            series.append({"frame": index, "ball": point})
        capture.release()
        report["clips"][identifier] = {
            "localization": localization(series, labels, truth["frames"], (width, height),
                                         truth["tolerance_model_pixels"]),
            "median_inference_seconds": float(np.median(elapsed)), "predictions": series,
        }
    output = root / f"yolo_{args.size}_screening.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for identifier, result in report["clips"].items():
        print(identifier, json.dumps(result["localization"]), result["median_inference_seconds"])


if __name__ == "__main__":
    main()
