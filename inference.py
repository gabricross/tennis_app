from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from config import MODELS
from court import LENGTH, WIDTH, project, session
from detection import Detection, PersonDetector


class PlayerSelector:
    def __init__(self) -> None:
        self.history: dict[str, tuple[float, np.ndarray]] = {}
        self.last_boxes: dict[str, tuple[float, tuple]] = {}

    def select(self, detections: list[Detection], matrix: np.ndarray, timestamp: float, commit: bool = True) -> list[dict]:
        if not detections:
            return []
        positions = project([d.foot for d in detections], np.linalg.inv(matrix))
        candidates = {"north": [], "south": []}
        for detection, position in zip(detections, positions, strict=True):
            x, y = position
            if not (-4 <= x <= WIDTH + 4 and -7 <= y <= LENGTH + 7):
                continue
            side = "north" if y < LENGTH / 2 else "south"
            expected = np.array([WIDTH / 2, -0.5 if side == "north" else LENGTH + 0.5])
            cost = 0.035 * float(np.linalg.norm(position - expected)) - detection.confidence
            if side in self.history:
                previous_time, previous = self.history[side]
                if timestamp - previous_time < 0.8:
                    distance = float(np.linalg.norm(position - previous))
                    if distance > 3.5 + 10 * (timestamp - previous_time):
                        continue
                    cost += 0.23 * distance
            cost += max(0, abs(x - WIDTH / 2) - WIDTH / 2) * 0.15
            candidates[side].append((cost, detection, position))
        result = []
        for side, values in candidates.items():
            if not values:
                continue
            _, detection, raw = min(values, key=lambda value: value[0])
            position = raw.copy()
            if side in self.history:
                previous_time, previous = self.history[side]
                if timestamp - previous_time < 0.35:
                    position = 0.65 * raw + 0.35 * previous
            if commit:
                self.history[side] = (timestamp, position)
                self.last_boxes[side] = (timestamp, detection.box)
            result.append(
                {"side": side, "box": list(detection.box), "pitch": position.tolist(),
                 "pitch_raw": raw.tolist(), "confidence": detection.confidence, "observed": True}
            )
        return result


class BallModel:
    def __init__(self) -> None:
        self.net = session(MODELS / "ball.onnx")
        self.input_name = self.net.get_inputs()[0].name
        self.previous: tuple[float, float, float] | None = None

    def reset(self) -> None:
        self.previous = None

    def detect(self, frames: deque[np.ndarray], timestamp: float) -> dict | None:
        if len(frames) < 3:
            return None
        current, previous, older = [
            cv2.resize(frame, (640, 360)) for frame in reversed(frames)
        ]
        blob = np.concatenate([current, previous, older], axis=2).astype(np.float32) / 255
        blob = blob.transpose(2, 0, 1)[None].copy()
        intensity = self.net.run(None, {self.input_name: blob})[0][0]
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            (intensity >= 150).astype(np.uint8)
        )
        candidates = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not 1 <= area <= 160 or max(width, height) > 23:
                continue
            if max(width, height) / max(1, min(width, height)) > 4:
                continue
            center = centers[index]
            strength = float(np.mean(intensity[labels == index]))
            cost = -strength / 255
            if self.previous is not None and timestamp - self.previous[0] < 0.4:
                distance = float(np.linalg.norm(center - self.previous[1:]))
                if distance > 210:
                    continue
                cost += distance / 120
            candidates.append((cost, center, strength))
        if not candidates:
            return None
        _, center, strength = min(candidates, key=lambda value: value[0])
        self.previous = (timestamp, float(center[0]), float(center[1]))
        height, width = frames[-1].shape[:2]
        return {
            "x": float(center[0] / 640 * width), "y": float(center[1] / 360 * height),
            "intensity": strength, "observed": True, "method": "TrackNet intensity candidate",
        }


class ModelBank:
    def __init__(self) -> None:
        self.people: PersonDetector | None = None
        self.ball: BallModel | None = None

    def person_detector(self) -> PersonDetector:
        if self.people is None:
            cv2.setNumThreads(2)
            self.people = PersonDetector(MODELS / "people.onnx", confidence=0.28)
        return self.people

    def ball_detector(self) -> BallModel:
        if self.ball is None:
            self.ball = BallModel()
        self.ball.reset()
        return self.ball


def benchmark(image: np.ndarray) -> dict:
    bank = ModelBank()
    start = time.perf_counter()
    people = bank.person_detector().detect(image)
    people_seconds = time.perf_counter() - start
    start = time.perf_counter()
    ball = bank.ball_detector().detect(deque([image, image, image], maxlen=3), 0)
    return {"people_seconds": people_seconds, "ball_seconds": time.perf_counter() - start,
            "people_detections": len(people), "static_ball_candidate": ball}
