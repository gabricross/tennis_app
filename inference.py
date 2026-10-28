from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

from config import MODELS
from court import LENGTH, PLAY_BOUNDS, WIDTH, project, session
from detection import Detection, PersonDetector
from ball_motion import ActiveBallSelector, MotionBallDetector


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
            left, top, right, bottom = PLAY_BOUNDS
            if not (left <= x <= right and top <= y <= bottom):
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


def ball_candidates(intensity: np.ndarray, threshold: int = 128) -> list[dict]:
    if intensity.shape != (360, 640):
        raise ValueError(f"Unexpected ball intensity shape: {intensity.shape}")
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        (intensity >= threshold).astype(np.uint8)
    )
    candidates = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if not 1 <= area <= 160 or max(width, height) > 23:
            continue
        if max(width, height) / max(1, min(width, height)) > 4:
            continue
        strength = float(np.mean(intensity[labels == index]))
        candidates.append({
            "x": float(centers[index, 0]), "y": float(centers[index, 1]),
            "intensity": strength,
        })
    return candidates


class BallTracker:
    """Associate neural measurements and verify at most two intervening frames."""

    def __init__(self, *, allow_reacquisition: bool = False) -> None:
        self.allow_reacquisition = allow_reacquisition
        self.history: deque[tuple[float, np.ndarray]] = deque(maxlen=2)
        self.gray: np.ndarray | None = None
        self.point: np.ndarray | None = None
        self.timestamp: float | None = None
        self.neural_time: float | None = None

    def update(self, image: np.ndarray, timestamp: float, candidates: list[dict] | None) -> dict | None:
        gray = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2GRAY)
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("Ball frames must have strictly increasing timestamps.")
        continuous = self.point is not None and self.timestamp is not None and timestamp - self.timestamp <= 0.05
        if self.gray is not None and np.mean(cv2.absdiff(gray, self.gray)) > 55:
            self.history.clear()
            self.point = None
            continuous = False
        point = None
        intensity = None
        method = "TrackNet + temporal association"
        velocity = np.zeros(2)
        if len(self.history) == 2:
            (t0, p0), (t1, p1) = self.history
            velocity = (p1 - p0) / (t1 - t0)
        if candidates is not None:
            recent = bool(self.history) and timestamp - self.history[-1][0] <= 0.3
            scored = []
            for candidate in candidates:
                position = np.array([candidate["x"], candidate["y"]], np.float32)
                cost = -candidate["intensity"] / 255
                if candidate.get("motion_support"):
                    cost -= 0.08 if self.allow_reacquisition else 0.35
                reset = not recent
                if recent:
                    dt = timestamp - self.history[-1][0]
                    expected = self.history[-1][1] + velocity * dt
                    distance = float(np.linalg.norm(position - expected))
                    # Permit impacts while rejecting unrelated distant components.
                    gate = 25 + 1100 * dt
                    if distance > gate and not self.allow_reacquisition:
                        continue
                    # A per-frame detector must escape a weaker, stationary spare-ball track.
                    cost += min(distance / gate, 1) * 0.08 if self.allow_reacquisition else distance / gate
                    reset = distance > gate
                scored.append((cost, position, candidate["intensity"], reset))
            if scored:
                _, point, intensity, reset = min(scored, key=lambda value: value[0])
                if reset:
                    self.history.clear()
                    continuous = False
                self.history.append((timestamp, point.copy()))
                self.neural_time = timestamp
        elif (
            self.point is not None and self.gray is not None and self.timestamp is not None
            and self.neural_time is not None and timestamp - self.neural_time < 0.09
            and timestamp - self.timestamp <= 0.05
        ):
            origin = self.point.reshape(1, 1, 2).astype(np.float32)
            moved, status, error = cv2.calcOpticalFlowPyrLK(
                self.gray, gray, origin, None, winSize=(15, 15), maxLevel=3,
                minEigThreshold=0.001,
            )
            if moved is not None and status is not None and error is not None and status[0, 0] and error[0, 0] < 20:
                backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.gray, moved, None, winSize=(15, 15), maxLevel=3,
                    minEigThreshold=0.001,
                )
                if (
                    backward is not None and back_status is not None and back_status[0, 0]
                    and np.linalg.norm(backward - origin) < 1
                ):
                    candidate = moved.reshape(2)
                    expected_step = velocity * (timestamp - self.timestamp)
                    error_step = np.linalg.norm(candidate - self.point - expected_step)
                    if (
                        0 <= candidate[0] < 640 and 0 <= candidate[1] < 360
                        and np.linalg.norm(candidate - self.point) <= 65
                        and error_step <= max(6, np.linalg.norm(expected_step) * 0.8)
                    ):
                        point = candidate
                        method = "bidirectional optical flow"
        self.gray, self.point, self.timestamp = gray, point, timestamp
        if point is None:
            return None
        height, width = image.shape[:2]
        return {
            "x": float(point[0] / 640 * width), "y": float(point[1] / 360 * height),
            "intensity": intensity, "observed": True, "method": method,
            "continuous": continuous,
        }


class BallModel:
    def __init__(self) -> None:
        self.net = session(MODELS / "ball.onnx")
        self.input_name = self.net.get_inputs()[0].name
        self.reset()

    def reset(self) -> None:
        self.tracker = BallTracker(allow_reacquisition=True)
        self.motion = MotionBallDetector()
        self.active = ActiveBallSelector()
        self.identity: int | None = None
        self.last_observation: tuple[float, int] | None = None

    def observe(self, image: np.ndarray, timestamp: float, candidates: list[dict] | None) -> dict | None:
        moving = self.motion.candidates(image)
        for candidate in candidates or []:
            candidate["motion_support"] = any(
                np.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"]) <= 8
                for other in moving
            )
        selected = self.active.update(candidates, timestamp, self.motion.affine)
        identity = selected["track_id"] if selected is not None else self.active.active_id
        if self.motion.affine is None or identity != self.identity:
            self.tracker = BallTracker(allow_reacquisition=True)
            self.identity = identity
            self.last_observation = None
        measurements: list[dict] | None = [selected] if selected is not None else []
        if candidates is None and identity is not None and self.motion.affine is not None:
            measurements = None
        point = self.tracker.update(image, timestamp, measurements)
        if point is not None and identity is not None:
            point["track_id"] = identity
            point["continuous"] = (
                point["continuous"] and self.last_observation is not None and self.last_observation[1] == identity
                and timestamp - self.last_observation[0] <= 0.05
            )
            self.last_observation = timestamp, identity
            return point
        self.last_observation = None
        return None

    def heatmap(self, frames: deque[np.ndarray]) -> np.ndarray:
        if len(frames) != 3:
            raise ValueError("TrackNet requires three consecutive frames.")
        resized = [cv2.resize(frame, (640, 360)) for frame in reversed(frames)]
        blob = np.concatenate(resized, axis=2).astype(np.float32) / 255
        blob = blob.transpose(2, 0, 1)[None].copy()
        return self.net.run(None, {self.input_name: blob})[0][0]

    def detect(self, frames: deque[np.ndarray], timestamp: float, *, infer: bool = True) -> dict | None:
        if len(frames) < 3:
            return self.observe(frames[-1], timestamp, [])
        candidates = ball_candidates(self.heatmap(frames)) if infer else None
        return self.observe(frames[-1], timestamp, candidates)


class BallObjectDetector:
    def __init__(self, path, *, size: int = 640, confidence: float = 0.15, multiscale: bool = False) -> None:
        if size not in (640, 960, 1280) or not 0 < confidence < 1 or (multiscale and size != 640):
            raise ValueError("Invalid ball detector size or confidence.")
        self.net = session(path)
        self.input_name = self.net.get_inputs()[0].name
        self.size, self.confidence = size, confidence
        self.multiscale = multiscale

    def candidates(self, image: np.ndarray) -> list[dict]:
        candidates = self._candidates(image, self.size)
        if not self.multiscale:
            return candidates
        candidates.extend(self._candidates(image, 960))
        merged = []
        for candidate in sorted(candidates, key=lambda value: -value["intensity"]):
            if not any(
                np.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"]) <= 3
                for other in merged
            ):
                merged.append(candidate)
        return merged

    def _candidates(self, image: np.ndarray, size: int) -> list[dict]:
        height, width = image.shape[:2]
        ratio = min(size / width, size / height)
        resized = cv2.resize(image, (round(width * ratio), round(height * ratio)))
        left, top = (size - resized.shape[1]) // 2, (size - resized.shape[0]) // 2
        padded = np.full((size, size, 3), 114, np.uint8)
        padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(padded, 1 / 255, swapRB=True)
        output = self.net.run(None, {self.input_name: blob})[0]
        if output.ndim != 3 or output.shape[:2] != (1, 5):
            raise ValueError(f"Unexpected tennis ball YOLO output: {output.shape}")
        values = output[0].T
        values = values[values[:, 4] >= self.confidence]
        if not len(values):
            return []
        boxes = np.column_stack((values[:, :2] - values[:, 2:4] / 2, values[:, 2:4]))
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), values[:, 4].tolist(), self.confidence, 0.4)
        candidates = []
        for index in np.asarray(indices).reshape(-1):
            x, y, _, _, score = values[int(index)]
            x, y = (float(x) - left) / ratio, (float(y) - top) / ratio
            if 0 <= x < width and 0 <= y < height:
                candidates.append({"x": x / width * 640, "y": y / height * 360,
                                   "intensity": float(score) * 255})
        return candidates


class YoloBallModel(BallModel):
    def __init__(self, *, multiscale: bool = False) -> None:
        self.detector = BallObjectDetector(MODELS / "ball_yolo.onnx", multiscale=multiscale)
        self.reset()

    def detect(self, frames: deque[np.ndarray], timestamp: float, *, infer: bool = True) -> dict | None:
        image = frames[-1]
        candidates = self.detector.candidates(image)
        point = self.observe(image, timestamp, candidates)
        if point is not None:
            point["detector_score"] = point["intensity"] / 255
            point["intensity"] = None
            point["method"] = "YOLOv8n + temporal association"
        return point


class ModelBank:
    def __init__(self) -> None:
        self.people: PersonDetector | None = None
        self.ball: BallModel | YoloBallModel | None = None
        self.ball_backend = "tracknet"

    def person_detector(self) -> PersonDetector:
        if self.people is None:
            cv2.setNumThreads(2)
            self.people = PersonDetector(MODELS / "people.onnx", confidence=0.28)
        return self.people

    def ball_detector(self, backend: str = "tracknet") -> BallModel | YoloBallModel:
        if backend not in ("tracknet", "yolo", "yolo_multiscale"):
            raise ValueError("Unknown ball backend.")
        if self.ball is None or backend != self.ball_backend:
            self.ball = BallModel() if backend == "tracknet" else YoloBallModel(multiscale=backend == "yolo_multiscale")
            self.ball_backend = backend
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
