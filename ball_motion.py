from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field


@dataclass
class _BallTrack:
    identity: int
    points: list[tuple[float, np.ndarray]] = field(default_factory=list)

@dataclass(eq=False)
class _StationaryCandidate:
    first: float
    last: float
    point: np.ndarray
    observations: int = 1


class ActiveBallSelector:
    """Confirm coherent motion, rather than choosing the strongest ball-shaped object."""

    def __init__(self) -> None:
        self.tracks: list[_BallTrack] = []
        self.next_id = 1
        self.active_id: int | None = None
        self.timestamp: float | None = None
        self.stationary: list[tuple[float, np.ndarray]] = []
        self.stationary_candidates: list[_StationaryCandidate] = []

    def update(self, candidates: list[dict] | None, timestamp: float,
               affine: np.ndarray | None) -> dict | None:
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("Ball candidate timestamps must increase.")
        self.timestamp = timestamp
        if affine is None:
            self.tracks.clear()
            self.active_id = None
            self.stationary.clear()
            self.stationary_candidates.clear()
        else:
            self.stationary = [
                (t, affine[:, :2] @ point + affine[:, 2])
                for t, point in self.stationary if timestamp - t <= 5
            ]
            self.stationary_candidates = [
                candidate for candidate in self.stationary_candidates if timestamp - candidate.last <= 0.35
            ]
            for candidate in self.stationary_candidates:
                candidate.point = affine[:, :2] @ candidate.point + affine[:, 2]
            for track in self.tracks:
                track.points = [
                    (t, affine[:, :2] @ point + affine[:, 2])
                    for t, point in track.points if timestamp - t <= 0.35
                ]
        self.tracks = [track for track in self.tracks
                       if track.points and timestamp - track.points[-1][0] <= 0.16]
        if candidates is None:
            return None
        blocked = set()
        for index, candidate in enumerate(candidates):
            position = np.array([candidate["x"], candidate["y"]], np.float64)
            known = next((
                item for item, (_, point) in enumerate(self.stationary)
                if np.linalg.norm(point - position) <= 3
            ), None)
            if known is not None:
                self.stationary[known] = timestamp, position.copy()
                blocked.add(index)
                continue
            stationary = next((
                item for item in self.stationary_candidates if np.linalg.norm(item.point - position) <= 1.2
            ), None)
            if stationary is None:
                self.stationary_candidates.append(_StationaryCandidate(timestamp, timestamp, position))
            else:
                stationary.last = timestamp
                stationary.observations += 1
                if stationary.observations >= 3 and timestamp - stationary.first >= 0.06:
                    self.stationary.append((timestamp, position.copy()))
                    self.stationary_candidates.remove(stationary)
                    blocked.add(index)
        assigned: dict[int, _BallTrack] = {}
        if self.tracks and candidates:
            costs = np.full((len(self.tracks), len(candidates)), 1e6)
            for row, track in enumerate(self.tracks):
                last_time, last = track.points[-1]
                dt = timestamp - last_time
                velocity = np.zeros(2)
                if len(track.points) >= 2:
                    prior_time, prior = track.points[-2]
                    velocity = (last - prior) / (last_time - prior_time)
                expected = last + velocity * dt
                gate = min(140, 12 + 1200 * dt)
                for column, candidate in enumerate(candidates):
                    if column in blocked:
                        continue
                    distance = float(np.linalg.norm(expected - [candidate["x"], candidate["y"]]))
                    if distance <= gate:
                        costs[row, column] = distance
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns, strict=True):
                if costs[row, column] < 1e6:
                    assigned[int(column)] = self.tracks[int(row)]
        eligible = []
        for index, candidate in enumerate(candidates):
            if index in blocked:
                continue
            track = assigned.get(index)
            if track is None:
                track = _BallTrack(self.next_id)
                self.next_id += 1
                self.tracks.append(track)
            position = np.array([candidate["x"], candidate["y"]], np.float64)
            track.points.append((timestamp, position))
            motion_points = track.points[-5:]
            if len(motion_points) < 3 or timestamp - motion_points[0][0] < 0.06:
                continue
            positions = np.array([point for _, point in motion_points])
            displacement = float(np.linalg.norm(positions[-1] - positions[0]))
            steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            travel = float(steps.sum())
            if displacement < 4 or np.count_nonzero(steps >= 1.2) < 2 or displacement / max(travel, 1e-6) < 0.55:
                continue
            eligible.append((track, candidate))
        if not eligible:
            return None
        track, candidate = min(
            eligible,
            key=lambda item: (
                item[0].identity != self.active_id,
                -item[1]["intensity"] / 255 - 0.1 * bool(item[1].get("motion_support")),
            ),
        )
        self.active_id = track.identity
        return {**candidate, "track_id": track.identity}


class MotionBallDetector:
    """Experimental yellow-component detector with static-camera compensation."""

    def __init__(self) -> None:
        self.previous: np.ndarray | None = None
        self.affine: np.ndarray | None = None

    def candidates(self, image: np.ndarray) -> list[dict]:
        resized = cv2.resize(image, (640, 360))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray
        self.affine = None
        if previous is None:
            return []
        features = cv2.goodFeaturesToTrack(previous, maxCorners=140, qualityLevel=0.02, minDistance=12)
        if features is None or len(features) < 12:
            return []
        moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, features, None)
        if moved is None or status is None:
            return []
        good = status.ravel() == 1
        if np.count_nonzero(good) < 12:
            return []
        affine, accepted = cv2.estimateAffinePartial2D(features[good], moved[good],
                                                      method=cv2.RANSAC, ransacReprojThreshold=2)
        if affine is None or accepted is None or np.mean(accepted) < 0.6:
            return []
        self.affine = affine
        aligned = cv2.warpAffine(previous, affine, (640, 360), borderMode=cv2.BORDER_REPLICATE)
        difference = cv2.absdiff(gray, aligned)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([20, 60, 130]), np.array([52, 255, 255]))
        count, labels, stats, centers = cv2.connectedComponentsWithStats(yellow)
        result = []
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if not 2 <= area <= 100 or max(width, height) > 24:
                continue
            if max(width, height) / min(width, height) > 4:
                continue
            component = labels[y:y + height, x:x + width] == index
            motion = difference[y:y + height, x:x + width][component]
            if np.mean(motion > 20) < 0.25:
                continue
            result.append({
                "x": float(centers[index, 0]), "y": float(centers[index, 1]),
                "intensity": float(np.mean(hsv[y:y + height, x:x + width, 2][component])),
            })
        return result
