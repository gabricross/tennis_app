from __future__ import annotations

from collections import deque

import numpy as np

from court import project

MAX_PLAYER_SPEED_MPS = 12.0
MAX_OBSERVATION_GAP = 0.25
BALL_SPEED = {
    "available": False, "speed_kmh": None,
    "reason": "La pelota está en el aire: hacen falta calibración 3D o mediciones adicionales.",
}


def _resolution(sample: dict, player: dict) -> float:
    matrix = sample["matrix"]
    if matrix is None:
        raise ValueError("A valid camera observation needs a projection matrix.")
    x1, _, x2, y2 = player["box"]
    foot = np.array([(x1 + x2) / 2, y2])
    offsets = [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]]
    points = project(foot + offsets, np.linalg.inv(matrix))
    return float(np.max(np.linalg.norm(points[1:] - points[0], axis=1)))


def player_kinematics(samples: list[dict]) -> dict:
    """Estimate observed ground travel; never bridge missing or rejected observations."""
    result = {}
    for side in ("north", "south"):
        history: deque[tuple[float, np.ndarray]] = deque(maxlen=3)
        speed_window: deque[tuple[float, float]] = deque()
        anchor = None
        previous_filtered = None
        previous_time = None
        distance = valid_seconds = 0.0
        rejected = 0
        speeds = []
        for sample in samples:
            timestamp = sample["time"]
            if previous_time is not None and timestamp <= previous_time:
                raise ValueError("Player observation timestamps must increase.")
            previous_time = timestamp
            player = next((p for p in sample["players"] if p["side"] == side), None)
            if player is None or not sample["camera_valid"]:
                history.clear()
                speed_window.clear()
                previous_filtered = None
                anchor = None
                continue
            if not sample.get("camera_continuous", True):
                history.clear()
                speed_window.clear()
                previous_filtered = None
                anchor = None
            motion = {"speed_kmh": None, "distance_m": distance if valid_seconds else None,
                      "status": "warming_up"}
            player["motion"] = motion
            resolution = _resolution(sample, player)
            position = np.array(player.get("pitch_raw", player["pitch"]), np.float64)
            if (
                not np.isfinite(position).all() or not np.isfinite(resolution)
                or resolution > 0.75 or sample.get("line_support", 1) < 0.3
            ):
                motion["status"] = "insufficient_calibration"
                history.clear()
                speed_window.clear()
                previous_filtered = None
                anchor = None
                rejected += 1
                continue
            if history:
                dt = timestamp - history[-1][0]
                if dt > MAX_OBSERVATION_GAP:
                    history.clear()
                    speed_window.clear()
                    previous_filtered = None
                    anchor = None
                elif np.linalg.norm(position - history[-1][1]) > MAX_PLAYER_SPEED_MPS * dt + 2 * resolution:
                    motion["status"] = "rejected_jump"
                    history.clear()
                    speed_window.clear()
                    previous_filtered = None
                    anchor = None
                    rejected += 1
                    continue
            history.append((timestamp, position))
            if len(history) < 3:
                continue
            median = np.median([p for _, p in history], axis=0)
            middle_time = history[1][0]
            deadband = max(0.2, 2 * resolution)
            motion.update(status="below_resolution", position_resolution_m=resolution)
            if anchor is not None and previous_filtered is not None:
                dt = middle_time - previous_filtered[0]
                step = float(np.linalg.norm(median - anchor[1]))
                window = middle_time - anchor[0]
                speed = step / window if step > deadband else None
                if (
                    np.linalg.norm(median - previous_filtered[1]) / dt > MAX_PLAYER_SPEED_MPS
                    or (speed is not None and speed > MAX_PLAYER_SPEED_MPS)
                ):
                    motion["status"] = "rejected_speed"
                    history.clear()
                    speed_window.clear()
                    previous_filtered = None
                    anchor = None
                    rejected += 1
                    continue
                valid_seconds += dt
                if speed is not None:
                    distance += step
                    anchor = middle_time, median.copy()
                elif window >= 0.6:
                    anchor = middle_time, median.copy()
            else:
                anchor = middle_time, median.copy()
            previous_filtered = middle_time, median.copy()
            motion["distance_m"] = distance if valid_seconds else None
            speed_window.append((middle_time, distance))
            while middle_time - speed_window[0][0] > 0.6 + 1e-6:
                speed_window.popleft()
            interval = middle_time - speed_window[0][0]
            travel = distance - speed_window[0][1]
            if interval >= 0.29 and travel > 0:
                speed = travel / interval
                if speed <= MAX_PLAYER_SPEED_MPS:
                    speed *= 3.6
                    motion.update(speed_kmh=speed, speed_window_seconds=interval, status="estimated")
                    speeds.append(speed)
                else:
                    motion["status"] = "rejected_speed"
        result[side] = {
            "distance_m": distance if valid_seconds else None,
            "observed_seconds": valid_seconds,
            "mean_speed_kmh": distance / valid_seconds * 3.6 if valid_seconds and (speeds or distance > 0) else None,
            "peak_speed_kmh": max(speeds) if speeds else None,
            "rejected_observations": rejected,
        }
    return result
