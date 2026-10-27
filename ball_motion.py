from __future__ import annotations

import cv2
import numpy as np


class MotionBallDetector:
    """Experimental yellow-component detector with static-camera compensation."""

    def __init__(self) -> None:
        self.previous: np.ndarray | None = None

    def candidates(self, image: np.ndarray) -> list[dict]:
        resized = cv2.resize(image, (640, 360))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, gray
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
