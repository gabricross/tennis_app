from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
RACKET_CLASS = 38  # COCO "tennis racket"
RACKET_CONFIDENCE = 0.1


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    confidence: float

    @property
    def foot(self) -> tuple[float, float]:
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


class PersonDetector:
    """Decode YOLOX-S with the RGB preprocessing specified by OpenCV Zoo."""

    def __init__(self, model: Path, confidence: float = 0.35) -> None:
        if not 0 < confidence < 1:
            raise ValueError("Confidence threshold must lie between zero and one.")
        self.confidence = confidence
        self.net = cv2.dnn.readNetFromONNX(str(model))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        grids = []
        scales = []
        for stride in (8, 16, 32):
            yy, xx = np.mgrid[: 640 // stride, : 640 // stride]
            grid = np.column_stack((xx.ravel(), yy.ravel()))
            grids.append(grid)
            scales.append(np.full((len(grid), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(scales).astype(np.float32)

    def _forward(self, image: Image) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("Cannot detect people in an empty image.")
        ratio = min(640 / height, 640 / width)
        resized = cv2.resize(image, (int(width * ratio), int(height * ratio)))
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        self.net.setInput(cv2.dnn.blobFromImage(padded, swapRB=True, ddepth=cv2.CV_32F))
        output = self.net.forward()[0]
        if output.shape != (8400, 85):
            raise ValueError(f"Unexpected YOLOX output shape: {output.shape}")
        return output, ratio

    def _decode(self, output: np.ndarray, ratio: float, shape: tuple[int, ...], column: int,
                confidence: float, min_height: float) -> list[Detection]:
        height, width = shape[:2]
        scores = output[:, 4] * output[:, column]
        keep = scores >= confidence
        raw = output[keep]
        centers = (raw[:, :2] + self.grid[keep]) * self.strides[keep]
        sizes = np.exp(raw[:, 2:4]) * self.strides[keep]
        boxes = np.column_stack((centers - sizes / 2, centers + sizes / 2)) / ratio
        results = []
        for box, score in zip(boxes, scores[keep], strict=True):
            x1, y1, x2, y2 = box
            x1, x2 = np.clip((x1, x2), 0, width - 1)
            y1, y2 = np.clip((y1, y2), 0, height - 1)
            if x2 - x1 < 4 or y2 - y1 < min_height:
                continue
            results.append(Detection((float(x1), float(y1), float(x2), float(y2)), float(score)))
        if not results:
            return []
        indices = cv2.dnn.NMSBoxes(
            [[d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]] for d in results],
            [d.confidence for d in results], confidence, 0.5,
        )
        return [results[int(i)] for i in np.asarray(indices).reshape(-1)]

    def detect(self, image: Image) -> list[Detection]:
        output, ratio = self._forward(image)
        return self._decode(output, ratio, image.shape, 5, self.confidence, 12)

    def detect_with_rackets(self, image: Image) -> tuple[list[Detection], list[Detection]]:
        """People plus COCO 'tennis racket' boxes from the same forward pass."""
        output, ratio = self._forward(image)
        return (
            self._decode(output, ratio, image.shape, 5, self.confidence, 12),
            self._decode(output, ratio, image.shape, 5 + RACKET_CLASS, RACKET_CONFIDENCE, 4),
        )
