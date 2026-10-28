from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import least_squares

from config import MODELS

WIDTH, LENGTH = 10.97, 23.77
SERVICE_NEAR_NET = 6.40
SERVICE_TOP, SERVICE_BOTTOM = LENGTH / 2 - SERVICE_NEAR_NET, LENGTH / 2 + SERVICE_NEAR_NET
PLAY_BOUNDS = (-4.0, -7.0, WIDTH + 4.0, LENGTH + 7.0)
WORLD_CORNERS = np.array([[0, 0], [WIDTH, 0], [WIDTH, LENGTH], [0, LENGTH]], np.float32)
CALIBRATION_REFERENCES = {
    "doubles": WORLD_CORNERS,
    "service_boxes": np.array(
        [[1.37, SERVICE_TOP], [9.60, SERVICE_TOP], [9.60, SERVICE_BOTTOM], [1.37, SERVICE_BOTTOM]], np.float32
    ),
    "service_left": np.array(
        [[1.37, SERVICE_TOP], [WIDTH / 2, SERVICE_TOP], [WIDTH / 2, SERVICE_BOTTOM], [1.37, SERVICE_BOTTOM]], np.float32
    ),
    "service_right": np.array(
        [[WIDTH / 2, SERVICE_TOP], [9.60, SERVICE_TOP], [9.60, SERVICE_BOTTOM], [WIDTH / 2, SERVICE_BOTTOM]], np.float32
    ),
}
COURT_LINES = [
    [[0, 0], [WIDTH, 0]], [[0, LENGTH], [WIDTH, LENGTH]],
    [[0, 0], [0, LENGTH]], [[WIDTH, 0], [WIDTH, LENGTH]],
    [[1.37, 0], [1.37, LENGTH]], [[9.60, 0], [9.60, LENGTH]],
    [[1.37, SERVICE_TOP], [9.60, SERVICE_TOP]], [[1.37, SERVICE_BOTTOM], [9.60, SERVICE_BOTTOM]],
    [[WIDTH / 2, SERVICE_TOP], [WIDTH / 2, SERVICE_BOTTOM]],
]
WORLD_LINE_SAMPLES = np.concatenate(
    [np.linspace(np.array(a), np.array(b), 32) for a, b in COURT_LINES]
).astype(np.float32)


def session(path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def project(points, matrix) -> np.ndarray:
    values = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(values, np.asarray(matrix, np.float64)).reshape(-1, 2)


def validate_corners(corners) -> np.ndarray:
    values = np.asarray(corners, np.float32)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("Marca exactamente cuatro esquinas válidas.")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Las esquinas deben estar dentro del fotograma.")
    if not cv2.isContourConvex(values.reshape(-1, 1, 2)):
        raise ValueError("Las esquinas se cruzan. Sigue el orden indicado.")
    if abs(cv2.contourArea(values)) < 0.035:
        raise ValueError("La pista ocupa muy poco espacio para analizarla con fiabilidad.")
    tl, tr, br, bl = values
    if not tl[0] < tr[0] or not bl[0] < br[0] or np.mean(values[:2, 1]) >= np.mean(values[2:, 1]):
        raise ValueError("Orden incorrecto: arriba izquierda, arriba derecha, abajo derecha y abajo izquierda.")
    if min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) < 0.08:
        raise ValueError("La pista está demasiado estrecha o las esquinas no corresponden.")
    return values


def court_matrix(corners, size: tuple[int, int], reference: str = "doubles") -> np.ndarray:
    if reference not in CALIBRATION_REFERENCES:
        raise ValueError("Referencia de calibración desconocida.")
    values = validate_corners(corners)
    matrix = cv2.getPerspectiveTransform(
        CALIBRATION_REFERENCES[reference], values * np.array(size, np.float32)
    )
    depth = np.column_stack((WORLD_CORNERS, np.ones(4))) @ matrix[2]
    if np.min(np.abs(depth)) < 1e-8 or np.any(np.sign(depth) != np.sign(depth[0])):
        raise ValueError(
            "Las referencias proyectan parte de la pista detrás de la cámara. "
            "Revisa los puntos o usa una toma más elevada."
        )
    return matrix


def white_distance(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 155]), np.array([179, 40, 255]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = max(9, round(image.shape[1] / 40) | 1)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((size, size), np.uint8))
    mask = cv2.bitwise_and(mask, np.where(contrast >= 12, 255, 0).astype(np.uint8))
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


def line_support(
    image: np.ndarray, matrix: np.ndarray, tolerance: float = 5.0, *, allow_partial: bool = False
) -> float:
    distances = white_distance(image)
    points = project(WORLD_LINE_SAMPLES, matrix)
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < image.shape[1])
        & (points[:, 1] >= 0) & (points[:, 1] < image.shape[0])
    )
    minimum_visible = 0.2 if allow_partial else 0.8
    if np.mean(inside) < minimum_visible:
        return 0.0
    xy = np.rint(points[inside]).astype(int)
    xy[:, 0] = np.clip(xy[:, 0], 0, image.shape[1] - 1)
    xy[:, 1] = np.clip(xy[:, 1], 0, image.shape[0] - 1)
    return float(np.mean(distances[xy[:, 1], xy[:, 0]] <= tolerance))


def calibration_preview(image: np.ndarray, corners, reference: str = "doubles") -> dict:
    height, width = image.shape[:2]
    matrix = court_matrix(corners, (width, height), reference)
    scale = np.array([width, height])
    outline = project(WORLD_CORNERS, matrix) / scale
    support = line_support(image, matrix, 6, allow_partial=reference != "doubles")
    return {
        "reference": reference,
        "partial": reference != "doubles",
        "lines": [(project(line, matrix) / scale).tolist() for line in COURT_LINES],
        "court_corners": outline.tolist(),
        "visible_corners": int(np.all((outline >= -1e-5) & (outline <= 1 + 1e-5), axis=1).sum()),
        "support": support,
        "usable": support >= 0.3,
    }


class CourtModel:
    def __init__(self) -> None:
        self.net = session(MODELS / "court.onnx")
        self.lock = threading.Lock()

    def propose(self, image: np.ndarray) -> dict:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image
        resized = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.Resampling.BILINEAR))
        values = resized.astype(np.float32) / 255
        values = (values - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        blob = values.transpose(2, 0, 1)[None].astype(np.float32)
        with self.lock:
            points = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].reshape(14, 2) / 224
        corners = points[[0, 1, 3, 2]]
        try:
            validate_corners(corners)
        except ValueError as error:
            return {"corners": None, "support": 0, "message": str(error), "method": "manual_required"}
        width, height = image.shape[1], image.shape[0]
        distances = white_distance(image)
        initial = corners.copy()

        def residual(parameters):
            quad = parameters.reshape(4, 2)
            matrix = cv2.getPerspectiveTransform(
                WORLD_CORNERS, (quad * [width, height]).astype(np.float32)
            )
            xy = project(WORLD_LINE_SAMPLES, matrix)
            x = np.clip(np.rint(xy[:, 0]), 0, width - 1).astype(int)
            y = np.clip(np.rint(xy[:, 1]), 0, height - 1).astype(int)
            return np.concatenate(
                [np.minimum(distances[y, x], 25) / 5, (parameters - initial.ravel()) * 80]
            )

        bounds = (np.maximum(0, initial.ravel() - 0.06), np.minimum(1, initial.ravel() + 0.06))
        fit = least_squares(residual, initial.ravel(), bounds=bounds, max_nfev=35, diff_step=0.002)
        fitted = fit.x.reshape(4, 2).astype(np.float32)
        try:
            validate_corners(fitted)
        except ValueError:
            fitted = initial
        initial_support = line_support(image, court_matrix(initial, (width, height)))
        fitted_support = line_support(image, court_matrix(fitted, (width, height)))
        if fitted_support > initial_support:
            corners, support = fitted, fitted_support
        else:
            corners, support = initial, initial_support
        return {
            "corners": corners.tolist(), "support": support, "method": "model_proposal",
            "message": "Propuesta automática: revisa las cuatro esquinas antes de confirmar. "
            "El apoyo sobre líneas no es una medida de precisión.",
        }


@dataclass
class CameraState:
    matrix: np.ndarray
    valid: bool
    reason: str
    support: float


class CourtCamera:
    def __init__(
        self, image: np.ndarray, matrix: np.ndarray, *, allow_partial: bool = False,
        boxes: list | None = None,
    ) -> None:
        self.scale = 640 / image.shape[1]
        self.gray = self._gray(image)
        self.matrix = matrix
        self.previous_image = image
        self.features = self._features(image, boxes if boxes is not None else [])
        self.allow_partial = allow_partial
        self.initial_support = line_support(image, matrix, 6, allow_partial=allow_partial)
        self.lost = False
        self.reference_matrix = matrix.copy()
        self.recovery_wait = 0
        self.orb = cv2.ORB_create(nfeatures=1800, edgeThreshold=15, fastThreshold=10)
        self.reference_keypoints, self.reference_descriptors = self.orb.detectAndCompute(
            self.gray, self._static_mask(self.gray.shape, boxes if boxes is not None else [])
        )

    def _gray(self, image):
        return cv2.cvtColor(cv2.resize(image, None, fx=self.scale, fy=self.scale), cv2.COLOR_BGR2GRAY)

    def _static_mask(self, shape, boxes):
        mask = np.full(shape, 255, np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = np.rint(np.array(box) * self.scale).astype(int)
            cv2.rectangle(mask, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
        return mask

    def _features(self, image, boxes):
        gray = self._gray(image)
        polygon = project(WORLD_CORNERS, self.matrix) * self.scale
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, polygon.astype(np.int32), 255)
        mask = cv2.dilate(mask, np.ones((21, 21), np.uint8))
        mask = cv2.bitwise_and(mask, self._static_mask(gray.shape, boxes))
        return cv2.goodFeaturesToTrack(gray, 400, 0.008, 7, mask=mask)

    def _recover(self, image, boxes) -> CameraState:
        unavailable = CameraState(self.matrix, False, "Sin correspondencias estáticas fiables", 0)
        if self.recovery_wait:
            self.recovery_wait -= 1
            return unavailable
        self.recovery_wait = 5
        if self.reference_descriptors is None:
            return unavailable
        gray = self._gray(image)
        keypoints, descriptors = self.orb.detectAndCompute(
            gray, self._static_mask(gray.shape, boxes)
        )
        if descriptors is None or len(descriptors) < 20:
            return unavailable
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
            self.reference_descriptors, descriptors, k=2
        )
        matches = [
            pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
        ]
        if len(matches) < 20:
            return unavailable
        reference = np.float32([self.reference_keypoints[m.queryIdx].pt for m in matches])
        current = np.float32([keypoints[m.trainIdx].pt for m in matches])
        motion, accepted = cv2.findHomography(reference, current, cv2.RANSAC, 2, confidence=0.995)
        if motion is None or accepted is None or np.mean(accepted) < 0.6:
            return unavailable
        inliers = reference[accepted.ravel() == 1]
        if len(inliers) < 15 or cv2.contourArea(cv2.convexHull(inliers)) < gray.size * 0.02:
            return unavailable
        height, width = gray.shape
        viewport = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        transformed = project(viewport, motion).astype(np.float32)
        area_ratio = abs(cv2.contourArea(transformed)) / gray.size
        if not cv2.isContourConvex(transformed) or not 0.5 < area_ratio < 2:
            return unavailable
        scale = np.diag([self.scale, self.scale, 1])
        candidate = np.linalg.inv(scale) @ motion @ scale @ self.reference_matrix
        support = line_support(image, candidate, 7, allow_partial=self.allow_partial)
        if support < max(0.3, self.initial_support * 0.55):
            return CameraState(self.matrix, False, "Reencuadre rechazado: líneas no alineadas", support)
        self.matrix = candidate
        self.gray = gray
        self.features = self._features(image, boxes)
        self.lost = False
        self.recovery_wait = 0
        return CameraState(candidate, True, "Registro recuperado con referencias ORB + RANSAC", support)

    def update(self, image, boxes) -> CameraState:
        if self.lost:
            return self._recover(image, boxes)
        gray = self._gray(image)
        if self.features is None or len(self.features) < 12:
            self.lost = True
            return CameraState(self.matrix, False, "Textura insuficiente para validar la cámara", 0)
        points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.features, None, winSize=(25, 25), maxLevel=3
        )
        if points is None or status is None:
            self.lost = True
            return CameraState(self.matrix, False, "Flujo óptico no disponible", 0)
        good = status.ravel() == 1
        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, points, None, winSize=(25, 25), maxLevel=3
        )
        if backward is None or backward_status is None:
            self.lost = True
            return CameraState(self.matrix, False, "No se puede verificar el flujo inverso", 0)
        backward_error = np.linalg.norm(self.features - backward, axis=2).ravel()
        good &= (backward_status.ravel() == 1) & (backward_error < 1.5)
        old, new = self.features[good].reshape(-1, 2), points[good].reshape(-1, 2)
        if len(old) < 12:
            self.lost = True
            return CameraState(self.matrix, False, "Se han perdido las referencias de la pista", 0)
        motion, accepted = cv2.findHomography(old, new, cv2.RANSAC, 2, confidence=0.995)
        if motion is None or accepted is None or np.mean(accepted) < 0.55:
            self.lost = True
            return CameraState(self.matrix, False, "El movimiento no supera los controles geométricos", 0)
        displacement = np.median(np.linalg.norm(old - new, axis=1))
        inliers = old[accepted.ravel() == 1]
        if len(inliers) < 12 or cv2.contourArea(cv2.convexHull(inliers)) < gray.size * 0.002:
            self.lost = True
            return CameraState(self.matrix, False, "Referencias estáticas demasiado concentradas", 0)
        height, width = gray.shape
        viewport = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
        transformed = project(viewport, motion).astype(np.float32)
        area_ratio = abs(cv2.contourArea(transformed)) / gray.size
        scale_valid = cv2.isContourConvex(transformed) and 0.65 < area_ratio < 1.55
        if displacement > 90 or not scale_valid:
            self.lost = True
            return CameraState(self.matrix, False, "Corte o zoom demasiado grande", 0)
        scale = np.diag([self.scale, self.scale, 1])
        candidate = np.linalg.inv(scale) @ motion @ scale @ self.matrix
        support = line_support(image, candidate, 7, allow_partial=self.allow_partial)
        if self.initial_support > 0.3 and support < max(0.12, self.initial_support * 0.4):
            self.lost = True
            return CameraState(self.matrix, False, "Las líneas ya no respaldan la proyección", support)
        self.matrix = candidate
        self.gray = gray
        self.features = self._features(image, boxes)
        self.previous_image = image
        return CameraState(candidate, True, "Registro activo", support)
