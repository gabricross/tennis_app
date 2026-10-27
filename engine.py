from __future__ import annotations

import bisect
import csv
import json
import logging
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

from config import FPS
from court import COURT_LINES, LENGTH, PLAY_BOUNDS, WIDTH, CourtCamera, court_matrix, project
from inference import Detection, ModelBank, PlayerSelector
from media import check_cancel, normalize
from kinematics import BALL_SPEED, player_kinematics

LOGGER = logging.getLogger("tennis.engine")
COLORS = {"north": (210, 224, 92), "south": (111, 161, 255)}


def encode_result(
    source: Path, destination: Path, samples: list[dict], frames: int,
    event: threading.Event, progress: Callable[[float, str], None], credit: str,
    ball_samples: list[dict] | None = None,
) -> None:
    capture = cv2.VideoCapture(str(source))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    error_path = destination.parent / "render.log"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-threads", "2", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    times = [sample["time"] for sample in samples]
    with error_path.open("wb") as error_stream:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_stream)
        try:
            if process.stdin is None:
                raise RuntimeError("No se ha podido iniciar la exportación de vídeo.")
            for index in range(frames):
                check_cancel(event)
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("El vídeo terminó antes de completar la exportación.")
                timestamp = index / FPS
                sample_index = max(0, bisect.bisect_right(times, timestamp) - 1)
                sample = samples[sample_index]
                following = samples[min(sample_index + 1, len(samples) - 1)]
                span = following["time"] - sample["time"]
                alpha = (timestamp - sample["time"]) / span if span else 0
                draw_frame(frame, sample, following, alpha, samples, sample_index, timestamp,
                           ball_samples=ball_samples, frame_index=index)
                if credit:
                    cv2.rectangle(frame, (0, height - 23), (width, height), (16, 29, 32), -1)
                    cv2.putText(frame, credit[:165], (9, height - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (190, 208, 210), 1, cv2.LINE_AA)
                process.stdin.write(frame.tobytes())
                if index == min(frames - 1, FPS):
                    cv2.imwrite(str(destination.parent / "poster.jpg"), frame)
                if index % 15 == 0:
                    progress(82 + 16 * index / frames, "Exportando el vídeo anotado")
            process.stdin.close()
            code = process.wait(timeout=60)
        except BrokenPipeError as error:
            raise RuntimeError("FFmpeg ha interrumpido la exportación. Consulta render.log.") from error
        finally:
            capture.release()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    if code:
        raise RuntimeError("No se ha podido completar el MP4: " + error_path.read_text(errors="replace")[-500:])


def draw_frame(frame, sample, following, alpha, samples, sample_index, timestamp,
               *, ball_samples=None, frame_index=0) -> None:
    height, width = frame.shape[:2]
    layer = frame.copy()
    valid = sample["camera_valid"] and following["camera_valid"] and following.get("camera_continuous", True)
    if valid:
        matrix = (1 - alpha) * np.array(sample["matrix"]) + alpha * np.array(following["matrix"])
        for line in COURT_LINES:
            points = np.rint(project(line, matrix)).astype(int)
            cv2.line(layer, tuple(points[0]), tuple(points[1]), (110, 230, 198), 1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.45, frame, 0.55, 0, frame)
    next_players = {player["side"]: player for player in following["players"]}
    players = []
    for source_player in sample["players"]:
        player = dict(source_player)
        if valid and player["side"] in next_players:
            for key in ("box", "pitch"):
                player[key] = (
                    (1 - alpha) * np.array(player[key])
                    + alpha * np.array(next_players[player["side"]][key])
                ).tolist()
        players.append(player)
        x1, y1, x2, y2 = map(round, player["box"])
        color = COLORS[player["side"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = "ARRIBA" if player["side"] == "north" else "ABAJO"
        cv2.putText(frame, label, (x1, max(20, y1 - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 2, cv2.LINE_AA)
        motion = player.get("motion") if valid else None
        if motion:
            speed = motion["speed_kmh"]
            distance = motion["distance_m"]
            detail = (f"{speed:.1f} km/h est." if speed is not None else "km/h --")
            detail += f" | {distance:.1f} m obs." if distance is not None else " | m --"
            cv2.putText(frame, detail, (max(0, min(x1, width - 260)), min(height - 35, y2 + 23)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
        cv2.ellipse(frame, ((x1 + x2) // 2, y2), (11, 4), 0, 0, 360, color, 2)
    ball_series = samples if ball_samples is None else ball_samples
    ball_index = sample_index if ball_samples is None else frame_index
    point = ball_series[ball_index]["ball"]
    if point is not None:
        history = []
        for entry in reversed(ball_series[max(0, ball_index - 6):ball_index + 1]):
            observation = entry["ball"]
            if observation is None or observation.get("track_id") != point.get("track_id"):
                break
            history.append((entry["time"], observation))
            if not observation.get("continuous", False):
                break
        history.reverse()
        for left, right in zip(history, history[1:]):
            dt = right[0] - left[0]
            distance = np.hypot((right[1]["x"] - left[1]["x"]) * 640 / width,
                                (right[1]["y"] - left[1]["y"]) * 360 / height)
            if dt <= 0.05 and distance <= 12 + 1200 * dt:
                cv2.line(frame, (round(left[1]["x"]), round(left[1]["y"])),
                         (round(right[1]["x"]), round(right[1]["y"])), (55, 240, 245), 2, cv2.LINE_AA)
        cv2.circle(frame, (round(point["x"]), round(point["y"])), 5, (55, 240, 245), 2)
    map_width = max(112, round(width * 0.13))
    left, top, right, bottom = PLAY_BOUNDS
    map_scale = map_width / (right - left)
    map_height = round((bottom - top) * map_scale)
    x, y = width - map_width - 22, 57
    def map_point(position):
        return (x + round((position[0] - left) * map_scale),
                y + round((position[1] - top) * map_scale))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 30), (x + map_width + 10, y + map_height + 31), (23, 43, 43), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    ink = (125, 161, 150)
    cv2.rectangle(frame, map_point((0, 0)), map_point((WIDTH, LENGTH)), ink, 1)
    for line in COURT_LINES:
        cv2.line(frame, map_point(line[0]), map_point(line[1]), ink, 1)
    cv2.line(frame, map_point((0, LENGTH / 2)), map_point((WIDTH, LENGTH / 2)), ink, 1)
    if valid:
        for player in players:
            px, py = player["pitch"]
            if left <= px <= right and top <= py <= bottom:
                cv2.circle(frame, map_point((px, py)), 4, COLORS[player["side"]], -1, cv2.LINE_AA)
    cv2.putText(frame, "PISTA" if valid else "SIN REGISTRO", (x - 1, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (215, 235, 230), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, 34), (16, 29, 32), -1)
    state = "REGISTRO ACTIVO" if valid else "CAMARA NO VALIDA - MAPA OCULTO"
    label = f"TENNIS LAB  |  {timestamp:05.1f}s  |  {state}  |  ANALISIS LOCAL"
    cv2.putText(frame, label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 235, 230), 1, cv2.LINE_AA)
    if any("motion" in p for p in sample["players"]):
        cv2.putText(frame, "PELOTA: km/h NO DISPONIBLES (sin 3D)", (12, height - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (215, 235, 230), 1, cv2.LINE_AA)


def summarize(samples: list[dict], mode: str, duration: float, elapsed: float,
              ball_samples: list[dict] | None = None) -> dict:
    total = len(samples)
    camera_coverage = sum(s["camera_valid"] for s in samples) / total
    players = {}
    warnings = []
    for side in ("north", "south"):
        observations = [p for s in samples for p in s["players"] if p["side"] == side]
        coverage = len(observations) / total
        zones = {"baseline": 0, "middle": 0, "net": 0}
        for observation in observations:
            y = observation["pitch"][1]
            distance_to_net = abs(y - LENGTH / 2)
            zone = "net" if distance_to_net < 4 else "middle" if distance_to_net < 8 else "baseline"
            zones[zone] += 1
        players[side] = {
            "observations": len(observations), "coverage": coverage,
            "mean_detector_score": float(np.mean([p["confidence"] for p in observations])) if observations else None,
            "zones": {name: count / len(observations) if observations else 0 for name, count in zones.items()},
        }
        if coverage < 0.75:
            warnings.append(f"Seguimiento incompleto del jugador {'de arriba' if side == 'north' else 'de abajo'} ({coverage:.0%} de muestras).")
    if camera_coverage < 0.9:
        warnings.append("Se ha perdido la calibración durante parte del clip. No se proyectan esas posiciones.")
    ball_series = samples if ball_samples is None else ball_samples
    ball_observations = sum(sample["ball"] is not None for sample in ball_series)
    ball_total = len(ball_series)
    if mode == "complete" and ball_observations / ball_total < 0.5:
        warnings.append("Pelota experimental: detecciones escasas o intermitentes. No interpretes la estela como una trayectoria completa.")
    if mode == "complete":
        warnings.append("La pelota se representa en píxeles de imagen, no en 3D ni como bote sobre la pista.")
    if not any(p["observations"] for p in players.values()):
        raise ValueError("No se han observado jugadores con esta calibración. Revisa las esquinas o elige una toma de individuales con la pista completa.")
    return {
        "duration": duration, "samples": total, "camera_coverage": camera_coverage,
        "players": players, "ball_observations": ball_observations,
        "ball_samples": ball_total if mode == "complete" else 0,
        "ball_coverage": ball_observations / ball_total if mode == "complete" else None,
        "ball_methods": {
            method: sum(s["ball"] is not None and s["ball"].get("method") == method for s in ball_series)
            for method in ("TrackNet + temporal association", "bidirectional optical flow",
                           "YOLOv8n + temporal association")
        } if mode == "complete" else {},
        "processing_seconds": elapsed, "mode": mode, "warnings": warnings,
        "metrics_note": "Cobertura = muestras observadas / muestras analizadas; pelota por fotograma, "
        "con el método de cada observación separado en JSON/CSV. No mide precisión. "
        "Las zonas se calculan solo entre observaciones válidas. Las métricas de movimiento, cuando existen, "
        "son estimaciones del apoyo sobre el suelo, no mediciones certificadas. Sin velocidad de pelota, marcador, botes ni decisiones arbitrales.",
    }


def analyze_clip(
    source: Path, directory: Path, settings: dict, event: threading.Event,
    progress: Callable[[float, str], None], models: ModelBank, source_credit: dict | None = None,
    source_notice: str | None = None,
) -> dict:
    started = time.perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.mp4"
    normalize(source, original, settings["start"], settings["duration"], event, progress)
    capture = cv2.VideoCapture(str(original))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = capture.read()
    if not success or total < FPS // 2:
        raise RuntimeError("No se ha podido preparar un fragmento de vídeo válido.")
    height, width = frame.shape[:2]
    reference = settings.get("calibration_reference", "doubles")
    matrix = court_matrix(settings["corners"], (width, height), reference)
    selector = PlayerSelector()
    people = models.person_detector()
    initial_detections = people.detect(frame)
    camera = CourtCamera(
        frame, matrix, allow_partial=reference != "doubles",
        boxes=[detection.box for detection in initial_detections],
    )
    ball_backend = settings.get("ball_backend", "tracknet")
    ball = models.ball_detector(ball_backend) if settings["mode"] == "complete" else None
    frames = deque(maxlen=3)
    stride = 3 if ball else 6
    samples = []
    ball_samples = [] if ball else None
    previous_boxes = [detection.box for detection in initial_detections]
    camera_continuous = True
    for index in range(total):
        check_cancel(event)
        if index:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("El vídeo normalizado se interrumpió durante el análisis.")
        frames.append(frame.copy())
        timestamp = index / FPS
        point = ball.detect(frames, timestamp, infer=index % 3 == 2) if ball else None
        if ball_samples is not None:
            ball_samples.append({"time": round(timestamp, 6), "frame": index, "ball": point})
        if index % stride and index != total - 1:
            state = camera.update(frame, previous_boxes)
            matrix = state.matrix
            camera_continuous = camera_continuous and state.valid
            continue
        detections = initial_detections if index == 0 else people.detect(frame)
        if index == 0:
            valid, reason, support = True, "Calibración confirmada por el usuario", camera.initial_support
        else:
            state = camera.update(frame, [d.box for d in detections])
            matrix, valid, reason, support = state.matrix, state.valid, state.reason, state.support
        provisional = selector.select(detections, matrix, timestamp, commit=False) if valid else []
        if valid and not any(player["side"] == "north" for player in provisional):
            prior = selector.last_boxes.get("north")
            if prior is not None and timestamp - prior[0] <= 0.6:
                x1, y1, x2, y2 = prior[1]
                center = (x1 + x2) / 2
                radius = max(160, (x2 - x1) * 4)
                left, right = max(0, round(center - radius)), min(width, round(center + radius))
                top, bottom = max(0, round(y1 - 80)), min(height, round(y2 + 100))
                extra = people.detect(frame[top:bottom, left:right])
                detections.extend([
                    Detection((d.box[0] + left, d.box[1] + top, d.box[2] + left, d.box[3] + top), d.confidence)
                    for d in extra
                ])
                provisional = selector.select(detections, matrix, timestamp, commit=False)
        if valid and not any(player["side"] == "north" for player in provisional):
            roi = project([[-2, -5], [WIDTH + 2, -5], [WIDTH + 2, LENGTH / 2], [-2, LENGTH / 2]], matrix)
            x1, y1 = np.maximum(0, np.floor(roi.min(axis=0))).astype(int)
            x2, y2 = np.minimum([width, height], np.ceil(roi.max(axis=0))).astype(int)
            if x2 - x1 > 50 and y2 - y1 > 40:
                extra = people.detect(frame[y1:y2, x1:x2])
                detections.extend([
                    Detection((d.box[0] + x1, d.box[1] + y1, d.box[2] + x1, d.box[3] + y1), d.confidence)
                    for d in extra
                ])
        selected = selector.select(detections, matrix, timestamp) if valid else []
        previous_boxes = [detection.box for detection in detections]
        samples.append({
            "time": round(timestamp, 6), "frame": index, "camera_valid": valid,
            "camera_continuous": camera_continuous and valid,
            "camera_reason": reason, "line_support": support,
            "matrix": matrix.tolist() if valid else None, "players": selected, "ball": point,
        })
        camera_continuous = valid
        progress(12 + 67 * (index + 1) / total,
                 "Analizando jugadores y pelota" if ball else "Analizando jugadores y cámara")
    capture.release()
    check_cancel(event)
    movement = player_kinematics(samples)
    summary = summarize(samples, settings["mode"], total / FPS, time.perf_counter() - started, ball_samples)
    for side, metrics in movement.items():
        summary["players"][side]["movement"] = metrics
    summary["ball_speed"] = BALL_SPEED.copy()
    summary["warnings"].append(
        "Velocidad y distancia de jugadores estimadas con apoyos y dimensiones de pista. "
        "Se filtra el ruido y se excluyen huecos, saltos y calibraciones insuficientes; "
        "la distancia observada no es el recorrido completo si falta seguimiento. "
        "La velocidad de pelota no está disponible sin calibración 3D o mediciones adicionales."
    )
    if reference != "doubles":
        summary["warnings"].insert(
            0, "Calibración parcial manual: la proyección fuera de las referencias visibles "
            "es una extrapolación geométrica, no una medición de precisión."
        )
    if source_notice:
        summary["warnings"].insert(0, source_notice)
    if ball is not None:
        summary["ball_backend"] = ball_backend
        summary["warnings"].append(
            "La pelota es un candidato visual experimental, no una identidad garantizada; "
            "puede confundirse con pelotas de reserva y otros objetos."
        )
    credit_text = ""
    if source_credit:
        provider = source_credit.get("provider", "Wikimedia Commons")
        credit_text = f"Source: {source_credit['author']} | {provider} | {source_credit['license']} | cropped, normalized, analysis overlays"
    encode_result(original, directory / "analysis.mp4", samples, total, event, progress, credit_text, ball_samples)
    summary["total_seconds_including_render"] = time.perf_counter() - started
    result = {
        "version": 3, "width": width, "height": height, "fps": FPS, "frames": total,
        "court_geometry": {"width_m": WIDTH, "length_m": LENGTH, "service_distance_from_net_m": 6.4},
        "config": settings, "summary": summary, "source_credit": source_credit,
        "source_notice": source_notice, "samples": samples, "ball_samples": ball_samples,
    }
    (directory / "analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (directory / "observations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seconds", "player_side", "x_reference", "y_reference", "detector_score", "observed",
                         "camera_valid", "estimated_speed_kmh", "observed_distance_m", "motion_status"])
        for sample in samples:
            for player in sample["players"]:
                motion = player["motion"]
                writer.writerow([sample["time"], player["side"], *player["pitch"], player["confidence"], True,
                                 sample["camera_valid"], motion["speed_kmh"], motion["distance_m"], motion["status"]])
    if ball:
        with (directory / "ball.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["seconds", "x_image_pixels", "y_image_pixels", "intensity_class", "observed", "method",
                             "detector_score", "track_id", "continuous", "speed_kmh"])
            for sample in ball_samples if ball_samples is not None else []:
                point = sample["ball"]
                if point is not None:
                    writer.writerow([sample["time"], point["x"], point["y"], point["intensity"],
                                     True, point["method"], point.get("detector_score"), point["track_id"],
                                     point["continuous"], None])
    return summary
