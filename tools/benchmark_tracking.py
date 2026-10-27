from __future__ import annotations

import argparse
import csv
import html
import json
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import FPS, HOME, MODELS
from engine import analyze_clip, encode_result
from inference import BallModel, BallObjectDetector, ModelBank, YoloBallModel, ball_candidates
from media import normalize
from setup_models import sha256

ROOT = HOME / "tracking_benchmark"


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_sources() -> list[dict]:
    sources = json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))
    if len(sources) != 10 or len({s["original_sha256"] for s in sources}) != 10:
        raise ValueError("The benchmark requires ten distinct recordings.")
    return sources


def settings(source: dict) -> dict:
    return {
        "start": {"night": 5, "rola": 2, "audience": 2}.get(source["id"], 0),
        "duration": 4, "mode": "complete",
    }


def prepare(identifiers: list[str] | None = None) -> None:
    for source in load_sources():
        if identifiers and source["id"] not in identifiers:
            continue
        directory = ROOT / source["id"] / "review"
        directory.mkdir(exist_ok=True)
        config = settings(source)
        normalize(Path(source["path"]), directory / "original.mp4", config["start"],
                  config["duration"], threading.Event(), lambda *_: None)
        capture = cv2.VideoCapture(str(directory / "original.mp4"))
        images = []
        targets = (15, 45, 75, 105)
        for index in range(120):
            success, image = capture.read()
            if not success:
                raise RuntimeError(f"Truncated review: {source['id']} at frame {index}")
            if index not in targets:
                continue
            image = cv2.resize(image, (640, 360))
            cv2.putText(image, f"{source['id']} f{index}", (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            images.append(image)
        capture.release()
        if not cv2.imwrite(str(directory / "labels.jpg"),
                           np.vstack([np.hstack(images[:2]), np.hstack(images[2:])])):
            raise RuntimeError("Cannot write label review.")
        print("Prepared", source["id"], flush=True)


class RecordedBall(BallModel):
    def __init__(self) -> None:
        super().__init__()
        self.maps: list[np.ndarray] = []
        self.neural_seconds: list[float] = []

    def heatmap(self, frames: deque[np.ndarray]) -> np.ndarray:
        started = time.perf_counter()
        intensity = super().heatmap(frames)
        self.neural_seconds.append(time.perf_counter() - started)
        self.maps.append(intensity.copy())
        return intensity


class RecordedObjectDetector(BallObjectDetector):
    def __init__(self) -> None:
        super().__init__(MODELS / "ball_yolo.onnx")
        self.neural_seconds: list[float] = []
        self.records: list[list[dict]] = []

    def candidates(self, image: np.ndarray) -> list[dict]:
        started = time.perf_counter()
        result = super().candidates(image)
        self.neural_seconds.append(time.perf_counter() - started)
        self.records.append(result)
        return result


class RecordedYolo(YoloBallModel):
    def __init__(self) -> None:
        self.detector = RecordedObjectDetector()
        self.reset()


def legacy_track(maps: list[np.ndarray], size: tuple[int, int]) -> list[dict]:
    previous = None
    result = []
    for index, intensity in enumerate(maps):
        timestamp = (2 + index * 3) / FPS
        candidates = []
        for candidate in ball_candidates(intensity, 150):
            position = np.array([candidate["x"], candidate["y"]])
            cost = -candidate["intensity"] / 255
            if previous is not None and timestamp - previous[0] < 0.4:
                distance = float(np.linalg.norm(position - previous[1]))
                if distance > 210:
                    continue
                cost += distance / 120
            candidates.append((cost, position))
        point = None
        if candidates:
            _, position = min(candidates, key=lambda value: value[0])
            previous = timestamp, position
            point = {"x": float(position[0] / 640 * size[0]),
                     "y": float(position[1] / 360 * size[1])}
        result.append({"time": timestamp, "frame": 2 + index * 3, "ball": point})
    return result


def worker(identifier: str, calibration: Path, output: Path, backend: str) -> None:
    source = next(s for s in load_sources() if s["id"] == identifier)
    review = json.loads(calibration.read_text(encoding="utf-8"))[identifier]
    directory = output / identifier
    directory.mkdir(parents=True, exist_ok=True)
    config = settings(source)
    config["ball_backend"] = backend
    started = time.perf_counter()
    models = ModelBank()
    models.ball = RecordedBall() if backend == "tracknet" else RecordedYolo()
    models.ball_backend = backend
    if review.get("corners") is not None:
        config.update({"corners": review["corners"],
                       "calibration_reference": review.get("reference", "doubles")})
        summary = analyze_clip(
            Path(source["path"]), directory, config, threading.Event(),
            lambda *_: None, models, source["source_credit"], review["assessment"],
        )
        result = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
        status = "calibrated"
    else:
        normalize(Path(source["path"]), directory / "original.mp4", config["start"],
                  config["duration"], threading.Event(), lambda *_: None)
        capture = cv2.VideoCapture(str(directory / "original.mp4"))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(3)), int(capture.get(4))
        samples, frames, person_counts = [], deque(maxlen=3), []
        ball = models.ball_detector(backend)
        people = models.person_detector()
        for index in range(total):
            success, image = capture.read()
            if not success:
                raise RuntimeError("Truncated normalized benchmark video.")
            frames.append(image)
            timestamp = index / FPS
            point = ball.detect(frames, timestamp, infer=index % 3 == 2)
            if index % 3 == 0:
                person_counts.append(len(people.detect(image)))
            samples.append({
                "frame": index, "time": timestamp, "ball": point, "players": [],
                "camera_valid": False, "matrix": None,
            })
        capture.release()
        encode_result(directory / "original.mp4", directory / "analysis.mp4", samples, total,
                      threading.Event(), lambda *_: None,
                      f"{source['source_credit']['author']} | {source['source_credit']['license']}")
        observed = [s for s in samples if s["ball"] is not None]
        summary = {
            "duration": total / FPS, "samples": total, "camera_coverage": None,
            "players": None, "ball_observations": len(observed), "ball_samples": total,
            "ball_coverage": len(observed) / total,
            "ball_methods": {method: sum(s["ball"]["method"] == method for s in observed)
                             for method in ("TrackNet + temporal association", "bidirectional optical flow",
                                            "YOLOv8n + temporal association")},
            "raw_person_detection_samples": len(person_counts),
            "mean_raw_person_count": float(np.mean(person_counts)),
            "warnings": [review["assessment"], "Solo pelota y detecciones de personas sin identidad. "
                         "Sin calibración: no se proyectan jugadores ni se valida la pista."],
        }
        result = {
            "width": width, "height": height, "frames": total, "fps": FPS,
            "config": config, "summary": summary, "samples": samples,
            "ball_samples": samples, "source_credit": source["source_credit"],
        }
        status = "ball_only_unscorable_court"
        save(directory / "analysis.json", result)
    elapsed = time.perf_counter() - started
    recorded = models.ball
    if isinstance(recorded, RecordedBall):
        np.savez_compressed(directory / "heatmaps.npz", *recorded.maps)
        legacy = legacy_track(recorded.maps, (result["width"], result["height"]))
        save(directory / "legacy.json", legacy)
        timings = recorded.neural_seconds
        legacy_coverage = sum(s["ball"] is not None for s in legacy) / len(legacy)
        new_coverage = sum(s["ball"] is not None for s in result["ball_samples"]
                           if s["frame"] % 3 == 2) / len(legacy)
    elif isinstance(recorded, RecordedYolo):
        timings = recorded.detector.neural_seconds
        save(directory / "yolo_candidates.json", recorded.detector.records)
        legacy_coverage = None
        new_coverage = summary["ball_coverage"]
    else:
        raise TypeError("Expected recorded ball model")
    save(directory / "metrics.json", {
        "id": identifier, "status": status, "assessment": review["assessment"],
        "video_seconds": summary["duration"], "processing_seconds": elapsed,
        "seconds_per_video_second": elapsed / summary["duration"],
        "processed_video_fps": result["frames"] / elapsed,
        "ball_backend": backend, "neural_calls": len(timings),
        "neural_seconds": sum(timings),
        "neural_median_seconds": float(np.median(timings)),
        "neural_p95_seconds": float(np.percentile(timings, 95)),
        "new_neural_coverage": new_coverage,
        "legacy_neural_coverage": legacy_coverage,
        "summary": summary, "config": config, "source_credit": source["source_credit"],
        "export_bytes": sum(p.stat().st_size for p in directory.iterdir() if p.suffix in (".mp4", ".json", ".csv")),
    })


def profile(command: list[str], log: Path) -> dict:
    cpu_by_pid = {}
    peak_rss = 0
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            rss = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                    cpu = child.cpu_times()
                    cpu_by_pid[child.pid] = cpu.user + cpu.system
                except psutil.NoSuchProcess:
                    continue
            peak_rss = max(peak_rss, rss)
            time.sleep(0.05)
    if process.returncode:
        raise RuntimeError(f"Worker failed ({process.returncode}); see {log}")
    return {
        "wall_seconds_including_startup": time.perf_counter() - started,
        "peak_process_tree_rss_mib": peak_rss / 1048576,
        "sampled_process_tree_cpu_seconds": sum(cpu_by_pid.values()),
        "sampling_interval_seconds": 0.05,
        "resource_note": "RSS summed across worker and FFmpeg children; shared pages may be counted twice. "
                         "CPU sampled per PID; very short-lived children and final <50ms may be missed.",
    }


def report(output: Path) -> None:
    rows = []
    for source in load_sources():
        metric_path = output / source["id"] / "metrics.json"
        if metric_path.exists():
            rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
    wall = sum(row.get("resources", {}).get("wall_seconds_including_startup", row["processing_seconds"]) for row in rows)
    quality_path = output / "quality.json"
    manifest = {
        "hardware": {"processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
                     "physical_cpus": psutil.cpu_count(logical=False),
                     "ram_gib": psutil.virtual_memory().total / 1073741824,
                     "os": platform.platform(), "python": platform.python_version()},
        "models": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                   for p in MODELS.glob("*.onnx")},
        "note": "Cohort of ten distinct short recordings; this report includes the completed recordings "
                "counted in aggregate. Single pass on a shared Windows desktop, not an isolated "
                "repeated laboratory timing or a population-wide accuracy study. "
                "Uncalibratable clips remain failures for court/player projection. "
                "Coverage is not accuracy. Only TrackNet legacy/new use identical neural frames "
                "(2,5,8,...) for a controlled association comparison; YOLO runs every frame. "
                "Review labels also informed diagnosis of association failures, so they are not a blind holdout. "
                "No phone, battery, electricity meter, paid API or cloud GPU was used.",
        "runtime_model_names": sorted(
            {"people.onnx"} | {
                "ball_yolo.onnx" if row["ball_backend"] == "yolo" else "ball.onnx" for row in rows
            }
        ),
        "model_note": "models lists installed ONNX files, not a simultaneous RAM requirement. "
                      "Manual calibration does not load the optional court proposal model.",
        "aggregate": {
            "completed_recordings": len(rows),
            "video_seconds": sum(row["video_seconds"] for row in rows),
            "worker_wall_seconds": wall,
            "peak_process_tree_rss_mib": max((row.get("resources", {}).get("peak_process_tree_rss_mib", 0) for row in rows), default=0),
            "exports_bytes": sum(row["export_bytes"] for row in rows),
            "calibrated_recordings": sum(row["status"] == "calibrated" for row in rows),
            "unscorable_court_recordings": sum(row["status"] != "calibrated" for row in rows),
        },
        "cost": {
            "paid_external_services_eur": 0, "electricity_measured": False, "phone_battery_measured": False,
            "illustrative_electricity_only": {
                "assumed_whole_computer_watts": 30, "assumed_eur_per_kwh": 0.25,
                "kwh": wall * 30 / 3600000, "eur": wall * 30 / 3600000 * 0.25,
                "note": "Scenario, NOT a power-meter reading; excludes downloads, preparation, idle time, "
                        "screen, other experiments, network charges and device wear.",
            },
        },
        "quality": json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None,
        "mobile": {
            "device_run": False, "native_app_built": False, "battery_or_thermals_measured": False,
            "static_check": "People/court supported well statically. TrackNet and fixed-shape YOLO "
                            "were not recommended by the ORT checker for NNAPI/CoreML due to graph partitions. "
                            "This is not proof that CPU/mobile execution is impossible, or a phone speed estimate.",
            "references": ["https://onnxruntime.ai/docs/tutorials/mobile/",
                           "https://onnxruntime.ai/docs/tutorials/mobile/helpers/model-usability-checker.html"],
        },
        "results": rows,
    }
    save(output / "report.json", manifest)
    with (output / "report.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "status", "video_s", "processing_s", "wall_s_including_startup", "time_multiplier",
                         "peak_rss_mib", "cpu_s", "ball_frame_coverage", "legacy_neural_coverage",
                         "new_neural_coverage", "export_bytes"])
        for row in rows:
            resources = row.get("resources", {})
            writer.writerow([row["id"], row["status"], row["video_seconds"], row["processing_seconds"],
                             resources.get("wall_seconds_including_startup"),
                             row["seconds_per_video_second"], resources.get("peak_process_tree_rss_mib"),
                             resources.get("sampled_process_tree_cpu_seconds"), row["summary"]["ball_coverage"],
                             row["legacy_neural_coverage"], row["new_neural_coverage"], row["export_bytes"]])
    cards = []
    for row in rows:
        escape = html.escape
        identifier = row["id"]
        credit = row["source_credit"]
        quality = manifest["quality"]["clips"].get(identifier) if manifest["quality"] else None
        quality_note = (
            f'<p>Revisión visible: {quality["localized_within_tolerance"]}/{quality["visible_labeled_frames"]} '
            f'localizadas; {quality["wrong_location"]} marcas incorrectas y '
            f'{quality["missing_on_visible_frames"]} huecos sobre esas etiquetas.</p>'
            if quality and quality["visible_labeled_frames"] else
            '<p>Sin posiciones visibles etiquetadas evaluables: no se calcula una tasa de acierto.</p>'
        )
        projection_note = '<p>Sin proyección de jugadores evaluable. No es un éxito completo.</p>'
        if row["status"] == "calibrated":
            summary = row["summary"]
            projection_note = (
                f'<p>Cobertura cámara {summary["camera_coverage"]:.1%}; jugador cercano '
                f'{summary["players"]["south"]["coverage"]:.1%}; lejano '
                f'{summary["players"]["north"]["coverage"]:.1%}. Cobertura, no precisión de posición.</p>'
            )
            for side, label in (("south", "Cercano"), ("north", "Lejano")):
                movement = summary["players"][side].get("movement")
                if movement is not None:
                    def number(value):
                        return f"{value:.1f}" if value is not None else "no disponible"
                    projection_note += (
                        f'<p>{label}: distancia observada {number(movement["distance_m"])} m; '
                        f'media {number(movement["mean_speed_kmh"])} km/h; pico estimado '
                        f'{number(movement["peak_speed_kmh"])} km/h sobre '
                        f'{movement["observed_seconds"]:.1f}s evaluables. No son medidas certificadas.</p>'
                    )
        cards.append(
            f'<article><h2>{escape(identifier)}</h2><p>{escape(row["assessment"])}</p>'
            f'<p>Clip {row["video_seconds"]:.1f}s | análisis {row["processing_seconds"]:.1f}s | '
            f'{row["seconds_per_video_second"]:.1f}x duración | '
            f'RAM pico {row.get("resources", {}).get("peak_process_tree_rss_mib", 0):.0f} MiB | '
            f'cobertura candidata pelota {row["summary"]["ball_coverage"]:.1%} (NO precisión)</p>'
            + quality_note + projection_note +
            f'<div class="videos"><video controls playsinline muted preload="metadata" src="{identifier}/original.mp4"></video>'
            f'<video controls playsinline muted preload="metadata" src="{identifier}/analysis.mp4"></video></div>'
            f'<p><a href="{escape(credit["source_url"], quote=True)}">{escape(credit["author"])}</a> | '
            f'<a href="{escape(credit["license_url"], quote=True)}">{escape(credit["license"])}</a>. '
            'Recorte, sin audio, normalización y anotaciones añadidas.</p>'
            f'<a href="{identifier}/analysis.json">Datos</a></article>'
        )
    aggregate = manifest["aggregate"]
    total_quality = manifest["quality"].get("totals") if manifest["quality"] else None
    quality_text = (
        f'<p>Revisión visual limitada: {total_quality["localized_within_tolerance"]}/'
        f'{total_quality["visible_labeled_frames"]} posiciones visibles localizadas a ≤8 px en una imagen '
        '640×360. Los fotogramas ambiguos se excluyen, no se cuentan como aciertos. '
        'Estas etiquetas también sirvieron para diagnosticar fallos: no son un test ciego. '
        'No equivale a precisión global ni a una validación independiente completa.</p>'
    ) if total_quality else '<p>La cobertura aún no está contrastada con las etiquetas de revisión.</p>'
    (output / "VER_RESULTADOS.html").write_text(
        '<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<title>Tennis Lab - {len(rows)} pruebas reales</title><style>body{{font:16px system-ui;background:#102126;color:#edf5ed;'
        'max-width:1400px;margin:auto;padding:24px}article{padding:20px;margin:25px 0;background:#172d32;border-radius:12px}'
        '.videos{display:grid;grid-template-columns:1fr 1fr;gap:14px}video{width:100%;max-height:650px;background:#000}'
        'a{color:#8ce8ce}p{line-height:1.6}@media(max-width:700px){.videos{grid-template-columns:1fr}}</style>'
        '<h1>Tennis Lab: pruebas de seguimiento</h1><p>Izquierda: original. Derecha: análisis real. '
        'Los vídeos sin calibración válida NO muestran jugadores proyectados. La pelota sigue siendo experimental. '
        'Las marcas pueden ser erróneas. Cobertura no significa precisión. '
        'Velocidad de pelota no disponible: requiere calibración 3D o mediciones adicionales.</p>'
        f'<p>{aggregate["completed_recordings"]} grabaciones, {aggregate["video_seconds"]:.0f}s de vídeo. '
        f'{aggregate["calibrated_recordings"]} con referencias revisadas; '
        f'{aggregate["unscorable_court_recordings"]} sin proyección evaluable. '
        f'Tiempo total de los procesos: {wall:.1f}s. RAM pico del árbol: '
        f'{aggregate["peak_process_tree_rss_mib"]:.0f} MiB.</p>'
        + quality_text +
        '<h2>Coste y móvil</h2><p>Servicios externos de pago: 0 €. La electricidad y la batería NO se han medido. '
        f'Como ejemplo, a 30 W y 0,25 €/kWh el procesamiento sería '
        f'{manifest["cost"]["illustrative_electricity_only"]["eur"]:.5f} €; no es una factura medida.</p>'
        '<p>No se ha ejecutado en un teléfono ni construido una app nativa. El comprobador estático de ONNX Runtime '
        'detecta fragmentación para la aceleración del modelo de pelota. La interfaz adaptable no implica '
        'que la inferencia se ejecute en el móvil. Faltan pruebas reales de latencia, memoria, batería y temperatura.</p>'
        '<p><a href="report.csv">Tiempos y recursos CSV</a> | <a href="report.json">Informe JSON</a></p>'
        + "".join(cards) + '</html>', encoding="utf-8",
    )


def verify(output: Path) -> None:
    videos = frames = 0
    for source in load_sources():
        directory = output / source["id"]
        result = json.loads((directory / "analysis.json").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        expected = round(settings(source)["duration"] * FPS)
        if result["frames"] != expected or len(result["ball_samples"]) != expected:
            raise ValueError(f"Incorrect per-frame output: {source['id']}")
        for index, sample in enumerate(result["ball_samples"]):
            if sample["frame"] != index or abs(sample["time"] - index / FPS) > 1e-6:
                raise ValueError(f"Misaligned ball timestamp: {source['id']} at {index}")
            point = sample["ball"]
            if point is not None and not (
                0 <= point["x"] < result["width"] and 0 <= point["y"] < result["height"]
            ):
                raise ValueError(f"Out-of-image ball: {source['id']} at {index}")
        if metrics["status"] != "calibrated" and (
            result["summary"]["camera_coverage"] is not None or result["summary"]["players"] is not None
        ):
            raise ValueError(f"Uncalibrated clip claims projection: {source['id']}")
        if not metrics["resources"]["peak_process_tree_rss_mib"] > 0:
            raise ValueError(f"Missing resource measurement: {source['id']}")
        for name in ("original.mp4", "analysis.mp4"):
            capture = cv2.VideoCapture(str(directory / name))
            count = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame.shape[:2] != (result["height"], result["width"]):
                    raise ValueError(f"Incorrect dimensions: {source['id']}/{name}")
                count += 1
            capture.release()
            if count != expected:
                raise ValueError(f"Truncated video: {source['id']}/{name}: {count}/{expected}")
            videos += 1
            frames += count
    save(output / "verification.json", {"recordings": videos // 2, "decoded_videos": videos,
                                      "decoded_frames": frames, "per_frame_contracts": True})
    print(f"Verified {videos} complete MP4s, {frames} decoded frames and per-frame JSON contracts.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible ten-recording CPU tracking benchmark.")
    parser.add_argument("action", choices=("prepare", "run", "report", "verify", "worker"))
    parser.add_argument("--calibration", type=Path, default=Path(__file__).with_name("tracking_cases.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--id")
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--backend", choices=("tracknet", "yolo"), default="tracknet")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.ids)
    elif args.action == "worker":
        worker(args.id, args.calibration, args.output, args.backend)
    elif args.action == "report":
        report(args.output)
    elif args.action == "verify":
        verify(args.output)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for source in load_sources():
            identifier = source["id"]
            if args.ids and identifier not in args.ids:
                continue
            directory = args.output / identifier
            directory.mkdir(exist_ok=True)
            if (directory / "metrics.json").exists():
                completed = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
                if completed.get("ball_backend", "tracknet") != args.backend:
                    raise ValueError(f"{identifier} has another backend; choose a different output directory.")
                if "resources" not in completed:
                    raise ValueError(f"{identifier} has incomplete profiling; choose a fresh output directory.")
                print("Keeping completed", identifier, flush=True)
                continue
            resources = profile([
                sys.executable, str(Path(__file__).resolve()), "worker", "--id", identifier,
                "--calibration", str(args.calibration), "--output", str(args.output),
                "--backend", args.backend,
            ], directory / "worker.log")
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            metrics["resources"] = resources
            save(directory / "metrics.json", metrics)
            report(args.output)
            print(identifier, json.dumps(resources), flush=True)


if __name__ == "__main__":
    main()
