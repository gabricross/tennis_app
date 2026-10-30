from __future__ import annotations

import asyncio
import json
import logging
import secrets
import shutil
import threading
import time
import urllib.parse
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.requests import ClientDisconnect

import storage
from config import DATA, EXAMPLE_VIDEO, MAX_ANALYSIS_SECONDS, MAX_PENDING_JOBS, MAX_UPLOAD_BYTES, MODELS, ROOT
from court import (
    CALIBRATION_REFERENCES, COURT_LINES, LENGTH, WIDTH, CourtModel,
    calibration_preview, validate_corners,
)
from engine import analyze_clip
from inference import ModelBank
from media import AnalysisCancelled, check_cancel, probe, snapshot

LOGGER = logging.getLogger("tennis.app")
TOKEN = secrets.token_urlsafe(32)
FINISHED = {"completed", "completed_with_warnings"}
ACTIVE = {"queued", "running", "cancelling"}


class CalibrationRequest(BaseModel):
    time: float = Field(ge=0, allow_inf_nan=False)
    corners: list[list[float]]
    calibration_reference: Literal["doubles", "service_boxes", "service_left", "service_right"] = "doubles"

    @model_validator(mode="after")
    def valid_points(self):
        validate_corners(self.corners)
        return self


class AnalysisRequest(BaseModel):
    clip_id: str
    start: float = Field(ge=0, allow_inf_nan=False)
    duration: float = Field(ge=1, le=MAX_ANALYSIS_SECONDS, allow_inf_nan=False)
    mode: str = "players"
    corners: list[list[float]]
    calibration_reference: Literal["doubles", "service_boxes", "service_left", "service_right"] = "doubles"
    ball_backend: Literal["tracknet", "yolo", "yolo_multiscale", "yolo_wasb"] = "tracknet"

    @model_validator(mode="after")
    def valid_configuration(self):
        if self.mode not in ("players", "complete"):
            raise ValueError("Modo de análisis no válido.")
        validate_corners(self.corners)
        return self


class Jobs:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tennis-analysis")
        self.events: dict[str, threading.Event] = {}
        self.futures: dict[str, Future] = {}
        self.lock = threading.RLock()
        self.models = ModelBank()

    def submit(self, configuration: dict) -> dict:
        with self.lock:
            if len(self.events) >= MAX_PENDING_JOBS + 1:
                raise HTTPException(409, "La cola está llena. Espera o cancela un análisis.")
            job = storage.create_job(configuration["clip_id"], configuration)
            event = threading.Event()
            self.events[job["id"]] = event
            future = self.executor.submit(self._run, job, event)
            self.futures[job["id"]] = future
            future.add_done_callback(lambda completed: self._finished(job, completed))
            return job

    def _run(self, job: dict, event: threading.Event) -> None:
        identity, clip_id = job["id"], job["clip_id"]
        directory = storage.job_dir(clip_id, identity)
        last_write = 0.0

        def progress(fraction: float, message: str) -> None:
            nonlocal last_write
            check_cancel(event)
            if time.monotonic() - last_write >= 0.35:
                storage.update_job(identity, progress=min(99, fraction), message=message)
                last_write = time.monotonic()

        try:
            with self.lock:
                check_cancel(event)
                storage.update_job(identity, status="running", progress=1, message="Preparando el análisis")
            clip = storage.get_clip(clip_id)
            if clip is None:
                raise ValueError("El clip ya no está en la biblioteca.")
            summary = analyze_clip(
                storage.resource_dir(clip_id) / "upload", directory, job["config"],
                event, progress, self.models, clip["metadata"].get("source_credit"),
                clip["metadata"].get("input_notice"),
            )
            with self.lock:
                check_cancel(event)
                status = "completed_with_warnings" if summary["warnings"] else "completed"
                storage.update_job(identity, status=status, progress=100, message="Análisis terminado", summary=summary)
        except AnalysisCancelled:
            storage.update_job(identity, status="cancelled", message="Cancelado. No se conservan resultados parciales.")
            if directory.exists():
                shutil.rmtree(directory)
        finally:
            self.models.ball = None

    def _finished(self, job: dict, future: Future) -> None:
        error = None if future.cancelled() else future.exception()
        if error is not None:
            LOGGER.error("Analysis %s failed", job["id"], exc_info=(type(error), error, error.__traceback__))
            storage.update_job(
                job["id"], status="failed", message="No se ha podido completar el análisis",
                error=str(error)[:1400],
            )
            directory = storage.job_dir(job["clip_id"], job["id"])
            if directory.exists():
                shutil.rmtree(directory)
        with self.lock:
            self.events.pop(job["id"], None)
            self.futures.pop(job["id"], None)

    def cancel(self, identity: str) -> dict:
        with self.lock:
            job = require_job(identity)
            if job["status"] not in ACTIVE:
                raise HTTPException(409, "Este análisis ya no está en ejecución.")
            event = self.events.get(identity)
            if event is None:
                raise HTTPException(409, "El análisis no pertenece a esta sesión del servidor.")
            event.set()
            if job["status"] == "queued":
                storage.update_job(identity, status="cancelled", message="Cancelado antes de empezar.")
                future = self.futures.get(identity)
                if future is not None:
                    future.cancel()
            else:
                storage.update_job(identity, status="cancelling", message="Cancelando al terminar la operación actual")
        return storage.get_job(identity)

    def close(self) -> None:
        with self.lock:
            for event in self.events.values():
                event.set()
        self.executor.shutdown(wait=True, cancel_futures=False)


manager: Jobs | None = None
court_model: CourtModel | None = None
court_lock = threading.Lock()


@asynccontextmanager
async def lifespan(application: FastAPI):
    global manager
    storage.initialize()
    manager = Jobs()
    yield
    manager.close()


app = FastAPI(title="Tennis Lab local", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.middleware("http")
async def local_only(request: Request, call_next):
    hostname = request.url.hostname
    if hostname not in ("127.0.0.1", "localhost", "testserver"):
        return JSONResponse({"detail": "Solo se permiten conexiones locales."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin is not None and urllib.parse.urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse({"detail": "Origen no permitido."}, status_code=403)
        if not secrets.compare_digest(request.headers.get("x-tennis-token", ""), TOKEN):
            return JSONResponse({"detail": "Sesión local no válida. Recarga la página."}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ValueError)
async def input_error(request: Request, error: ValueError):
    LOGGER.warning("Invalid local request: %s", error)
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    detail = "; ".join(str(item["msg"]) for item in error.errors()[:3])
    return JSONResponse({"detail": detail}, status_code=422)


def require_clip(identity: str) -> dict:
    try:
        uuid.UUID(identity)
    except ValueError as error:
        raise HTTPException(404, "Clip no encontrado.") from error
    clip = storage.get_clip(identity)
    if clip is None:
        raise HTTPException(404, "Clip no encontrado.")
    return clip


def require_job(identity: str) -> dict:
    job = storage.get_job(identity)
    if job is None:
        raise HTTPException(404, "Análisis no encontrado.")
    return job


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/config")
def configuration():
    return {
        "app": "TennisLab", "version": "1.0", "token": TOKEN,
        "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024,
        "max_analysis_seconds": MAX_ANALYSIS_SECONDS,
        "models": {name: (MODELS / f"{name}.onnx").exists() for name in ("people", "court", "ball", "ball_yolo", "ball_wasb")},
        "storage": str(DATA), "private_local_storage": True,
        "previous_demo_available": EXAMPLE_VIDEO.is_file(),
        "court": {
            "width_m": WIDTH, "length_m": LENGTH, "lines": COURT_LINES,
            "references": {name: points.tolist() for name, points in CALIBRATION_REFERENCES.items()},
        },
    }


@app.get("/api/health")
def health():
    return {"app": "TennisLab", "status": "ok", "worker": "CPU", "paid_services": False}


@app.get("/api/library")
def library():
    return {"clips": storage.list_clips(), "jobs": storage.list_jobs()}


@app.post("/api/clips", status_code=201)
async def upload(request: Request):
    name = request.headers.get("x-file-name")
    if not name:
        raise HTTPException(400, "El archivo necesita un nombre.")
    name = Path(urllib.parse.unquote(name)).name[:180]
    if Path(name).suffix.lower() not in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        raise HTTPException(415, "Usa un vídeo MP4, MOV, WebM, MKV o AVI.")
    declared = request.headers.get("content-length")
    if declared and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "El vídeo supera los 300 MB.")
    identity = str(uuid.uuid4())
    directory = storage.resource_dir(identity)
    directory.mkdir(parents=True)
    committed = False
    try:
        received = 0
        with (directory / "upload").open("wb") as output:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "El vídeo supera los 300 MB.")
                output.write(chunk)
        if received < 64:
            raise HTTPException(400, "El archivo está vacío o incompleto.")
        try:
            metadata = await asyncio.to_thread(probe, directory / "upload")
        except (RuntimeError, OSError, StopIteration) as error:
            LOGGER.warning("Cannot decode upload %s: %s", identity, str(error)[:500])
            raise HTTPException(422, "No se puede decodificar este vídeo. Prueba a exportarlo como MP4 H.264.") from error
        if await request.is_disconnected():
            LOGGER.info("Upload cancelled after video validation: %s", identity)
            raise HTTPException(499, "Carga cancelada por el cliente.")
        clip = storage.create_clip(identity, name, metadata)
        committed = True
        return clip
    except ClientDisconnect as error:
        LOGGER.info("Upload cancelled during streaming: %s", identity)
        raise HTTPException(499, "Carga cancelada por el cliente.") from error
    finally:
        if not committed and directory.exists():
            shutil.rmtree(directory)


@app.post("/api/examples/previous", status_code=201)
async def previous_example():
    source = EXAMPLE_VIDEO
    if not source.exists():
        raise HTTPException(404, "No se encuentra tu demo de tenis anterior.")
    identity = str(uuid.uuid4())
    directory = storage.resource_dir(identity)
    directory.mkdir(parents=True)
    committed = False
    try:
        await asyncio.to_thread(shutil.copyfile, source, directory / "upload")
        metadata = await asyncio.to_thread(probe, directory / "upload")
        metadata["input_notice"] = "Demo anterior con overlays. Sirve para probar el flujo, no como benchmark de precisión."
        clip = storage.create_clip(identity, "Mi demo de tenis anterior.mp4", metadata)
        committed = True
        return clip
    finally:
        if not committed:
            shutil.rmtree(directory)


@app.get("/api/clips/{identity}/poster")
def poster(identity: str):
    require_clip(identity)
    return FileResponse(storage.resource_dir(identity) / "poster.jpg", media_type="image/jpeg")


def preview_path(identity: str, time: float) -> Path:
    clip = require_clip(identity)
    if not 0 <= time <= clip["metadata"]["duration"] - 0.05:
        raise HTTPException(400, "El instante está fuera del vídeo.")
    directory = storage.resource_dir(identity)
    target = directory / f"preview_{round(time * 1000)}.jpg"
    if not target.exists():
        snapshot(directory / "upload", time, target)
    return target


@app.get("/api/clips/{identity}/frame")
def frame(identity: str, time: float = 0):
    return FileResponse(preview_path(identity, time), media_type="image/jpeg")


@app.post("/api/clips/{identity}/calibration")
def review_calibration(identity: str, configuration: CalibrationRequest):
    image = cv2.imread(str(preview_path(identity, configuration.time)))
    if image is None:
        raise HTTPException(422, "No se ha podido leer el fotograma de calibración.")
    try:
        return calibration_preview(image, configuration.corners, configuration.calibration_reference)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/api/clips/{identity}/court")
def propose_court(identity: str, time: float = 0):
    global court_model
    target = preview_path(identity, time)
    if not (MODELS / "court.onnx").exists():
        return {"corners": None, "method": "manual_required", "message": "El modelo de pista no está instalado. Marca las esquinas manualmente."}
    image = cv2.imread(str(target))
    if image is None:
        raise HTTPException(422, "No se ha podido leer el fotograma de calibración.")
    with court_lock:
        if court_model is None:
            court_model = CourtModel()
        return court_model.propose(image)


@app.post("/api/jobs", status_code=202)
def start_job(configuration: AnalysisRequest):
    clip = require_clip(configuration.clip_id)
    if configuration.start + configuration.duration > clip["metadata"]["duration"] + 0.02:
        raise HTTPException(422, "El fragmento termina después del vídeo.")
    if not (MODELS / "people.onnx").exists():
        raise HTTPException(503, "Falta el modelo de jugadores. Ejecuta la preparación de modelos.")
    ball_files = {"tracknet": ["ball.onnx"], "yolo_wasb": ["ball_yolo.onnx", "ball_wasb.onnx"]}.get(
        configuration.ball_backend, ["ball_yolo.onnx"])
    if configuration.mode == "complete" and not all((MODELS / name).exists() for name in ball_files):
        raise HTTPException(503, "Falta el modelo de pelota seleccionado. Elige otro motor o instala sus pesos.")
    if manager is None:
        raise HTTPException(503, "El motor todavía no está preparado.")
    return manager.submit(configuration.model_dump())


@app.get("/api/jobs/{identity}")
def job_status(identity: str):
    return require_job(identity)


@app.get("/api/jobs/{identity}/comparison")
def compare_job(identity: str):
    job = require_job(identity)
    if job["status"] not in FINISHED:
        raise HTTPException(409, "La comparación estará disponible cuando termine el análisis.")
    return FileResponse(ROOT / "web" / "comparison.html", media_type="text/html")


@app.post("/api/jobs/{identity}/cancel")
def cancel_job(identity: str):
    if manager is None:
        raise HTTPException(503, "El motor no está disponible.")
    return manager.cancel(identity)


@app.delete("/api/clips/{identity}")
def delete_clip(identity: str):
    require_clip(identity)
    if any(job["clip_id"] == identity and job["status"] in ACTIVE for job in storage.list_jobs()):
        raise HTTPException(409, "Cancela los análisis de este clip antes de borrarlo.")
    directory = storage.resource_dir(identity)
    if directory.exists():
        shutil.rmtree(directory)
    storage.remove_clip(identity)
    return {"deleted": True}


@app.get("/api/jobs/{identity}/files/{filename}")
def result_file(identity: str, filename: str):
    job = require_job(identity)
    if job["status"] not in FINISHED:
        raise HTTPException(409, "Los resultados todavía no están disponibles.")
    allowed = {"original.mp4", "analysis.mp4", "analysis.json", "observations.csv", "ball.csv", "poster.jpg", "report.zip"}
    if filename not in allowed:
        raise HTTPException(404, "Archivo no encontrado.")
    directory = storage.job_dir(job["clip_id"], identity)
    target = directory / filename
    if filename == "report.zip" and not target.exists():
        temporary = directory / "report.zip.partial"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for name in ("analysis.mp4", "original.mp4", "analysis.json", "observations.csv", "ball.csv"):
                asset = directory / name
                if asset.exists():
                    archive.write(asset, arcname=name)
        temporary.replace(target)
    if not target.is_file():
        raise HTTPException(404, "Este análisis no contiene ese archivo.")
    media_type = {"mp4": "video/mp4", "json": "application/json", "csv": "text/csv", "jpg": "image/jpeg", "zip": "application/zip"}
    download = filename not in ("analysis.json", "poster.jpg", "original.mp4", "analysis.mp4")
    return FileResponse(
        target, media_type=media_type[target.suffix[1:]],
        filename=f"tennis-lab-{identity[:8]}-{filename}" if download else None,
    )


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
