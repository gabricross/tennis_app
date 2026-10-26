from __future__ import annotations

import logging
import math
import subprocess
import threading
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import numpy as np

from config import FPS, MAX_SOURCE_SECONDS

LOGGER = logging.getLogger("tennis.media")


class AnalysisCancelled(Exception):
    pass


def check_cancel(event: threading.Event) -> None:
    if event.is_set():
        raise AnalysisCancelled("Análisis cancelado por el usuario.")


def validate_signature(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(32)
    iso_media = len(header) >= 12 and header[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free")
    matroska = header.startswith(b"\x1a\x45\xdf\xa3")
    avi = header.startswith(b"RIFF") and header[8:12] == b"AVI "
    if not (iso_media or matroska or avi):
        raise ValueError("El archivo no parece un vídeo MP4, MOV, WebM, MKV o AVI válido.")


def probe(path: Path) -> dict:
    validate_signature(path)
    reader = imageio_ffmpeg.read_frames(
        str(path), pix_fmt="rgb24",
        input_params=["-protocol_whitelist", "file,pipe", "-threads", "2", "-max_alloc", "268435456"],
        output_params=["-frames:v", "1"],
    )
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        duration = float(metadata["duration"])
        fps = float(metadata["fps"])
        if not math.isfinite(duration) or not 1 <= duration <= MAX_SOURCE_SECONDS:
            raise ValueError("El vídeo debe durar entre 1 segundo y 30 minutos.")
        if width < 240 or height < 180 or max(width, height) > 4096:
            raise ValueError("Resolución no compatible. Usa un vídeo entre 240 × 180 y 4K.")
        if width * height > 4096 * 2160:
            raise ValueError("La resolución excede el límite de esta aplicación local.")
        pixels = next(reader)
    finally:
        reader.close()
    first = np.frombuffer(pixels, np.uint8).reshape(height, width, 3)
    preview = cv2.cvtColor(first, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path.parent / "poster.jpg"), preview):
        raise RuntimeError("No se ha podido crear la vista previa.")
    return {
        "width": width, "height": height, "duration": duration,
        "fps": fps if fps > 0 and math.isfinite(fps) else None,
        "bytes": path.stat().st_size, "codec": metadata["codec"],
        "normalization_fps": FPS,
    }


def snapshot(source: Path, time: float, destination: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-protocol_whitelist", "file,pipe", "-ss", f"{time:.3f}", "-i", str(source),
        "-frames:v", "1", "-vf", "scale=1280:720:force_original_aspect_ratio=decrease",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, timeout=25)
    if result.returncode or not destination.exists():
        raise ValueError("No se ha podido leer el fotograma seleccionado. Prueba un instante anterior.")


def normalize(
    source: Path, destination: Path, start: float, duration: float,
    event: threading.Event, progress: Callable[[float, str], None],
) -> None:
    stderr = destination.parent / "ffmpeg.log"
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-y", "-protocol_whitelist", "file,pipe", "-ss", str(start), "-i", str(source),
        "-t", str(duration), "-an", "-vf",
        f"scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2,fps={FPS}",
        "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats", str(destination),
    ]
    with stderr.open("wb") as error_stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=error_stream, text=True)
        try:
            if process.stdout is None:
                raise RuntimeError("No se ha podido abrir el progreso de FFmpeg.")
            for line in process.stdout:
                check_cancel(event)
                if line.startswith("out_time_us="):
                    value = line.partition("=")[2].strip()
                    if value.isdigit():
                        fraction = min(1.0, int(value) / 1e6 / duration)
                        progress(2 + 8 * fraction, "Preparando el fragmento a 30 fps")
            return_code = process.wait(timeout=20)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdout:
                process.stdout.close()
    check_cancel(event)
    if return_code != 0:
        detail = stderr.read_text(encoding="utf-8", errors="replace")[-700:]
        raise RuntimeError(f"FFmpeg no pudo preparar el vídeo: {detail}")
