from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from config import DATA, DATABASE


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    database = sqlite3.connect(DATABASE, timeout=15)
    database.row_factory = sqlite3.Row
    try:
        yield database
        database.commit()
    finally:
        database.close()


def initialize() -> None:
    with connection() as database:
        database.execute("PRAGMA journal_mode=WAL")
        database.executescript(
            "CREATE TABLE IF NOT EXISTS clips (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "metadata TEXT NOT NULL, created TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, clip_id TEXT NOT NULL, "
            "config TEXT NOT NULL, status TEXT NOT NULL, progress REAL NOT NULL, "
            "message TEXT NOT NULL, error TEXT, summary TEXT, created TEXT NOT NULL, updated TEXT NOT NULL);"
        )
        database.execute(
            "UPDATE jobs SET status='interrupted', message='El servidor se cerró durante el análisis. "
            "Puedes volver a ejecutarlo.', updated=? WHERE status IN ('queued','running','cancelling')",
            (now(),),
        )


def resource_dir(identity: str) -> Path:
    if str(uuid.UUID(identity)) != identity:
        raise ValueError("Invalid resource identifier.")
    directory = DATA / "clips" / identity
    if not directory.resolve().is_relative_to((DATA / "clips").resolve()):
        raise ValueError("Resource path is outside local storage.")
    return directory


def job_dir(clip_id: str, job_id: str) -> Path:
    if str(uuid.UUID(job_id)) != job_id:
        raise ValueError("Invalid job identifier.")
    return resource_dir(clip_id) / "jobs" / job_id


def create_clip(identity: str, name: str, metadata: dict) -> dict:
    with connection() as database:
        database.execute(
            "INSERT INTO clips VALUES (?,?,?,?)",
            (identity, name, json.dumps(metadata, allow_nan=False), now()),
        )
    return get_clip(identity)


def get_clip(identity: str) -> dict | None:
    with connection() as database:
        row = database.execute("SELECT * FROM clips WHERE id=?", (identity,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["metadata"] = json.loads(result["metadata"])
    return result


def list_clips() -> list[dict]:
    with connection() as database:
        rows = database.execute("SELECT * FROM clips ORDER BY created DESC").fetchall()
    return [{**dict(row), "metadata": json.loads(row["metadata"])} for row in rows]


def create_job(clip_id: str, config: dict) -> dict:
    identity = str(uuid.uuid4())
    timestamp = now()
    with connection() as database:
        database.execute(
            "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
            (identity, clip_id, json.dumps(config), "queued", 0, "En cola", None, None, timestamp, timestamp),
        )
    return get_job(identity)


def get_job(identity: str) -> dict | None:
    with connection() as database:
        row = database.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["config"] = json.loads(result["config"])
    result["summary"] = json.loads(result["summary"]) if result["summary"] else None
    return result


def list_jobs() -> list[dict]:
    with connection() as database:
        rows = database.execute(
            "SELECT jobs.*,clips.name AS clip_name FROM jobs JOIN clips ON clips.id=jobs.clip_id "
            "ORDER BY jobs.created DESC"
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["config"] = json.loads(item["config"])
        item["summary"] = json.loads(item["summary"]) if item["summary"] else None
        result.append(item)
    return result


def update_job(identity: str, **values) -> None:
    allowed = {"status", "progress", "message", "error", "summary"}
    if not values or not set(values).issubset(allowed):
        raise ValueError("Invalid job update.")
    if "summary" in values:
        values["summary"] = json.dumps(values["summary"], allow_nan=False)
    values["updated"] = now()
    assignments = ", ".join(f"{key}=?" for key in values)
    with connection() as database:
        database.execute(
            f"UPDATE jobs SET {assignments} WHERE id=?", (*values.values(), identity)
        )


def remove_clip(identity: str) -> None:
    with connection() as database:
        database.execute("DELETE FROM jobs WHERE clip_id=?", (identity,))
        database.execute("DELETE FROM clips WHERE id=?", (identity,))
