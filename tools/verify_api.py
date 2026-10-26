from __future__ import annotations

import http.client
import json
import struct
import socket
import time
from pathlib import Path

import cv2
import numpy as np
from verification_config import VerificationConfig, parse_arguments

SETTINGS: VerificationConfig
TOKEN = ""


def request(method, path, body=None, authenticated=True, extra=None):
    connection = http.client.HTTPConnection(SETTINGS.host, SETTINGS.port, timeout=90)
    headers = dict(extra or {})
    if authenticated:
        headers["X-Tennis-Token"] = TOKEN
    if isinstance(body, dict):
        body = json.dumps(body)
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        data = json.loads(payload) if "application/json" in response.getheader("Content-Type", "") else payload
        return response.status, data, dict(response.getheaders())
    finally:
        connection.close()


def main():
    global SETTINGS, TOKEN
    SETTINGS = parse_arguments()
    status, configuration, _ = request("GET", "/api/config", authenticated=False)
    if status != 200:
        raise RuntimeError("The TennisLab server is not responsive.")
    TOKEN = configuration["token"]
    QUAD = SETTINGS.corners
    checks = {}
    created = []

    def checked(name, condition):
        if not condition:
            raise AssertionError(name)
        checks[name] = True
        print("OK", name, flush=True)

    checked("health", request("GET", "/api/health")[0] == 200)
    checked("csrf_required", request("POST", "/api/jobs", {}, authenticated=False)[0] == 403)
    checked("foreign_origin_rejected", request("POST", "/api/jobs", {}, extra={"Origin": "https://example.org"})[0] == 403)
    checked("foreign_host_rejected", request("GET", "/api/config", extra={"Host": "example.org"})[0] == 403)
    checked("unknown_clip_404", request("GET", "/api/clips/00000000-0000-0000-0000-000000000000/poster")[0] == 404)
    checked("empty_upload_rejected", request("POST", "/api/clips", b"", extra={"X-File-Name": "empty.mp4"})[0] == 400)
    checked("invalid_container_rejected", request("POST", "/api/clips", b"not-video" * 30, extra={"X-File-Name": "invalid.mp4"})[0] == 400)
    malformed = struct.pack(">I", 24) + b"ftypisom" + b"\0" * 100
    checked("malformed_video_is_actionable_422", request("POST", "/api/clips", malformed, extra={"X-File-Name": "broken.mp4"})[0] == 422)
    checked("wrong_extension_rejected", request("POST", "/api/clips", b"invalid" * 20, extra={"X-File-Name": "script.html"})[0] == 415)
    connection = socket.create_connection((SETTINGS.host, SETTINGS.port), timeout=5)
    connection.sendall(
        (f"POST /api/clips HTTP/1.1\r\nHost: {SETTINGS.host}:{SETTINGS.port}\r\n"
         f"X-Tennis-Token: {TOKEN}\r\nX-File-Name: interrupted-upload-test.mp4\r\n"
         "Content-Type: application/octet-stream\r\nContent-Length: 1048576\r\n\r\n").encode()
        + b"incomplete" * 1000
    )
    connection.close()
    time.sleep(0.3)
    _, current_library, _ = request("GET", "/api/library")
    checked("interrupted_upload_does_not_create_library_item",
            not any(c["name"] == "interrupted-upload-test.mp4" for c in current_library["clips"]))
    status, clip, _ = request(
        "POST", "/api/clips", SETTINGS.video.read_bytes(),
        extra={"X-File-Name": "verification-input" + SETTINGS.video.suffix},
    )
    checked("local_video_uploaded_through_real_ingestion", status == 201 and clip["metadata"]["duration"] >= 3)
    created.append(clip["id"])
    mode = "players" if SETTINGS.players_only else "complete"
    base = {"clip_id": clip["id"], "start": SETTINGS.start, "duration": 2, "mode": mode, "corners": QUAD}
    checked("invalid_corner_order_rejected", request("POST", "/api/jobs", {**base, "corners": [QUAD[0], QUAD[2], QUAD[1], QUAD[3]]})[0] == 422)
    checked("out_of_range_fragment_rejected", request("POST", "/api/jobs", {
        **base, "start": clip["metadata"]["duration"], "duration": 10,
    })[0] == 422)
    status, job, _ = request("POST", "/api/jobs", base)
    checked("real_complete_job_accepted", status == 202)
    checked("active_clip_cannot_be_deleted", request("DELETE", f"/api/clips/{clip['id']}")[0] == 409)
    status, queued, _ = request("POST", "/api/jobs", {**base, "mode": "players"})
    checked("second_job_is_queued", status == 202 and queued["status"] == "queued")
    checked("queued_job_cancel_accepted", request("POST", f"/api/jobs/{queued['id']}/cancel", {})[0] == 200)
    deadline = time.monotonic() + 300
    observations = []
    while time.monotonic() < deadline:
        _, current, _ = request("GET", f"/api/jobs/{job['id']}")
        observations.append(current["progress"])
        if current["status"] not in ("queued", "running", "cancelling"):
            break
        print("Progress", current["status"], round(current["progress"]), current["message"], flush=True)
        time.sleep(3)
    checked("complete_job_finishes", current["status"] in ("completed", "completed_with_warnings"))
    checked("progress_is_measured_and_monotonic", len(set(observations)) > 2 and observations == sorted(observations))
    status, analysis, _ = request("GET", f"/api/jobs/{job['id']}/files/analysis.json")
    minimum_samples = 10 if SETTINGS.players_only else 20
    checked("real_samples_exported", status == 200 and len(analysis["samples"]) >= minimum_samples)
    checked("requested_mode_was_executed", analysis["config"]["mode"] == mode and all("ball" in s for s in analysis["samples"]))
    status, first_bytes, headers = request(
        "GET", f"/api/jobs/{job['id']}/files/analysis.mp4", extra={"Range": "bytes=0-99"}
    )
    checked("video_range_and_seek_supported", status == 206 and len(first_bytes) == 100 and any(key.lower() == "content-range" for key in headers))
    checked("csv_download_available", request("GET", f"/api/jobs/{job['id']}/files/observations.csv")[0] == 200)
    status, archive, _ = request("GET", f"/api/jobs/{job['id']}/files/report.zip")
    checked("zip_download_available", status == 200 and archive.startswith(b"PK"))
    _, cancelled, _ = request("GET", f"/api/jobs/{queued['id']}")
    checked("queued_job_cancelled_without_success_fallback", cancelled["status"] == "cancelled")
    checked("cancelled_results_not_exposed", request("GET", f"/api/jobs/{queued['id']}/files/analysis.mp4")[0] == 409)
    checked("source_files_not_publicly_served", request("GET", "/config.py")[0] == 404)
    private = SETTINGS.output
    private.mkdir(exist_ok=True)
    blank_path = private / "synthetic_blank.mp4"
    writer = cv2.VideoWriter(str(blank_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Cannot create the negative video fixture.")
    for _ in range(36):
        writer.write(np.zeros((360, 640, 3), dtype=np.uint8))
    writer.release()
    status, blank, _ = request(
        "POST", "/api/clips", blank_path.read_bytes(), extra={"X-File-Name": "prueba-negativa-vacio.mp4"}
    )
    checked("valid_container_without_tennis_is_ingested", status == 201)
    created.append(blank["id"])
    negative_config = {**base, "clip_id": blank["id"], "start": 0, "duration": 1, "mode": "players"}
    _, failed_job, _ = request("POST", "/api/jobs", negative_config)
    for _ in range(60):
        _, failed_job, _ = request("GET", f"/api/jobs/{failed_job['id']}")
        if failed_job["status"] not in ("queued", "running", "cancelling"):
            break
        time.sleep(1)
    checked("no_players_is_reported_as_failure", failed_job["status"] == "failed" and bool(failed_job["error"]))
    checked("failed_job_does_not_claim_success", failed_job["progress"] < 100 and failed_job["summary"] is None)
    checked("failed_job_does_not_expose_partial_results", request("GET", f"/api/jobs/{failed_job['id']}/files/analysis.mp4")[0] == 409)
    _, running_job, _ = request("POST", "/api/jobs", {**base, "mode": "players", "duration": 3})
    for _ in range(30):
        _, running_job, _ = request("GET", f"/api/jobs/{running_job['id']}")
        if running_job["status"] == "running":
            break
        time.sleep(0.1)
    checked("running_cancel_request_accepted", request("POST", f"/api/jobs/{running_job['id']}/cancel", {})[0] == 200)
    for _ in range(60):
        _, running_job, _ = request("GET", f"/api/jobs/{running_job['id']}")
        if running_job["status"] not in ("queued", "running", "cancelling"):
            break
        time.sleep(0.2)
    checked("running_job_cancels_without_results", running_job["status"] == "cancelled" and running_job["summary"] is None)
    checked("owned_negative_fixture_can_be_deleted", request("DELETE", f"/api/clips/{blank['id']}")[0] == 200)
    blank_path.unlink()
    report = {
        "checks": checks, "job_id": job["id"], "clip_id": clip["id"],
        "summary": analysis["summary"],
        "created_clip_ids": created,
        "notice": "A local user-provided clip tests application flow, not ground-truth model accuracy.",
    }
    target = SETTINGS.output
    target.mkdir(exist_ok=True)
    (target / "api_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": len(checks), "summary": analysis["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
