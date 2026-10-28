from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS
from court import project
from inference import ModelBank
from kinematics import player_kinematics


def ground_matrix(camera: dict, size: tuple[int, int], floor_y_cm: float = 0) -> tuple[np.ndarray, np.ndarray]:
    """Map studio ground (X,Z), in metres, to undistorted video pixels."""
    width, height = size
    intrinsic = np.array(camera["K"], dtype=np.float64)
    intrinsic[0] *= width / camera["resolution"][0]
    intrinsic[1] *= height / camera["resolution"][1]
    rotation = np.array(camera["R"], dtype=np.float64)
    translation = np.array(camera["t"], dtype=np.float64).reshape(3)
    matrix = intrinsic @ np.column_stack((
        100 * rotation[:, 0], 100 * rotation[:, 2],
        translation + floor_y_cm * rotation[:, 1],
    ))
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-9:
        raise ValueError("The camera cannot project the requested ground plane.")
    return intrinsic, matrix


def load_reference(path: Path, start: int, count: int, body_id: int) -> dict[int, np.ndarray]:
    poses = {}
    with tarfile.open(path) as archive:
        for member in archive:
            match = re.fullmatch(r"hdPose3d_stage1_coco19/body3DScene_(\d{8})\.json", member.name)
            if not member.isfile() or match is None:
                continue
            frame = int(match[1])
            if not start <= frame < start + count:
                continue
            with archive.extractfile(member) as stream:
                data = json.load(stream)
            if data["fpsType"] != "hd_29_97":
                raise ValueError("This benchmark requires native HD 29.97 Hz reference poses.")
            body = next((body for body in data["bodies"] if body["id"] == body_id), None)
            if body is not None:
                joints = np.array(body["joints19"], dtype=np.float64).reshape(19, 4)
                if not np.isfinite(joints).all():
                    raise ValueError(f"Nonfinite reference joints at frame {frame}.")
                if np.min(joints[[2, 8, 14], 3]) >= 0.2:
                    poses[frame] = joints
    if not poses:
        raise ValueError("No reference poses pass the body/ankle confidence threshold.")
    return poses


def error_summary(errors: list[float]) -> dict:
    return {
        "count": len(errors),
        "mae": float(np.mean(errors)) if errors else None,
        "median": float(np.median(errors)) if errors else None,
        "p95": float(np.percentile(errors, 95)) if errors else None,
    }


def compare(samples: list[dict], reference: list[np.ndarray | None]) -> dict:
    summary = player_kinematics(samples)["north"]
    filtered = [
        np.median(reference[i - 2:i + 1], axis=0)
        if i >= 2 and all(point is not None for point in reference[i - 2:i + 1]) else None
        for i in range(len(reference))
    ]
    position_errors = []
    speed_errors = []
    reference_distance = observed_distance = 0.0
    compared_seconds = 0.0
    for i, sample in enumerate(samples):
        if sample["players"] and reference[i] is not None:
            player = sample["players"][0]
            position_errors.append(float(np.linalg.norm(np.array(player["pitch_raw"]) - reference[i])))
        else:
            continue
        motion = player["motion"]
        if (
            i >= 3 and filtered[i] is not None and filtered[i - 1] is not None
            and samples[i - 1]["players"]
            and motion["status"] in ("estimated", "below_resolution")
            and samples[i - 1]["players"][0]["motion"]["status"] in ("estimated", "below_resolution")
            and sample["time"] - samples[i - 1]["time"] <= 0.25
        ):
            reference_distance += float(np.linalg.norm(filtered[i] - filtered[i - 1]))
            previous_distance = samples[i - 1]["players"][0]["motion"]["distance_m"]
            observed_distance += motion["distance_m"] - (previous_distance or 0)
            compared_seconds += sample["time"] - samples[i - 1]["time"]
        if motion["speed_kmh"] is None or filtered[i] is None:
            continue
        # Production's median-of-three is timestamped at the middle observation.
        end_time = samples[i - 1]["time"]
        start_time = end_time - motion["speed_window_seconds"]
        start = next((j for j in range(2, i)
                      if abs(samples[j - 1]["time"] - start_time) < 1e-6), None)
        if start is None or any(point is None for point in filtered[start:i + 1]):
            continue
        path = float(np.linalg.norm(np.diff(filtered[start:i + 1], axis=0), axis=1).sum())
        reference_speed = path / motion["speed_window_seconds"] * 3.6
        speed_errors.append(abs(motion["speed_kmh"] - reference_speed))
    return {
        "production": summary, "position_error_m": error_summary(position_errors),
        "speed_error_kmh": error_summary(speed_errors),
        "matched_distance": {
            "observed_seconds": compared_seconds,
            "production_m": observed_distance if compared_seconds else None,
            "reference_polyline_m": reference_distance if compared_seconds else None,
            "difference_m": observed_distance - reference_distance if compared_seconds else None,
        },
    }


def benchmark(args: argparse.Namespace) -> dict:
    cv2.setNumThreads(2)
    cameras = json.loads(args.calibration.read_text(encoding="utf-8"))["cameras"]
    camera = next((camera for camera in cameras if camera["name"] == args.camera), None)
    if camera is None:
        raise ValueError(f"Camera not present in calibration: {args.camera}")
    poses = load_reference(args.poses, args.start_frame, args.frames, args.body_id)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("Cannot open reference video.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if abs(fps - 30000 / 1001) > 0.001:
        capture.release()
        raise ValueError("Keep the reference video's native 30000/1001 fps; do not normalize it to 30 fps.")
    intrinsic, matrix = ground_matrix(camera, size, args.floor_y_cm)
    inverse = np.linalg.inv(matrix)
    maps = cv2.initUndistortRectifyMap(
        intrinsic, np.array(camera["distCoef"]), None, intrinsic, size, cv2.CV_32FC1,
    )
    detector = ModelBank().person_detector()
    detected, ideal, feet_reference, root_reference = [], [], [], []
    counts = {"no_person": 0, "multiple_people": 0, "reference_missing_or_low_confidence": 0}
    try:
        for index in range(args.frames):
            success, image = capture.read()
            if not success:
                raise RuntimeError(f"Reference video ended at {index}; expected {args.frames} frames.")
            if index % 3:
                continue
            image = cv2.remap(image, *maps, interpolation=cv2.INTER_LINEAR)
            detections = detector.detect(image)
            joints = poses.get(args.start_frame + index)
            feet = joints[[8, 14]][:, [0, 2]].mean(axis=0) / 100 if joints is not None else None
            root = joints[2, [0, 2]] / 100 if joints is not None else None
            feet_reference.append(feet)
            root_reference.append(root)
            sample = {
                "time": index / fps, "frame": args.start_frame + index,
                "camera_valid": True, "camera_continuous": True,
                "matrix": matrix.tolist(), "players": [],
            }
            if len(detections) == 1:
                detection = detections[0]
                position = project([detection.foot], inverse)[0].tolist()
                sample["players"].append({
                    "side": "north", "box": list(detection.box), "pitch_raw": position,
                    "pitch": position, "confidence": detection.confidence,
                })
            else:
                counts["no_person" if not detections else "multiple_people"] += 1
            detected.append(sample)
            reference_sample = {**sample, "players": []}
            if feet is not None:
                x, y = project([feet], matrix)[0]
                reference_sample["players"].append({
                    "side": "north", "box": [x - 10, y - 100, x + 10, y],
                    "pitch_raw": feet.tolist(), "pitch": feet.tolist(),
                })
            else:
                counts["reference_missing_or_low_confidence"] += 1
            ideal.append(reference_sample)
            if index % 150 == 0:
                print("Compared reference frame", index, flush=True)
    finally:
        capture.release()
    return {
        "scope": "Single-person studio reference, not tennis court calibration, player identity or sports accuracy.",
        "reference": "CMU Panoptic triangulated multiview joints19; cm -> m; X,Z ground; Y vertical.",
        "assumptions": {
            "floor_y_cm": args.floor_y_cm,
            "fixed_published_camera": True,
            "person_detector_confidence": detector.confidence,
            "distortion": "Video undistorted with published K and distCoef before detection/projection.",
            "reference_distance": "Median-of-three 10 Hz ankle-centre ground polyline, on matching valid intervals.",
            "speed": "Same time windows and median delay; independent reference polyline, no production deadband.",
            "selection": "Exactly one detected person; multiple-person frames are missing, not matched using truth.",
            "synchronization": "Clip frame 0 must be original HD start_frame; inspect joint reprojections before scoring.",
        },
        "provenance": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (
                ("video", args.video), ("calibration", args.calibration),
                ("poses", args.poses), ("person_model", MODELS / "people.onnx"),
            )
        },
        "camera": args.camera, "body_id": args.body_id, "start_frame": args.start_frame,
        "video_fps": fps, "video_frames": args.frames, "sample_count": len(detected),
        "counts": counts, "reference_fed_filter": compare(ideal, feet_reference),
        "detected_vs_ankle_centre": compare(detected, feet_reference),
        "detected_vs_body_centre": compare(detected, root_reference),
        "samples": detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-only CMU Panoptic metric-reference benchmark.")
    for name in ("video", "calibration", "poses", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--body-id", type=int, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--floor-y-cm", type=float, default=0)
    args = parser.parse_args()
    if args.start_frame < 0 or args.frames < 4 or not np.isfinite(args.floor_y_cm):
        parser.error("Use a nonnegative reference start, at least four frames and a finite floor height.")
    report = benchmark(args)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
