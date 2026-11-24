#!/usr/bin/env python3
"""Release acceptance: real dataset + model, pause/resume, tiled vs full, ORT, optional E2E."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from app.config import settings  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Dataset, JobStatus, TrainingJob  # noqa: E402
from app.services.dataset_validator import import_dataset_from_yaml  # noqa: E402
from app.services.hardware import detect_device  # noqa: E402
from app.services.inference_engine import (  # noqa: E402
    compare_pytorch_onnx,
    export_detections_payload,
    infer_pytorch,
    infer_tiled,
    infer_video,
)
from app.services.job_state import apply_pause, apply_unpause, recover_orphan_jobs  # noqa: E402
from app.services.ultralytics_trainer import run_training_job  # noqa: E402
from ultralytics.utils.downloads import safe_download  # noqa: E402

FIXED_IMAGE_NAME = "000000000009.jpg"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_coco128() -> Path:
    assets = ROOT / "data" / "datasets" / "coco128"
    zip_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    if not (assets / "images" / "train2017").exists():
        if assets.exists():
            shutil.rmtree(assets)
        assets.parent.mkdir(parents=True, exist_ok=True)
        zip_path = safe_download(url=zip_url, dir=assets.parent)
        if zip_path and Path(zip_path).suffix == ".zip":
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(assets.parent)
            nested = assets.parent / "coco128"
            if nested.exists() and not (assets / "images").exists():
                if assets.exists():
                    shutil.rmtree(assets)
                nested.rename(assets)
    yaml_path = assets / "data.yaml"
    if not yaml_path.exists():
        import ultralytics

        template = Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml"
        yaml_path.write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    text = yaml_path.read_text(encoding="utf-8")
    text = text.replace("path: ../datasets/coco128", f"path: {assets}")
    text = text.replace("path: coco128", f"path: {assets}")
    yaml_path.write_text(text, encoding="utf-8")
    return assets


def versions() -> dict:
    import fastapi
    import numpy
    import onnxruntime
    import torch
    import ultralytics

    hw = detect_device()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": bool(hw.cuda_available),
        "device": hw.device,
        "ultralytics": ultralytics.__version__,
        "onnxruntime": onnxruntime.__version__,
        "fastapi": fastapi.__version__,
        "numpy": numpy.__version__,
        "note": "This VM is Linux. Windows start is scripts/setup.ps1 + scripts/start-dev.ps1.",
    }


def persist_dataset(name: str, yaml_path: Path, stats, validation) -> int:
    with SessionLocal() as db:
        ds = db.query(Dataset).filter(Dataset.name == name).first()
        if not ds:
            ds = Dataset(
                name=name,
                yaml_path=str(yaml_path),
                root_dir=str(yaml_path.parent),
                stats_json=stats.model_dump(),
                validation_json=validation.model_dump(),
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)
        return ds.id


def create_job(name: str, dataset_id: int, epochs: int, **kwargs) -> int:
    with SessionLocal() as db:
        job = TrainingJob(
            name=name,
            dataset_id=dataset_id,
            status=JobStatus.pending.value,
            device="cpu",
            epochs=epochs,
            batch_size=kwargs.get("batch_size", 4),
            imgsz=kwargs.get("imgsz", 320),
            base_weights=kwargs.get("base_weights", "yolov8n.pt"),
            resume_from=kwargs.get("resume_from"),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def job_row(job_id: int) -> TrainingJob | None:
    with SessionLocal() as db:
        return db.get(TrainingJob, job_id)


def main() -> int:
    init_db()
    report: dict = {
        "started_at": _now(),
        "hardware_versions": versions(),
        "smoke_vs_formal": (
            "COCO128 1-epoch is a smoke / plumbing check, not a generalization or SOTA eval. "
            "Do not treat mAP here as proof the model works on unseen data."
        ),
        "steps": {},
    }

    assets = ensure_coco128()
    yaml_path = assets / "data.yaml"
    t0 = time.perf_counter()
    name, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name="coco128")
    report["steps"]["import_validate_seconds"] = round(time.perf_counter() - t0, 2)
    report["dataset"] = {
        "name": name,
        "public": "Ultralytics COCO128",
        "images_train": stats.num_images_train,
        "images_val": stats.num_images_val,
        "labels_train": stats.num_labels_train,
        "valid": validation.valid,
        "warnings": validation.warnings[:8],
    }
    if not validation.valid:
        print("dataset invalid", validation.errors, file=sys.stderr)
        return 1
    ds_id = persist_dataset(name, resolved, stats, validation)

    job_id = create_job("release-coco128-1ep", ds_id, epochs=1)
    t0 = time.perf_counter()
    run_training_job(job_id)
    report["steps"]["train_1epoch_seconds"] = round(time.perf_counter() - t0, 2)
    job = job_row(job_id)
    if not job or job.status != JobStatus.succeeded.value:
        print("train failed", job.error_message if job else "missing", file=sys.stderr)
        return 2
    weights = Path(job.best_weights or "")
    onnx = Path(job.onnx_path or "")
    report["train"] = {
        "job_id": job.id,
        "status": job.status,
        "params": {"epochs": 1, "batch": 4, "imgsz": 320, "device": "cpu", "base_weights": "yolov8n.pt"},
        "metrics": job.metrics_json,
        "weights": job.best_weights,
        "onnx": job.onnx_path,
        "registered": True,
    }
    if not weights.exists() or not onnx.exists():
        report["onnx_compare"] = {"skipped": True, "reason": "missing weights or onnx"}
        (ROOT / "data" / "acceptance-release.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 3

    images = sorted((assets / "images" / "train2017").glob("*.jpg"))
    fixed = next((p for p in images if p.name == FIXED_IMAGE_NAME), images[0])
    extra = [p for p in images if p != fixed][:3]
    t0 = time.perf_counter()
    pt, ox, mean_iou, count_diff = compare_pytorch_onnx(weights, onnx, fixed, device="cpu")
    report["steps"]["pt_vs_onnx_seconds"] = round(time.perf_counter() - t0, 2)
    report["pt_vs_ort"] = {
        "image": str(fixed),
        "pytorch": {
            "count": len(pt.detections),
            "latency_ms": pt.latency_ms,
            "boxes": [d.model_dump() for d in pt.detections[:8]],
        },
        "onnx": {
            "count": len(ox.detections),
            "latency_ms": ox.latency_ms,
            "boxes": [d.model_dump() for d in ox.detections[:8]],
        },
        "mean_iou": mean_iou,
        "count_diff": count_diff,
    }

    import cv2
    import numpy as np

    large_path = ROOT / "data" / "runs" / "large_tile.jpg"
    large_path.parent.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(fixed))
    if src is None:
        return 4
    tiled_src = np.tile(src, (3, 3, 1))
    cv2.imwrite(str(large_path), tiled_src)
    t0 = time.perf_counter()
    full = infer_pytorch(weights, large_path, device="cpu", imgsz=320)
    tiled = infer_tiled(weights, large_path, tile_size=320, overlap=0.2, device="cpu")
    report["steps"]["large_image_seconds"] = round(time.perf_counter() - t0, 2)
    report["large_image"] = {
        "path": str(large_path),
        "width": int(tiled_src.shape[1]),
        "height": int(tiled_src.shape[0]),
        "full_count": len(full.detections),
        "full_latency_ms": full.latency_ms,
        "tiled_count": len(tiled.detections),
        "tiled_latency_ms": tiled.latency_ms,
        "tile_count": tiled.tile_count,
    }

    batch_items = []
    t0 = time.perf_counter()
    for img in [fixed, *extra[:2]]:
        r = infer_pytorch(weights, img, device="cpu")
        batch_items.append({"image": img.name, "count": len(r.detections), "latency_ms": r.latency_ms})
    report["steps"]["batch_infer_seconds"] = round(time.perf_counter() - t0, 2)
    report["batch_infer"] = batch_items
    media, filename, body = export_detections_payload(pt.detections, "json")
    export_path = ROOT / "data" / "runs" / filename
    export_path.write_text(body, encoding="utf-8")
    report["export"] = {"format": media, "path": str(export_path), "bytes": export_path.stat().st_size}

    video_path = ROOT / "data" / "runs" / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 2.0, (src.shape[1], src.shape[0]))
    for img_path in [fixed, *extra[:3]]:
        fr = cv2.imread(str(img_path))
        if fr is not None:
            if fr.shape[1] != src.shape[1] or fr.shape[0] != src.shape[0]:
                fr = cv2.resize(fr, (src.shape[1], src.shape[0]))
            vw.write(fr)
    vw.release()
    t0 = time.perf_counter()
    video = infer_video(weights, video_path, device="cpu", max_frames=4)
    report["steps"]["video_infer_seconds"] = round(time.perf_counter() - t0, 2)
    report["video_infer"] = {
        "path": str(video_path),
        "frames": len(video.items),
        "total_latency_ms": video.total_latency_ms,
    }

    pause_job_id = create_job("release-pause-2ep", ds_id, epochs=2)
    err: list[str] = []

    def _pause() -> None:
        deadline = time.time() + 90
        while time.time() < deadline:
            row = job_row(pause_job_id)
            if row and row.status == JobStatus.running.value:
                time.sleep(2.0)
                with SessionLocal() as db:
                    j = db.get(TrainingJob, pause_job_id)
                    if j:
                        apply_pause(db, j)
                return
            time.sleep(0.5)
        err.append("never saw running")

    t = threading.Thread(target=_pause, daemon=True)
    t.start()
    t0 = time.perf_counter()
    run_training_job(pause_job_id)
    t.join(timeout=5)
    paused = job_row(pause_job_id)
    report["pause"] = {
        "job_id": pause_job_id,
        "status_after_pause": paused.status if paused else None,
        "error": err or (paused.error_message if paused else None),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    if not paused or paused.status != JobStatus.paused.value:
        report["pause"]["ok"] = False
    else:
        with SessionLocal() as db:
            j = db.get(TrainingJob, pause_job_id)
            apply_unpause(db, j)
        t0 = time.perf_counter()
        run_training_job(pause_job_id)
        resumed = job_row(pause_job_id)
        report["pause"]["ok"] = bool(resumed and resumed.status == JobStatus.succeeded.value)
        report["pause"]["status_after_resume"] = resumed.status if resumed else None
        report["pause"]["resume_seconds"] = round(time.perf_counter() - t0, 2)

    crash_id = create_job("release-crash-lease", ds_id, epochs=1)
    with SessionLocal() as db:
        j = db.get(TrainingJob, crash_id)
        j.status = JobStatus.running.value
        j.claimed_by = "dead:1"
        j.heartbeat_at = None
        j.workspace_dir = str(settings.workspaces_dir / f"job_{crash_id}")
        db.commit()
        recovered = recover_orphan_jobs(db)
    report["crash_recovery"] = {"job_id": crash_id, "recovered": crash_id in recovered}

    ort = {"started": False}
    ort_proc = None
    try:
        env = os.environ.copy()
        env["YOLOFLOW_ORT_MODEL"] = str(onnx)
        env["PORT"] = "8001"
        env["PYTHONPATH"] = str(ROOT / "backend") + os.pathsep + env.get("PYTHONPATH", "")
        ort_proc = subprocess.Popen(
            [sys.executable, str(ROOT / "deploy" / "ort_server" / "main.py")],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        import httpx

        for _ in range(40):
            try:
                h = httpx.get("http://127.0.0.1:8001/health", timeout=1.0)
                if h.status_code == 200:
                    ort["health"] = h.json()
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.25)
        with fixed.open("rb") as fh:
            det = httpx.post(
                "http://127.0.0.1:8001/v1/detect",
                files={"file": (fixed.name, fh, "image/jpeg")},
                data={"conf": "0.25"},
                timeout=60.0,
            )
        ort["started"] = True
        ort["status_code"] = det.status_code
        try:
            body_json = det.json()
        except Exception:  # noqa: BLE001
            body_json = {"raw": det.text[:500]}
        ort["count"] = body_json.get("count")
        ort["sample"] = (body_json.get("detections") or [])[:3]
        if det.status_code != 200:
            ort["error"] = body_json
    except Exception as exc:  # noqa: BLE001
        ort["error"] = str(exc)
    finally:
        if ort_proc is not None:
            ort_proc.terminate()
            try:
                ort_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ort_proc.kill()
    report["ort_service"] = ort

    report["finished_at"] = _now()
    out = ROOT / "data" / "acceptance-release.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report.get("onnx_compare", {}).get("skipped"):
        return 3
    if not report.get("pause", {}).get("ok"):
        print("pause/resume did not succeed", file=sys.stderr)
        return 5
    if not ort.get("started") or ort.get("status_code") != 200:
        print("ORT service call failed", file=sys.stderr)
        return 6
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
