#!/usr/bin/env python3
"""YF-1 acceptance: import COCO128, train on CPU, export ONNX, compare inference."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT)

from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Dataset, TrainingJob  # noqa: E402
from app.services.dataset_validator import import_dataset_from_yaml  # noqa: E402
from app.services.inference_engine import compare_pytorch_onnx  # noqa: E402
from app.services.ultralytics_trainer import run_training_job  # noqa: E402
from ultralytics.utils.downloads import safe_download  # noqa: E402


def main() -> int:
    init_db()
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
        text = template.read_text(encoding="utf-8").replace("path: coco128", f"path: {assets}")
        yaml_path.write_text(text, encoding="utf-8")

    name, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name="coco128")
    if not validation.valid:
        print("dataset invalid", validation.errors, file=sys.stderr)
        return 1

    report: dict = {
        "dataset": stats.model_dump(),
        "validation": validation.model_dump(),
        "timings": {},
    }

    with SessionLocal() as db:
        ds = db.query(Dataset).filter(Dataset.name == name).first()
        if not ds:
            ds = Dataset(
                name=name,
                yaml_path=str(resolved),
                root_dir=str(resolved.parent),
                stats_json=stats.model_dump(),
                validation_json=validation.model_dump(),
            )
            db.add(ds)
            db.commit()
            db.refresh(ds)

        job = TrainingJob(
            name="acceptance-coco128",
            dataset_id=ds.id,
            status="pending",
            device="cpu",
            epochs=1,
            batch_size=4,
            imgsz=320,
            base_weights="yolov8n.pt",
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id

    t0 = time.perf_counter()
    run_training_job(job_id)
    report["timings"]["train_seconds"] = round(time.perf_counter() - t0, 2)

    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if not job or job.status != "succeeded":
            print("training failed", job.error_message if job else "missing", file=sys.stderr)
            return 2
        weights = Path(job.best_weights or "")
        onnx = Path(job.onnx_path or "")
        sample = next((assets / "images" / "train2017").glob("*.jpg"), None)
        if sample and weights.exists() and onnx.exists():
            t1 = time.perf_counter()
            pt, ox, mean_iou, count_diff = compare_pytorch_onnx(weights, onnx, sample, device="cpu")
            report["timings"]["compare_seconds"] = round(time.perf_counter() - t1, 2)
            report["onnx_compare"] = {
                "pytorch_detections": len(pt.detections),
                "onnx_detections": len(ox.detections),
                "mean_iou": mean_iou,
                "count_diff": count_diff,
                "sample_image": str(sample),
            }
        report["job"] = {
            "id": job.id,
            "status": job.status,
            "metrics": job.metrics_json,
            "weights": job.best_weights,
            "onnx": job.onnx_path,
        }

    out = ROOT / "data" / "acceptance-report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())