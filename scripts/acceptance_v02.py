#!/usr/bin/env python3
"""v0.2.0 acceptance: independent-split VOC eval + full vs tiled comparison."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.models import Dataset, JobStatus, TrainingJob  # noqa: E402
from app.services.dataset_validator import import_dataset_from_yaml  # noqa: E402
from app.services.hardware import detect_device  # noqa: E402
from app.services.infer_compare import (  # noqa: E402
    match_detections_to_gt,
    parse_yolo_label_file,
    remap_boxes,
)
from app.services.inference_engine import infer_pytorch, infer_tiled_detailed  # noqa: E402
from app.services.ultralytics_trainer import run_training_job  # noqa: E402
from prepare_coco_subset import prepare as prepare_coco  # noqa: E402
from prepare_public_split import prepare as prepare_voc  # noqa: E402
from sanitize_artifacts import write_json  # noqa: E402

OUT_JSON = ROOT / "data" / "acceptance-v02.json"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
        "windows_scripts": "NOT RUN — this VM is Linux; no self-hosted Windows worker is connected",
    }


def persist_dataset(name: str, yaml_path: Path, stats, validation) -> int:
    with SessionLocal() as db:
        ds = db.query(Dataset).filter(Dataset.name == name).first()
        if ds:
            ds.yaml_path = str(yaml_path)
            ds.root_dir = str(yaml_path.parent)
            ds.stats_json = stats.model_dump()
            ds.validation_json = validation.model_dump()
            db.commit()
            return ds.id
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
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job.id


def job_row(job_id: int) -> TrainingJob | None:
    with SessionLocal() as db:
        return db.get(TrainingJob, job_id)


def _gt_for_image(img_path: Path, split: str) -> tuple[list[list[float]], list[int], tuple[int, int]]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(img_path)
    h, w = img.shape[:2]
    label = img_path.parent.parent.parent / "labels" / split / f"{img_path.stem}.txt"
    if not label.exists():
        # images/{split}/file.jpg -> labels/{split}/file.txt
        label = img_path.parent.parent.parent / "labels" / split / f"{img_path.stem}.txt"
    alt = img_path.parent.parent.parent / "labels" / split / f"{img_path.stem}.txt"
    label_path = alt if alt.exists() else (img_path.parents[2] / "labels" / split / f"{img_path.stem}.txt")
    text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
    boxes, classes = parse_yolo_label_file(text, w, h)
    return boxes, classes, (w, h)


def _label_path(img_path: Path) -> Path:
    # .../images/<split>/<stem>.jpg -> .../labels/<split>/<stem>.txt
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        return Path(*parts[:idx]) / "labels" / parts[idx + 1] / f"{img_path.stem}.txt"
    return img_path.with_suffix(".txt")


def compare_image(
    weights: Path, img_path: Path, tile_size: int = 320, overlap: float = 0.2, conf: float = 0.25
) -> dict:
    gt_boxes, gt_cls, (w, h) = (None, None, (0, 0))  # type: ignore[assignment]
    label_path = _label_path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"cannot read {img_path}")
    h, w = img.shape[:2]
    text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
    gt_boxes, gt_cls = parse_yolo_label_file(text, w, h)

    full = infer_pytorch(weights, img_path, conf=conf, device="cpu", imgsz=tile_size)
    tiled, tiled_stats = infer_tiled_detailed(
        weights, img_path, conf=conf, tile_size=tile_size, overlap=overlap, device="cpu", nms_iou=0.5
    )
    full_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in full.detections]
    full_cls = [d.class_id for d in full.detections]
    tiled_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in tiled.detections]
    tiled_cls = [d.class_id for d in tiled.detections]
    return {
        "image": img_path.name,
        "width": w,
        "height": h,
        "n_gt": len(gt_boxes),
        "full": {
            "count": len(full.detections),
            "latency_ms": round(full.latency_ms, 2),
            "gt_match": match_detections_to_gt(full_boxes, full_cls, gt_boxes, gt_cls),
        },
        "tiled": {
            "count": len(tiled.detections),
            "latency_ms": round(tiled.latency_ms, 2),
            "gt_match": match_detections_to_gt(tiled_boxes, tiled_cls, gt_boxes, gt_cls),
            **tiled_stats,
        },
    }


def mosaic_compare(
    weights: Path, images: list[Path], tile_size: int = 320, overlap: float = 0.2, conf: float = 0.25
) -> dict:
    imgs = []
    gts: list[tuple[list[list[float]], list[int], int, int]] = []
    for p in images[:4]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        text = _label_path(p).read_text(encoding="utf-8") if _label_path(p).exists() else ""
        boxes, cls = parse_yolo_label_file(text, w, h)
        imgs.append(im)
        gts.append((boxes, cls, w, h))
    while len(imgs) < 4:
        imgs.append(imgs[-1] if imgs else np.zeros((320, 320, 3), dtype=np.uint8))
        gts.append(([], [], imgs[-1].shape[1], imgs[-1].shape[0]))
    th = max(imgs[0].shape[0], imgs[1].shape[0])
    tw = max(imgs[0].shape[1], imgs[1].shape[1])
    bh = max(imgs[2].shape[0], imgs[3].shape[0])
    bw = max(imgs[2].shape[1], imgs[3].shape[1])
    canvas_h = th + bh
    canvas_w = max(imgs[0].shape[1] + imgs[1].shape[1], imgs[2].shape[1] + imgs[3].shape[1])
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    origins = [
        (0, 0),
        (imgs[0].shape[1], 0),
        (0, imgs[0].shape[0]),
        (imgs[2].shape[1], imgs[0].shape[0]),
    ]
    # Use actual placements
    origins = [
        (0, 0),
        (imgs[0].shape[1], 0),
        (0, imgs[0].shape[0]),
        (imgs[2].shape[1], imgs[0].shape[0]),
    ]
    canvas_w = max(origins[1][0] + imgs[1].shape[1], origins[3][0] + imgs[3].shape[1])
    canvas_h = max(origins[2][1] + imgs[2].shape[0], origins[3][1] + imgs[3].shape[0])
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    mosaic_boxes: list[list[float]] = []
    mosaic_cls: list[int] = []
    for im, (boxes, cls, _w, _h), (ox, oy) in zip(imgs, gts, origins, strict=False):
        canvas[oy : oy + im.shape[0], ox : ox + im.shape[1]] = im
        mosaic_boxes.extend(remap_boxes(boxes, ox, oy))
        mosaic_cls.extend(cls)
    mosaic_path = ROOT / "data" / "runs" / "eval_mosaic.jpg"
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mosaic_path), canvas)
    full = infer_pytorch(weights, mosaic_path, conf=conf, device="cpu", imgsz=tile_size)
    tiled, tiled_stats = infer_tiled_detailed(
        weights, mosaic_path, conf=conf, tile_size=tile_size, overlap=overlap, device="cpu", nms_iou=0.5
    )
    full_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in full.detections]
    tiled_boxes = [[d.x1, d.y1, d.x2, d.y2] for d in tiled.detections]
    return {
        "path": str(mosaic_path),
        "width": canvas_w,
        "height": canvas_h,
        "source_images": [p.name for p in images[:4]],
        "n_gt": len(mosaic_boxes),
        "full": {
            "count": len(full.detections),
            "latency_ms": round(full.latency_ms, 2),
            "gt_match": match_detections_to_gt(
                full_boxes, [d.class_id for d in full.detections], mosaic_boxes, mosaic_cls
            ),
        },
        "tiled": {
            "count": len(tiled.detections),
            "latency_ms": round(tiled.latency_ms, 2),
            "gt_match": match_detections_to_gt(
                tiled_boxes, [d.class_id for d in tiled.detections], mosaic_boxes, mosaic_cls
            ),
            **tiled_stats,
        },
    }


def _mean(rows: list[dict], path: tuple[str, ...]) -> float | None:
    vals: list[float] = []
    for row in rows:
        cur: object = row
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and isinstance(cur, (int, float)):
            vals.append(float(cur))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def main() -> int:
    init_db()
    report: dict = {
        "started_at": _now(),
        "hardware_versions": versions(),
        "note": (
            "Primary eval: COCO 2017 official train2017 vs val2017 subset (80-class, yolov8n head kept). "
            "Secondary: PASCAL VOC 2007 official train.txt vs val.txt (20-class head re-init). "
            "Not COCO128 overlapping 1-epoch smoke."
        ),
    }
    voc_info = prepare_voc(n_train=48, n_val=24, seed=42)
    coco_info = prepare_coco(n_train=40, n_val=20, seed=42)
    report["dataset_prepare_voc"] = voc_info
    report["dataset_prepare_coco"] = coco_info
    # Primary eval keeps the 80-class COCO head (yolov8n.pt transfer). VOC 20-class
    # fine-tune is recorded too but its detect-conf@0.25 is expected to be weak.
    ds_info = coco_info
    yaml_path = Path(ds_info["yaml"])
    t0 = time.perf_counter()
    name, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name="coco2017-mini")
    report["import_seconds"] = round(time.perf_counter() - t0, 2)
    report["dataset"] = {
        "name": name,
        "public": ds_info["dataset"],
        "source_url": ds_info.get("source_url"),
        "images_train": stats.num_images_train,
        "images_val": stats.num_images_val,
        "labels_train": stats.num_labels_train,
        "labels_val": stats.num_labels_val,
        "class_names": stats.class_names,
        "valid": validation.valid,
        "warnings": validation.warnings[:12],
        "overlap_image_ids": 0,
    }
    if not validation.valid:
        report["error"] = validation.errors
        write_json(OUT_JSON, report)
        print(json.dumps(report, indent=2, default=str))
        return 1
    ds_id = persist_dataset(name, resolved, stats, validation)

    epochs = int(os.environ.get("YOLOFLOW_EVAL_EPOCHS", "10"))
    batch = int(os.environ.get("YOLOFLOW_EVAL_BATCH", "4"))
    imgsz = int(os.environ.get("YOLOFLOW_EVAL_IMGSZ", "320"))
    prior = {}
    if OUT_JSON.exists():
        try:
            prior = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prior = {}
    if prior.get("train") and "voc" in str(prior.get("dataset", {}).get("name", "")).lower():
        report["voc2007_prior"] = {
            "dataset": prior.get("dataset"),
            "train": prior.get("train"),
            "infer_note": (
                "VOC 20-class head was re-initialized; detect@0.25 is often empty while "
                "Ultralytics mAP uses conf≈0.001. See voc2007_prior.train.metrics."
            ),
        }

    job_id = create_job("v02-coco2017-mini", ds_id, epochs=epochs, batch_size=batch, imgsz=imgsz)
    t0 = time.perf_counter()
    run_training_job(job_id)
    train_s = round(time.perf_counter() - t0, 2)
    job = job_row(job_id)
    if not job or job.status != JobStatus.succeeded.value:
        report["train"] = {
            "job_id": job_id,
            "status": job.status if job else None,
            "error": job.error_message if job else "missing",
            "seconds": train_s,
        }
        write_json(OUT_JSON, report)
        print(json.dumps(report, indent=2, default=str))
        return 2
    metrics = job.metrics_json or {}
    report["train"] = {
        "job_id": job.id,
        "status": job.status,
        "seconds": train_s,
        "params": {
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": "cpu",
            "base_weights": "yolov8n.pt",
        },
        "metrics": {
            "mAP50": metrics.get("metrics/mAP50(B)"),
            "mAP50-95": metrics.get("metrics/mAP50-95(B)"),
            "precision": metrics.get("metrics/precision(B)"),
            "recall": metrics.get("metrics/recall(B)"),
            "elapsed_seconds": metrics.get("elapsed_seconds"),
        },
        "raw_keys": sorted(str(k) for k in metrics.keys() if str(k).startswith("metrics/")),
        "weights": job.best_weights,
        "onnx": job.onnx_path,
    }
    weights = Path(job.best_weights or "")
    if not weights.exists():
        write_json(OUT_JSON, report)
        return 3

    val_dir = yaml_path.parent / "images" / "val"
    val_images = sorted(val_dir.glob("*.jpg"))
    per_image = []
    t0 = time.perf_counter()
    for img in val_images:
        per_image.append(compare_image(weights, img, tile_size=imgsz, overlap=0.2))
    report["infer_compare_seconds"] = round(time.perf_counter() - t0, 2)
    report["infer_compare_val"] = {
        "n_images": len(per_image),
        "mean_full_latency_ms": _mean(per_image, ("full", "latency_ms")),
        "mean_tiled_latency_ms": _mean(per_image, ("tiled", "latency_ms")),
        "mean_full_precision": _mean(per_image, ("full", "gt_match", "precision")),
        "mean_full_recall": _mean(per_image, ("full", "gt_match", "recall")),
        "mean_tiled_precision": _mean(per_image, ("tiled", "gt_match", "precision")),
        "mean_tiled_recall": _mean(per_image, ("tiled", "gt_match", "recall")),
        "mean_pre_nms_duplicate_pairs": _mean(per_image, ("tiled", "pre_nms_duplicate_pairs")),
        "mean_post_nms_duplicate_pairs": _mean(per_image, ("tiled", "post_nms_duplicate_pairs")),
        "mean_tiled_suppressed": _mean(per_image, ("tiled", "suppressed")),
        "images": per_image,
    }
    mosaic = mosaic_compare(weights, val_images[:4], tile_size=imgsz, overlap=0.2)
    report["infer_compare_mosaic"] = mosaic
    report["finished_at"] = _now()
    write_json(OUT_JSON, report)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
