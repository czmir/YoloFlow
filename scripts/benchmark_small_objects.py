"""Compare full-image and sliced inference on small COCO128 bounding boxes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import onnxruntime
import torch
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.infer_compare import match_detections_to_gt, parse_yolo_label_file
from app.services.inference_engine import infer_onnx, infer_pytorch, infer_tiled

SIZE = 320
CONFIDENCE = 0.25
OVERLAP = 0.2


def benchmark(dataset: Path, weights: Path, limit: int) -> dict:
    torch.set_num_threads(2)
    onnx = Path(YOLO(str(weights)).export(format="onnx", imgsz=SIZE, opset=12, simplify=False))
    modes = {
        "pytorch_full": lambda path: infer_pytorch(weights, path, conf=CONFIDENCE, imgsz=SIZE),
        "pytorch_tiled": lambda path: infer_tiled(
            weights, path, conf=CONFIDENCE, tile_size=SIZE, overlap=OVERLAP,
        ),
        "onnx_full": lambda path: infer_onnx(onnx, path, conf=CONFIDENCE, imgsz=SIZE),
        "onnx_tiled": lambda path: infer_tiled(
            onnx, path, conf=CONFIDENCE, tile_size=SIZE, overlap=OVERLAP, backend="onnx",
        ),
    }
    rows = []
    for image in sorted((dataset / "images/train2017").glob("*.jpg")):
        label = dataset / "labels/train2017" / f"{image.stem}.txt"
        if not label.is_file():
            continue
        with Image.open(image) as decoded:
            width, height = decoded.size
        if max(width, height) <= SIZE:
            continue
        boxes, classes = parse_yolo_label_file(label.read_text(encoding="utf-8"), width, height)
        small = [
            (box, cls)
            for box, cls in zip(boxes, classes, strict=True)
            if (box[2] - box[0]) * (box[3] - box[1]) < 32 ** 2
        ]
        if not small:
            continue
        small_boxes, small_classes = zip(*small, strict=True)
        row = {"image": image.name, "width": width, "height": height, "small_boxes": len(small), "modes": {}}
        for name, infer in modes.items():
            before = time.perf_counter()
            result = infer(image)
            elapsed_ms = (time.perf_counter() - before) * 1000
            predictions = sorted(result.detections, key=lambda box: box.confidence, reverse=True)
            match = match_detections_to_gt(
                [[box.x1, box.y1, box.x2, box.y2] for box in predictions],
                [box.class_id for box in predictions],
                small_boxes, small_classes,
            )
            row["modes"][name] = {
                "matched_small_boxes": match["tp"],
                "detections": len(predictions),
                "tile_count": result.tile_count,
                "wall_ms": round(elapsed_ms, 3),
            }
        rows.append(row)
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise ValueError(f"Expected {limit} qualifying images, found {len(rows)}")
    total = sum(row["small_boxes"] for row in rows)
    summary = {}
    for name in modes:
        measurements = [row["modes"][name] for row in rows]
        matched = sum(row["matched_small_boxes"] for row in measurements)
        summary[name] = {
            "matched": matched,
            "small_box_recall": matched / total,
            "median_wall_ms": round(statistics.median(row["wall_ms"] for row in measurements), 3),
        }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "selection": "Sorted COCO128 image IDs with side > 320 and at least one box below 1024 px²",
        "metric": "Class-aware one-to-one match, IoU > 0.5; area uses bounding box, not segmentation",
        "timing": "Single call per mode and image, includes model/session creation and postprocessing",
        "imgsz": SIZE,
        "confidence": CONFIDENCE,
        "overlap": OVERLAP,
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": {"torch": torch.__version__, "ort": onnxruntime.__version__, "ultralytics": ultralytics.__version__},
        "checkpoint": weights.name,
        "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "small_boxes": total,
        "summary": summary,
        "images": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    options = parser.parse_args()
    if options.limit < 1:
        parser.error("--limit must be positive")
    result = benchmark(options.dataset, options.weights, options.limit)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
