#!/usr/bin/env python3
"""Build a tiny YOLO zip (coco8: 4 train / 4 val, disjoint) for UI upload E2E."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "datasets" / "_cache"
OUT_DIR = ROOT / "data" / "runs"
OUT_ZIP = OUT_DIR / "e2e-coco8.zip"

COCO8_CANDIDATES = [
    CACHE / "coco8.zip",
    Path("/tmp/yoloflow-dl/coco8.zip"),
]

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _find_zip() -> Path:
    for p in COCO8_CANDIDATES:
        if p.exists() and p.stat().st_size > 1000:
            return p
    raise FileNotFoundError(
        "coco8.zip not found. Download "
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip"
    )


def prepare() -> Path:
    src = _find_zip()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "coco8.zip"
    if src != cached:
        shutil.copy2(src, cached)
        src = cached
    extract = CACHE / "coco8"
    if extract.exists():
        shutil.rmtree(extract)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(CACHE)
    root = extract if (extract / "images").exists() else CACHE / "coco8"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO80))
    (root / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(root))
    return OUT_ZIP


if __name__ == "__main__":
    p = prepare()
    print(p, p.stat().st_size)
