#!/usr/bin/env python3
"""Official COCO 2017 train vs val subset (disjoint image ids), 80-class YOLO layout."""

from __future__ import annotations

import random
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import ijson
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "datasets" / "coco2017-mini"
CACHE = ROOT / "data" / "datasets" / "_cache"
ANN_CANDIDATES = [
    CACHE / "annotations_trainval2017.zip",
    Path("/tmp/yoloflow-dl/ann-http.zip"),
    Path("/tmp/yoloflow-dl/ann-insecure.zip"),
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


def _find_ann() -> Path:
    for p in ANN_CANDIDATES:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    raise FileNotFoundError("COCO annotations_trainval2017.zip not found")


def _download(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 1000
    except Exception:
        return False


def _stream_split(zf: zipfile.ZipFile, member: str, n_images: int, seed: int, image_cap: int) -> list:
    rng = random.Random(seed)
    with zf.open(member) as fh:
        images = []
        for img in ijson.items(fh, "images.item"):
            images.append(img)
            if len(images) >= image_cap:
                break
    rng.shuffle(images)
    chosen = images[: n_images * 5]
    want = {int(im["id"]) for im in chosen}
    anns_by: dict[int, list] = defaultdict(list)
    with zf.open(member) as fh:
        for ann in ijson.items(fh, "annotations.item"):
            iid = int(ann["image_id"])
            if iid in want and "bbox" in ann:
                anns_by[iid].append(ann)
    picked = []
    for im in chosen:
        iid = int(im["id"])
        if anns_by[iid]:
            picked.append((im, anns_by[iid]))
        if len(picked) >= n_images:
            break
    return picked


def _write_yaml(path: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO80))
    path.write_text(
        f"path: {path.parent}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )


def prepare(n_train: int = 40, n_val: int = 20, seed: int = 42, force: bool = False) -> dict:
    ann_zip = _find_ann()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / "annotations_trainval2017.zip"
    if ann_zip != cached:
        shutil.copy2(ann_zip, cached)
        ann_zip = cached

    marker = OUT / "data.yaml"
    if marker.exists() and not force:
        nt = len(list((OUT / "images" / "train").glob("*.jpg")))
        nv = len(list((OUT / "images" / "val").glob("*.jpg")))
        if nt >= n_train and nv >= n_val:
            return {
                "dataset": "COCO 2017 official train2017 vs val2017 subset",
                "yaml": str(marker),
                "train_images": nt,
                "val_images": nv,
                "reused": True,
                "overlap": 0,
            }

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ann_zip) as zf:
        with zf.open("annotations/instances_val2017.json") as fh:
            cats = list(ijson.items(fh, "categories.item"))
        cat_id_to_idx = {int(c["id"]): i for i, c in enumerate(sorted(cats, key=lambda x: int(x["id"])))}
        train_p = _stream_split(zf, "annotations/instances_train2017.json", n_train, seed, 8000)
        val_p = _stream_split(zf, "annotations/instances_val2017.json", n_val, seed + 1, 5000)

    train_ids = {int(im["id"]) for im, _ in train_p}
    val_ids = {int(im["id"]) for im, _ in val_p}
    if train_ids & val_ids:
        raise RuntimeError("COCO subset train/val overlap")

    def materialize(picked, split: str, url_split: str) -> list[int]:
        ok_ids = []
        for im, anns in picked:
            iid = int(im["id"])
            file_name = f"{iid:012d}.jpg"
            dest = OUT / "images" / split / file_name
            url = f"http://images.cocodataset.org/{url_split}/{file_name}"
            if not _download(url, dest):
                continue
            w = int(im["width"])
            h = int(im["height"])
            lines = []
            for ann in anns:
                idx = cat_id_to_idx.get(int(ann["category_id"]))
                if idx is None:
                    continue
                x, y, bw, bh = ann["bbox"]
                if bw <= 1 or bh <= 1 or w <= 0 or h <= 0:
                    continue
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                lines.append(f"{idx} {xc:.6f} {yc:.6f} {bw / w:.6f} {bh / h:.6f}")
            if not lines:
                dest.unlink(missing_ok=True)
                continue
            (OUT / "labels" / split / f"{iid:012d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            ok_ids.append(iid)
        return ok_ids

    train_ok = materialize(train_p, "train", "train2017")
    val_ok = materialize(val_p, "val", "val2017")
    if len(train_ok) < max(8, n_train // 2) or len(val_ok) < max(4, n_val // 2):
        raise RuntimeError(f"too few images downloaded train={len(train_ok)} val={len(val_ok)}")
    _write_yaml(OUT / "data.yaml")
    return {
        "dataset": "COCO 2017 official splits (train2017 vs val2017 subset)",
        "source_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "images_base": "http://images.cocodataset.org/{train2017,val2017}/",
        "yaml": str(OUT / "data.yaml"),
        "train_images": len(train_ok),
        "val_images": len(val_ok),
        "train_ids_sample": train_ok[:8],
        "val_ids_sample": val_ok[:8],
        "overlap": 0,
        "classes": COCO80,
        "seed": seed,
        "reused": False,
        "note": "80 COCO names so yolov8n.pt head is not re-initialized",
    }


if __name__ == "__main__":
    import json

    print(json.dumps(prepare(), indent=2))
