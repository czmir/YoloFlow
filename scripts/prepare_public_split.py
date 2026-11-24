#!/usr/bin/env python3
"""Prepare a small official VOC 2007 train/val subset (disjoint ImageSets)."""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "datasets" / "voc2007-mini"
CACHE = ROOT / "data" / "datasets" / "_cache"

VOC_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
NAME_TO_ID = {n: i for i, n in enumerate(VOC_NAMES)}

ZIP_CANDIDATES = [
    CACHE / "VOCtrainval_06-Nov-2007.zip",
    Path("/tmp/yoloflow-dl/VOC2007_trainval.zip"),
]


def _find_zip() -> Path:
    for p in ZIP_CANDIDATES:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    raise FileNotFoundError(
        "VOC 2007 trainval zip not found. Download "
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip"
    )


def _read_id_list(zf: zipfile.ZipFile, split: str) -> list[str]:
    name = f"VOCdevkit/VOC2007/ImageSets/Main/{split}.txt"
    text = zf.read(name).decode("utf-8")
    ids = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    # Main/train.txt is image ids only (not the per-class +1/-1 files)
    ids = [i.split()[0] for i in ids]
    return ids


def _xml_to_yolo(xml_bytes: bytes) -> tuple[int, int, str]:
    root = ET.fromstring(xml_bytes)
    size = root.find("size")
    if size is None:
        raise ValueError("VOC xml missing size")
    w = int(size.findtext("width") or "0")
    h = int(size.findtext("height") or "0")
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size in xml")
    lines: list[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in NAME_TO_ID:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        xmin = min(max(xmin, 0), w)
        xmax = min(max(xmax, 0), w)
        ymin = min(max(ymin, 0), h)
        ymax = min(max(ymax, 0), h)
        if xmax <= xmin or ymax <= ymin:
            continue
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        lines.append(f"{NAME_TO_ID[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return w, h, "\n".join(lines) + ("\n" if lines else "")


def _write_yaml(path: Path) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(VOC_NAMES))
    path.write_text(
        f"path: {path.parent}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )


def prepare(n_train: int = 48, n_val: int = 24, seed: int = 42, force: bool = False) -> dict:
    zip_path = _find_zip()
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / zip_path.name
    if zip_path != cached:
        shutil.copy2(zip_path, cached)
        zip_path = cached

    marker = OUT / "data.yaml"
    if marker.exists() and not force:
        train_n = len(list((OUT / "images" / "train").glob("*.jpg")))
        val_n = len(list((OUT / "images" / "val").glob("*.jpg")))
        if train_n >= n_train and val_n >= n_val:
            return {
                "dataset": "PASCAL VOC 2007 trainval (official ImageSets)",
                "zip": str(zip_path),
                "out": str(OUT),
                "yaml": str(marker),
                "train_images": train_n,
                "val_images": val_n,
                "reused": True,
            }

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    with zipfile.ZipFile(zip_path) as zf:
        train_ids = _read_id_list(zf, "train")
        val_ids = _read_id_list(zf, "val")
        train_set, val_set = set(train_ids), set(val_ids)
        overlap = train_set & val_set
        if overlap:
            raise RuntimeError(f"VOC train/val overlap ({len(overlap)} ids) — refusing")

        def pick(ids: list[str], n: int, split: str) -> list[str]:
            shuffled = list(ids)
            rng.shuffle(shuffled)
            chosen: list[str] = []
            for image_id in shuffled:
                xml_name = f"VOCdevkit/VOC2007/Annotations/{image_id}.xml"
                jpg_name = f"VOCdevkit/VOC2007/JPEGImages/{image_id}.jpg"
                try:
                    xml_bytes = zf.read(xml_name)
                    jpg_bytes = zf.read(jpg_name)
                except KeyError:
                    continue
                _w, _h, label_text = _xml_to_yolo(xml_bytes)
                if not label_text.strip():
                    continue
                (OUT / "images" / split / f"{image_id}.jpg").write_bytes(jpg_bytes)
                (OUT / "labels" / split / f"{image_id}.txt").write_text(label_text, encoding="utf-8")
                chosen.append(image_id)
                if len(chosen) >= n:
                    break
            if len(chosen) < n:
                raise RuntimeError(f"only found {len(chosen)} labeled {split} images, wanted {n}")
            return chosen

        train_chosen = pick(train_ids, n_train, "train")
        val_chosen = pick(val_ids, n_val, "val")

    _write_yaml(OUT / "data.yaml")
    return {
        "dataset": "PASCAL VOC 2007 trainval (official ImageSets/Main train.txt vs val.txt)",
        "source_url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtrainval_06-Nov-2007.zip",
        "zip": str(zip_path),
        "out": str(OUT),
        "yaml": str(OUT / "data.yaml"),
        "train_images": len(train_chosen),
        "val_images": len(val_chosen),
        "train_ids_sample": train_chosen[:8],
        "val_ids_sample": val_chosen[:8],
        "overlap": 0,
        "classes": VOC_NAMES,
        "seed": seed,
        "reused": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(prepare(), indent=2))
