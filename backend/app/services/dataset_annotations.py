from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from app.schemas import AnnotationBox, SampleAnnotation
from app.services.dataset_validator import _images_in_split, _label_path_for_image, _parse_class_names, _resolve_path


def list_sample_images(yaml_path: Path, split: str = "train", limit: int = 24) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path", yaml_path.parent)).resolve()
    rel = cfg.get("train" if split == "train" else "val", f"images/{split}")
    images = _images_in_split(root, str(rel))[:limit]
    return [{"file_name": p.name, "split": split, "path": str(p)} for p in images]


def load_annotations(yaml_path: Path, file_name: str, split: str) -> SampleAnnotation:
    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path", yaml_path.parent)).resolve()
    class_names = _parse_class_names(cfg)
    rel = cfg.get("train" if split == "train" else "val", f"images/{split}")
    img_dir = _resolve_path(root, str(rel))
    img_path = img_dir / file_name
    if not img_path.exists():
        raise FileNotFoundError(file_name)

    with Image.open(img_path) as im:
        w, h = im.size

    label_file = _label_path_for_image(img_path)
    boxes: list[AnnotationBox] = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            x2 = (xc + bw / 2) * w
            y2 = (yc + bh / 2) * h
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            boxes.append(
                AnnotationBox(
                    class_id=cls_id,
                    class_name=name,
                    x_center=xc,
                    y_center=yc,
                    width=bw,
                    height=bh,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return SampleAnnotation(
        dataset_id=0,
        split=split,
        file_name=file_name,
        image_width=w,
        image_height=h,
        boxes=boxes,
    )