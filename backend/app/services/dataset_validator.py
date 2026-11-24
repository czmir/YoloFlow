from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from app.schemas import DatasetStats, DatasetValidation


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _count_labels(label_file: Path) -> int:
    if not label_file.exists():
        return 0
    lines = [ln.strip() for ln in label_file.read_text().splitlines() if ln.strip()]
    return len(lines)


def _label_path_for_image(img_path: Path) -> Path:
    parts = img_path.parts
    if "images" in parts:
        idx = parts.index("images")
        base = Path(*parts[:idx])
        split = parts[idx + 1]
        return base / "labels" / split / f"{img_path.stem}.txt"
    return img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt"


def _scan_split(root: Path, rel_dir: str) -> tuple[int, int, dict[str, int], list[str], list[str]]:
    img_dir = _resolve_path(root, rel_dir)
    errors: list[str] = []
    warnings: list[str] = []
    if not img_dir.exists():
        errors.append(f"image directory missing: {img_dir}")
        return 0, 0, {}, errors, warnings

    images = list(img_dir.rglob("*"))
    images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    label_count = 0
    class_counts: dict[str, int] = {}

    for img in images:
        label_file = _label_path_for_image(img)
        if not label_file.exists():
            warnings.append(f"missing label for {img.name}")
            continue
        try:
            Image.open(img).verify()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"bad image {img}: {exc}")
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                errors.append(f"malformed label line in {label_file}")
                continue
            class_counts[parts[0]] = class_counts.get(parts[0], 0) + 1
            label_count += 1

    return len(images), label_count, class_counts, errors, warnings


def validate_yolo_dataset(yaml_path: Path) -> tuple[DatasetStats, DatasetValidation]:
    errors: list[str] = []
    warnings: list[str] = []

    if not yaml_path.exists():
        return DatasetStats(), DatasetValidation(valid=False, errors=["dataset yaml not found"])

    with yaml_path.open() as f:
        cfg = yaml.safe_load(f) or {}

    root = Path(cfg.get("path", yaml_path.parent)).resolve()
    names = cfg.get("names")
    class_names: list[str] = []
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = list(names)

    train_rel = cfg.get("train", "images/train")
    val_rel = cfg.get("val", "images/val")

    n_tr, l_tr, cc_tr, e_tr, w_tr = _scan_split(root, train_rel)
    n_va, l_va, cc_va, e_va, w_va = _scan_split(root, val_rel)
    errors.extend(e_tr)
    errors.extend(e_va)
    warnings.extend(w_tr)
    warnings.extend(w_va)

    if n_tr == 0:
        errors.append("no training images found")
    if n_va == 0:
        warnings.append("no validation images found")

    stats = DatasetStats(
        num_images_train=n_tr,
        num_images_val=n_va,
        num_labels_train=l_tr,
        num_labels_val=l_va,
        class_names=class_names,
        class_counts_train={class_names[int(k)] if k.isdigit() and int(k) < len(class_names) else k: v for k, v in cc_tr.items()},
        class_counts_val={class_names[int(k)] if k.isdigit() and int(k) < len(class_names) else k: v for k, v in cc_va.items()},
    )

    return stats, DatasetValidation(valid=len(errors) == 0, errors=errors, warnings=warnings)


def import_dataset_from_yaml(yaml_path: Path, name: str | None = None) -> tuple[str, Path, DatasetStats, DatasetValidation]:
    yaml_path = yaml_path.resolve()
    stats, validation = validate_yolo_dataset(yaml_path)
    ds_name = name or yaml_path.parent.name
    return ds_name, yaml_path, stats, validation