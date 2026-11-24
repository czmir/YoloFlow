from __future__ import annotations

from pathlib import Path

import yaml
from PIL import Image

from app.schemas import DatasetStats, DatasetValidation, ValidationIssue

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _label_path_for_image(img_path: Path) -> Path:
    parts = img_path.parts
    if "images" in parts:
        idx = parts.index("images")
        base = Path(*parts[:idx])
        split = parts[idx + 1]
        return base / "labels" / split / f"{img_path.stem}.txt"
    return img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt"


def _label_path_for_split_label(root: Path, split: str, stem: str) -> Path:
    return root / "labels" / split / f"{stem}.txt"


def _images_in_split(root: Path, rel_dir: str) -> list[Path]:
    img_dir = _resolve_path(root, rel_dir)
    if not img_dir.exists():
        return []
    return [p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def _parse_class_names(cfg: dict) -> list[str]:
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return list(names)
    return []


def _issue(severity: str, code: str, message: str, path: str | None = None) -> ValidationIssue:
    return ValidationIssue(severity=severity, code=code, message=message, path=path)  # type: ignore[arg-type]


def _scan_split(
    root: Path,
    rel_dir: str,
    split: str,
    class_names: list[str],
) -> tuple[int, int, dict[str, int], list[ValidationIssue]]:
    img_dir = _resolve_path(root, rel_dir)
    issues: list[ValidationIssue] = []
    if not img_dir.exists():
        issues.append(_issue("error", "missing_split", f"image directory missing: {img_dir}", str(img_dir)))
        return 0, 0, {}, issues

    images = _images_in_split(root, rel_dir)
    label_count = 0
    class_counts: dict[str, int] = {}
    seen_stems: set[str] = set()
    num_classes = len(class_names)

    for img in images:
        if img.stem in seen_stems:
            issues.append(_issue("warning", "duplicate_image", f"duplicate image stem {img.stem}", str(img)))
        seen_stems.add(img.stem)

        label_file = _label_path_for_image(img)
        if not label_file.exists():
            issues.append(_issue("warning", "missing_label", f"missing label for {img.name}", str(img)))
            continue
        try:
            with Image.open(img) as im:
                im.verify()
            with Image.open(img) as im:
                w, h = im.size
        except Exception as exc:  # noqa: BLE001
            issues.append(_issue("error", "corrupt_image", f"bad image {img.name}: {exc}", str(img)))
            continue

        for line_no, line in enumerate(label_file.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                issues.append(
                    _issue("error", "malformed_label", f"malformed line {line_no} in {label_file.name}", str(label_file))
                )
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
            except ValueError:
                issues.append(_issue("error", "malformed_label", f"non-numeric fields in {label_file}", str(label_file)))
                continue

            if num_classes and cls_id >= num_classes:
                issues.append(
                    _issue(
                        "error",
                        "class_id_out_of_range",
                        f"class id {cls_id} >= nc {num_classes} in {label_file.name}",
                        str(label_file),
                    )
                )
            if num_classes:
                expected_ids = set(range(num_classes))
                # gap check done globally below
            for coord in (xc, yc, bw, bh):
                if coord < 0 or coord > 1:
                    issues.append(
                        _issue("warning", "coord_out_of_range", f"normalized coord out of [0,1] in {label_file}", str(label_file))
                    )
            if bw <= 0 or bh <= 0:
                issues.append(_issue("error", "invalid_bbox", f"zero-size box in {label_file}", str(label_file)))

            key = class_names[cls_id] if num_classes and cls_id < num_classes else str(cls_id)
            class_counts[key] = class_counts.get(key, 0) + 1
            label_count += 1

    # orphan labels
    label_dir = root / "labels" / Path(rel_dir).name if "/" in rel_dir.replace("\\", "/") else root / "labels" / split
    if not label_dir.exists() and "images" in rel_dir:
        split_name = Path(rel_dir).name
        label_dir = root / "labels" / split_name
    if label_dir.exists():
        image_stems = {p.stem for p in images}
        for lf in label_dir.glob("*.txt"):
            if lf.stem not in image_stems:
                issues.append(_issue("warning", "orphan_label", f"label without image: {lf.name}", str(lf)))

    return len(images), label_count, class_counts, issues


def validate_yolo_dataset(yaml_path: Path) -> tuple[DatasetStats, DatasetValidation]:
    if not yaml_path.exists():
        v = DatasetValidation(valid=False, errors=["dataset yaml not found"], issues=[_issue("error", "no_yaml", "dataset yaml not found")])
        return DatasetStats(), v

    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    root = Path(cfg.get("path", yaml_path.parent)).resolve()
    class_names = _parse_class_names(cfg)
    train_rel = str(cfg.get("train", "images/train"))
    val_rel = str(cfg.get("val", "images/val"))

    train_split = Path(train_rel).name if train_rel else "train"
    val_split = Path(val_rel).name if val_rel else "val"

    n_tr, l_tr, cc_tr, i_tr = _scan_split(root, train_rel, train_split, class_names)
    n_va, l_va, cc_va, i_va = _scan_split(root, val_rel, val_split, class_names)
    issues = i_tr + i_va

    if n_tr == 0:
        issues.append(_issue("error", "no_train_images", "no training images found"))
    if n_va == 0:
        issues.append(_issue("warning", "no_val_images", "no validation images found"))

    used_ids: set[int] = set()
    for split_cc in (cc_tr, cc_va):
        for key in split_cc:
            if key.isdigit():
                used_ids.add(int(key))
            elif class_names and key in class_names:
                used_ids.add(class_names.index(key))

    if class_names:
        missing_ids = sorted(set(range(len(class_names))) - used_ids)
        if missing_ids and (l_tr + l_va) > 0:
            issues.append(
                _issue(
                    "warning",
                    "class_id_gap",
                    f"class ids with zero instances: {missing_ids}",
                )
            )

    errors = [x.message for x in issues if x.severity == "error"]
    warnings = [x.message for x in issues if x.severity == "warning"]

    total: dict[str, int] = {}
    for cc in (cc_tr, cc_va):
        for k, v in cc.items():
            total[k] = total.get(k, 0) + v

    stats = DatasetStats(
        num_images_train=n_tr,
        num_images_val=n_va,
        num_labels_train=l_tr,
        num_labels_val=l_va,
        class_names=class_names,
        class_counts_train=cc_tr,
        class_counts_val=cc_va,
        class_counts_total=total,
    )

    validation = DatasetValidation(valid=len(errors) == 0, errors=errors, warnings=warnings, issues=issues)
    return stats, validation


def import_dataset_from_yaml(yaml_path: Path, name: str | None = None) -> tuple[str, Path, DatasetStats, DatasetValidation]:
    yaml_path = yaml_path.resolve()
    stats, validation = validate_yolo_dataset(yaml_path)
    ds_name = name or yaml_path.parent.name
    return ds_name, yaml_path, stats, validation


def discover_yaml_in_tree(root: Path) -> Path | None:
    root = root.resolve()
    candidates = list(root.rglob("data.yaml")) + list(root.rglob("dataset.yaml"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def scan_yolo_folder(folder: Path) -> tuple[Path | None, list[str]]:
    """Scan folder for YOLO layout notes (not persisted)."""
    folder = folder.resolve()
    notes: list[str] = []
    if not folder.exists():
        return None, ["folder does not exist"]
    yaml_path = discover_yaml_in_tree(folder)
    if yaml_path:
        notes.append(f"found {yaml_path.name}")
        return yaml_path, notes
    images = list(folder.rglob("images"))
    labels = list(folder.rglob("labels"))
    if images and labels:
        notes.append("images/ and labels/ present but no data.yaml — create one before import")
    else:
        notes.append("no YOLO layout detected (need data.yaml or images+labels)")
    return None, notes