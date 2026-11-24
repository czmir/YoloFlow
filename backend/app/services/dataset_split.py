from __future__ import annotations

import random
import shutil
from pathlib import Path

import yaml

from app.schemas import SplitConfig
from app.services.dataset_validator import validate_yolo_dataset, _parse_class_names


def apply_split(
    yaml_path: Path,
    config: SplitConfig,
) -> Path:
    yaml_path = yaml_path.resolve()
    with yaml_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    root = Path(cfg.get("path", yaml_path.parent)).resolve()
    class_names = _parse_class_names(cfg)

    train_img = root / config.train_rel
    val_img = root / config.val_rel
    train_lbl = root / "labels" / train_img.name
    val_lbl = root / "labels" / val_img.name
    train_img.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)
    val_lbl.mkdir(parents=True, exist_ok=True)

    all_images = sorted(
        [p for p in train_img.rglob("*") if p.is_file()]
        + [p for p in val_img.rglob("*") if p.is_file()]
    )
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [p for p in all_images if p.suffix.lower() in exts]
    # dedupe by stem
    by_stem: dict[str, Path] = {}
    for p in all_images:
        by_stem[p.stem] = p
    all_images = list(by_stem.values())

    if config.mode == "manual" and config.manual_val_files:
        val_set = {Path(f).name for f in config.manual_val_files}
        val_files = [p for p in all_images if p.name in val_set]
        train_files = [p for p in all_images if p.name not in val_set]
    else:
        rng = random.Random(config.seed)
        shuffled = all_images[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * config.val_ratio))
        val_files = shuffled[:n_val]
        train_files = shuffled[n_val:]

    def move_pair(img: Path, dest_img_dir: Path, dest_lbl_dir: Path) -> None:
        split = img.parent.name
        lbl = root / "labels" / split / f"{img.stem}.txt"
        if not lbl.exists():
            for candidate in root.rglob(f"labels/**/{img.stem}.txt"):
                lbl = candidate
                break
        target_img = dest_img_dir / img.name
        if img.resolve() != target_img.resolve():
            shutil.move(str(img), str(target_img))
        if lbl.exists():
            target_lbl = dest_lbl_dir / f"{img.stem}.txt"
            if lbl.resolve() != target_lbl.resolve():
                shutil.move(str(lbl), str(target_lbl))

    for p in train_files:
        move_pair(p, train_img, train_lbl)
    for p in val_files:
        move_pair(p, val_img, val_lbl)

    cfg["path"] = str(root)
    cfg["train"] = str(Path(config.train_rel).as_posix())
    cfg["val"] = str(Path(config.val_rel).as_posix())
    if class_names and "names" not in cfg:
        cfg["names"] = class_names
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return yaml_path