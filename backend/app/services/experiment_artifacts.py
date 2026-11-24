from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from app.models import TrainingJob


def _train_dir(job: TrainingJob) -> Path | None:
    if not job.workspace_dir:
        return None
    d = Path(job.workspace_dir) / "train"
    return d if d.exists() else None


def collect_post_train_artifacts(job: TrainingJob) -> dict:
    """Scan Ultralytics output dir for paths usable by experiment UI."""
    extra: dict = {}
    td = _train_dir(job)
    if td is None:
        return extra

    cm_png = td / "confusion_matrix.png"
    cm_norm = td / "confusion_matrix_normalized.png"
    results_csv = td / "results.csv"
    if cm_png.exists():
        extra["confusion_matrix_image"] = str(cm_png)
    if cm_norm.exists():
        extra["confusion_matrix_normalized_image"] = str(cm_norm)

    cm_npy = td / "confusion_matrix.npy"
    if cm_npy.exists():
        try:
            extra["confusion_matrix"] = np.load(cm_npy).tolist()
        except Exception:  # noqa: BLE001
            pass

    if results_csv.exists():
        with results_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if rows:
                extra["results_last_row"] = rows[-1]

    return extra


def read_confusion_matrix(job: TrainingJob) -> dict:
    td = _train_dir(job)
    if td is None:
        return {"available": False}
    cm_npy = td / "confusion_matrix.npy"
    if cm_npy.exists():
        try:
            matrix = np.load(cm_npy)
            return {"available": True, "matrix": matrix.tolist(), "shape": list(matrix.shape)}
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": str(exc)}
    cm_png = td / "confusion_matrix.png"
    if cm_png.exists():
        return {"available": True, "image_path": str(cm_png), "matrix": None}
    return {"available": False}