from __future__ import annotations

import hashlib
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import ModelVersion, OnnxExportRecord, TrainingJob


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register_job_model(db: Session, job: TrainingJob) -> ModelVersion | None:
    if not job.best_weights:
        return None
    pt = Path(job.best_weights)
    if not pt.exists():
        return None

    existing = db.query(ModelVersion).filter(ModelVersion.job_id == job.id).first()
    version_label = f"job-{job.id}-v1"
    onnx_path = job.onnx_path

    row = existing or ModelVersion(
        job_id=job.id,
        version=version_label,
        pytorch_path=str(pt),
        onnx_path=onnx_path,
        pytorch_sha256=_sha256(pt),
        onnx_sha256=_sha256(Path(onnx_path)) if onnx_path else None,
        metrics_json=job.metrics_json,
    )
    if existing:
        row.pytorch_path = str(pt)
        row.onnx_path = onnx_path
        row.pytorch_sha256 = _sha256(pt)
        row.onnx_sha256 = _sha256(Path(onnx_path)) if onnx_path else None
        row.metrics_json = job.metrics_json
    else:
        db.add(row)
    db.commit()
    db.refresh(row)

    if onnx_path and Path(onnx_path).exists():
        _record_onnx_export(db, row.id, Path(onnx_path))
    return row


def _record_onnx_export(db: Session, model_version_id: int, onnx_path: Path, opset: int = 12) -> OnnxExportRecord:
    rec = OnnxExportRecord(
        model_version_id=model_version_id,
        onnx_path=str(onnx_path),
        opset=opset,
        size_bytes=onnx_path.stat().st_size,
        sha256=_sha256(onnx_path),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def export_onnx_version(db: Session, model_version_id: int, weights_path: Path, out_path: Path) -> OnnxExportRecord:
    from app.services.onnx_export import export_to_onnx

    export_to_onnx(weights_path, out_path)
    return _record_onnx_export(db, model_version_id, out_path)