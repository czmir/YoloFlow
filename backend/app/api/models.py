from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ModelVersion, OnnxExportRecord, TrainingJob
from app.schemas import BatchCompareItem, BatchOnnxCompareResponse, ModelVersionOut, OnnxExportRecordOut
from app.services.hardware import detect_device
from app.services.inference_engine import compare_pytorch_onnx
from app.services.model_registry import export_onnx_version, register_job_model

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db)) -> list[ModelVersion]:
    return db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()


@router.get("/{model_id}", response_model=ModelVersionOut)
def get_model(model_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    row = db.get(ModelVersion, model_id)
    if not row:
        raise HTTPException(status_code=404, detail="model version not found")
    return row


@router.get("/job/{job_id}", response_model=ModelVersionOut)
def get_model_by_job(job_id: int, db: Session = Depends(get_db)) -> ModelVersion:
    row = db.query(ModelVersion).filter(ModelVersion.job_id == job_id).first()
    if not row:
        job = db.get(TrainingJob, job_id)
        if job and job.status == "succeeded":
            row = register_job_model(db, job)
        if not row:
            raise HTTPException(status_code=404, detail="no model registered for job")
    return row


@router.get("/{model_id}/onnx-exports", response_model=list[OnnxExportRecordOut])
def onnx_history(model_id: int, db: Session = Depends(get_db)) -> list[OnnxExportRecord]:
    if not db.get(ModelVersion, model_id):
        raise HTTPException(status_code=404, detail="model version not found")
    return (
        db.query(OnnxExportRecord)
        .filter(OnnxExportRecord.model_version_id == model_id)
        .order_by(OnnxExportRecord.created_at.desc())
        .all()
    )


@router.post("/{model_id}/re-export-onnx", response_model=OnnxExportRecordOut)
def re_export_onnx(model_id: int, db: Session = Depends(get_db)) -> OnnxExportRecord:
    row = db.get(ModelVersion, model_id)
    if not row:
        raise HTTPException(status_code=404, detail="model version not found")
    out = Path(row.pytorch_path).parent.parent / f"export_{model_id}.onnx"
    return export_onnx_version(db, model_id, Path(row.pytorch_path), out)


@router.post("/{model_id}/compare-batch", response_model=BatchOnnxCompareResponse)
async def compare_batch(
    model_id: int,
    files: list[UploadFile] = File(...),
    conf: float = Form(default=0.25),
    db: Session = Depends(get_db),
) -> BatchOnnxCompareResponse:
    row = db.get(ModelVersion, model_id)
    if not row or not row.onnx_path:
        raise HTTPException(status_code=404, detail="model or onnx not found")
    weights = Path(row.pytorch_path)
    onnx = Path(row.onnx_path)
    if not weights.exists() or not onnx.exists():
        raise HTTPException(status_code=404, detail="artifact files missing on disk")

    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"
    items: list[BatchCompareItem] = []
    ious: list[float] = []

    for file in files:
        suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            path = Path(tmp.name)
        try:
            pt, ox, mean_iou, count_diff = compare_pytorch_onnx(weights, onnx, path, conf=conf, device=device)
            items.append(
                BatchCompareItem(
                    image_name=file.filename or "upload",
                    mean_iou=mean_iou,
                    count_diff=count_diff,
                    pytorch_detections=len(pt.detections),
                    onnx_detections=len(ox.detections),
                )
            )
            ious.append(mean_iou)
        finally:
            path.unlink(missing_ok=True)

    return BatchOnnxCompareResponse(
        model_version_id=model_id,
        items=items,
        mean_iou_avg=float(sum(ious) / len(ious)) if ious else 0.0,
    )