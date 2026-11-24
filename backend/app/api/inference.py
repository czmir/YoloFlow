from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import InferenceRun, TrainingJob
from app.schemas import InferenceResponse, OnnxCompareResponse
from app.services.hardware import detect_device
from app.services.inference_engine import (
    compare_pytorch_onnx,
    infer_pytorch,
    infer_tiled_pytorch,
)

router = APIRouter(prefix="/inference", tags=["inference"])


def _resolve_weights(job_id: int | None, weights_path: str | None, db: Session) -> tuple[Path, Path | None]:
    if weights_path:
        wp = Path(weights_path)
        if not wp.exists():
            raise HTTPException(status_code=404, detail="weights not found")
        return wp, None
    if job_id is None and weights_path is None:
        default = Path("yolov8n.pt")
        if default.exists():
            return default, None
        raise HTTPException(status_code=400, detail="job_id or weights_path required")
    job = db.get(TrainingJob, job_id)
    if not job or not job.best_weights:
        raise HTTPException(status_code=404, detail="trained weights not found for job")
    onnx = Path(job.onnx_path) if job.onnx_path else None
    return Path(job.best_weights), onnx


@router.post("/detect", response_model=InferenceResponse)
async def detect(
    file: UploadFile = File(...),
    job_id: int | None = Form(default=None),
    weights_path: str | None = Form(default=None),
    conf: float = Form(default=0.25),
    tiled: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> InferenceResponse:
    weights, _ = _resolve_weights(job_id, weights_path, db)
    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = Path(tmp.name)

    try:
        if tiled:
            result = infer_tiled_pytorch(weights, image_path, conf=conf, device=device)
        else:
            result = infer_pytorch(weights, image_path, conf=conf, device=device)

        db.add(
            InferenceRun(
                job_id=job_id,
                backend=result.backend,
                image_name=file.filename or "upload",
                detections_json=[d.model_dump() for d in result.detections],
                latency_ms=result.latency_ms,
            )
        )
        db.commit()
        return result
    finally:
        image_path.unlink(missing_ok=True)


@router.post("/compare-onnx", response_model=OnnxCompareResponse)
async def compare_onnx(
    file: UploadFile = File(...),
    job_id: int | None = Form(default=None),
    weights_path: str | None = Form(default=None),
    conf: float = Form(default=0.25),
    db: Session = Depends(get_db),
) -> OnnxCompareResponse:
    weights, onnx_path = _resolve_weights(job_id, weights_path, db)
    if onnx_path is None or not onnx_path.exists():
        raise HTTPException(status_code=404, detail="onnx export not found for job")

    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = Path(tmp.name)

    try:
        pt, ox, mean_iou, count_diff = compare_pytorch_onnx(weights, onnx_path, image_path, conf=conf, device=device)
        return OnnxCompareResponse(
            pytorch=pt,
            onnx=ox,
            iou_overlap_mean=mean_iou,
            count_diff=count_diff,
        )
    finally:
        image_path.unlink(missing_ok=True)