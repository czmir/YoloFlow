from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InferenceRun, ModelVersion, TrainingJob
from app.schemas import BatchInferenceResponse, InferenceResponse, OnnxCompareResponse
from app.services.hardware import detect_device
from app.services.inference_engine import (
    compare_pytorch_onnx,
    export_detections_payload,
    infer_onnx,
    infer_pytorch,
    infer_tiled,
    infer_video,
)

router = APIRouter(prefix="/inference", tags=["inference"])


def _resolve_weights(
    job_id: int | None,
    model_version_id: int | None,
    weights_path: str | None,
    db: Session,
) -> tuple[Path, Path | None]:
    if model_version_id is not None:
        mv = db.get(ModelVersion, model_version_id)
        if not mv:
            raise HTTPException(status_code=404, detail="model version not found")
        return Path(mv.pytorch_path), Path(mv.onnx_path) if mv.onnx_path else None
    if weights_path:
        wp = Path(weights_path)
        if not wp.exists():
            raise HTTPException(status_code=404, detail="weights not found")
        return wp, None
    if job_id is None:
        default = Path("yolov8n.pt")
        if default.exists():
            return default, None
        raise HTTPException(status_code=400, detail="job_id, model_version_id, or weights_path required")
    job = db.get(TrainingJob, job_id)
    if not job or not job.best_weights:
        raise HTTPException(status_code=404, detail="trained weights not found for job")
    onnx = Path(job.onnx_path) if job.onnx_path else None
    return Path(job.best_weights), onnx


@router.post("/detect", response_model=InferenceResponse)
async def detect(
    file: UploadFile = File(...),
    job_id: int | None = Form(default=None),
    model_version_id: int | None = Form(default=None),
    weights_path: str | None = Form(default=None),
    conf: float = Form(default=0.25),
    tiled: bool = Form(default=False),
    tile_size: int = Form(default=640),
    overlap: float = Form(default=0.2),
    backend: str = Form(default="pytorch"),
    db: Session = Depends(get_db),
) -> InferenceResponse:
    weights, onnx_path = _resolve_weights(job_id, model_version_id, weights_path, db)
    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = Path(tmp.name)

    try:
        if tiled:
            use_path = str(onnx_path) if backend == "onnx" and onnx_path else weights
            result = infer_tiled(use_path, image_path, conf=conf, tile_size=tile_size, overlap=overlap, device=device, backend=backend)
        elif backend == "onnx" and onnx_path:
            result = infer_onnx(onnx_path, image_path, conf=conf)
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


@router.post("/detect-batch", response_model=BatchInferenceResponse)
async def detect_batch(
    files: list[UploadFile] = File(...),
    job_id: int | None = Form(default=None),
    model_version_id: int | None = Form(default=None),
    conf: float = Form(default=0.25),
    tiled: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> BatchInferenceResponse:
    weights, onnx_path = _resolve_weights(job_id, model_version_id, None, db)
    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"
    from app.schemas import BatchInferenceItem
    import time

    items: list[BatchInferenceItem] = []
    t0 = time.perf_counter()
    for file in files:
        suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            path = Path(tmp.name)
        try:
            if tiled:
                result = infer_tiled(weights, path, conf=conf, device=device, backend="pytorch")
            else:
                result = infer_pytorch(weights, path, conf=conf, device=device)
            items.append(BatchInferenceItem(image_name=file.filename or "upload", result=result))
        finally:
            path.unlink(missing_ok=True)
    _ = onnx_path
    return BatchInferenceResponse(backend="pytorch", items=items, total_latency_ms=(time.perf_counter() - t0) * 1000)


@router.post("/detect-video", response_model=BatchInferenceResponse)
async def detect_video(
    file: UploadFile = File(...),
    job_id: int | None = Form(default=None),
    model_version_id: int | None = Form(default=None),
    conf: float = Form(default=0.25),
    max_frames: int = Form(default=32),
    db: Session = Depends(get_db),
) -> BatchInferenceResponse:
    weights, _ = _resolve_weights(job_id, model_version_id, None, db)
    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = Path(tmp.name)
    try:
        return infer_video(weights, path, conf=conf, device=device, max_frames=max_frames)
    finally:
        path.unlink(missing_ok=True)


@router.post("/export-detections")
async def export_detections(
    file: UploadFile = File(...),
    job_id: int | None = Form(default=None),
    model_version_id: int | None = Form(default=None),
    conf: float = Form(default=0.25),
    format: str = Form(default="json"),
    tiled: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> Response:
    weights, onnx_path = _resolve_weights(job_id, model_version_id, None, db)
    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = Path(tmp.name)
    try:
        if tiled:
            use = str(onnx_path) if onnx_path else weights
            result = infer_tiled(use, image_path, conf=conf, device=device)
        else:
            result = infer_pytorch(weights, image_path, conf=conf, device=device)
        media, filename, body = export_detections_payload(result.detections, format)
        return Response(content=body, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})
    finally:
        image_path.unlink(missing_ok=True)


@router.post("/compare-onnx", response_model=OnnxCompareResponse)
async def compare_onnx(
    file: UploadFile = File(...),
    job_id: int | None = Form(default=None),
    model_version_id: int | None = Form(default=None),
    weights_path: str | None = Form(default=None),
    conf: float = Form(default=0.25),
    db: Session = Depends(get_db),
) -> OnnxCompareResponse:
    weights, onnx_path = _resolve_weights(job_id, model_version_id, weights_path, db)
    if onnx_path is None or not onnx_path.exists():
        raise HTTPException(status_code=404, detail="onnx export not found")

    hw = detect_device()
    device = hw.device if hw.cuda_available else "cpu"
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        image_path = Path(tmp.name)

    try:
        pt, ox, mean_iou, count_diff = compare_pytorch_onnx(weights, onnx_path, image_path, conf=conf, device=device)
        return OnnxCompareResponse(pytorch=pt, onnx=ox, iou_overlap_mean=mean_iou, count_diff=count_diff)
    finally:
        image_path.unlink(missing_ok=True)