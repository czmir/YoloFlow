from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ultralytics import YOLO

from app.database import get_db
from app.models import Dataset, JobStatus, TrainingJob
from app.schemas import (
    CheckpointInfo,
    CheckpointVerifyResult,
    HardwareInfo,
    MetricPointOut,
    QueueJobItem,
    TrainingJobCreate,
    TrainingJobOut,
    TrainingQueueOut,
)
from app.services.hardware import detect_device
from app.services.training_queue import get_queue

router = APIRouter(prefix="/training", tags=["training"])


def _job_out(job: TrainingJob) -> TrainingJobOut:
    return TrainingJobOut.model_validate(job)


@router.get("/hardware", response_model=HardwareInfo)
def hardware() -> HardwareInfo:
    return detect_device()


@router.get("/queue", response_model=TrainingQueueOut)
def queue_status(db: Session = Depends(get_db)) -> TrainingQueueOut:
    q = get_queue()
    active_id = q.active_job_id
    running = db.get(TrainingJob, active_id) if active_id else None
    pending_rows = (
        db.query(TrainingJob)
        .filter(TrainingJob.status == JobStatus.pending.value)
        .order_by(TrainingJob.created_at)
        .all()
    )
    paused_rows = (
        db.query(TrainingJob).filter(TrainingJob.status == JobStatus.paused.value).order_by(TrainingJob.created_at).all()
    )
    pending = [
        QueueJobItem(id=j.id, name=j.name, status=j.status, position=i + 1, created_at=j.created_at)
        for i, j in enumerate(pending_rows)
    ]
    paused = [QueueJobItem(id=j.id, name=j.name, status=j.status, position=None, created_at=j.created_at) for j in paused_rows]
    return TrainingQueueOut(
        active_job_id=active_id,
        running=_job_out(running) if running else None,
        pending=pending,
        paused=paused,
    )


@router.get("/jobs", response_model=list[TrainingJobOut])
def list_jobs(db: Session = Depends(get_db)) -> list[TrainingJob]:
    return db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()


@router.get("/jobs/{job_id}", response_model=TrainingJobOut)
def get_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/jobs/{job_id}/metrics", response_model=list[MetricPointOut])
def job_metrics(job_id: int, db: Session = Depends(get_db)):
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return sorted(job.metric_points, key=lambda m: m.epoch)


@router.get("/jobs/{job_id}/checkpoints", response_model=list[CheckpointInfo])
def list_checkpoints(job_id: int, db: Session = Depends(get_db)) -> list[CheckpointInfo]:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    root = Path(job.workspace_dir or "") / "train" / "weights"
    if not root.exists():
        return []
    out: list[CheckpointInfo] = []
    for p in sorted(root.glob("*.pt")):
        out.append(CheckpointInfo(path=str(p), kind=p.stem, size_bytes=p.stat().st_size))
    return out


@router.post("/jobs/{job_id}/verify-checkpoint", response_model=CheckpointVerifyResult)
def verify_checkpoint(job_id: int, path: str | None = None, db: Session = Depends(get_db)) -> CheckpointVerifyResult:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    ckpt = Path(path) if path else Path(job.best_weights or "")
    if not ckpt.exists():
        return CheckpointVerifyResult(path=str(ckpt), loadable=False, message="checkpoint file not found")
    try:
        YOLO(str(ckpt))
        return CheckpointVerifyResult(path=str(ckpt), loadable=True, message="ok")
    except Exception as exc:  # noqa: BLE001
        return CheckpointVerifyResult(path=str(ckpt), loadable=False, message=str(exc))


@router.post("/jobs", response_model=TrainingJobOut, status_code=201)
def create_job(payload: TrainingJobCreate, db: Session = Depends(get_db)) -> TrainingJob:
    ds = db.get(Dataset, payload.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    validation = ds.validation_json or {}
    if not validation.get("valid", True):
        raise HTTPException(status_code=422, detail="dataset failed validation")

    hw = detect_device()
    job = TrainingJob(
        name=payload.name,
        dataset_id=payload.dataset_id,
        status=JobStatus.pending.value,
        device=hw.device if hw.cuda_available else "cpu",
        epochs=payload.epochs,
        batch_size=payload.batch_size,
        imgsz=payload.imgsz,
        base_weights=payload.base_weights,
        resume_from=payload.resume_from,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    queue = get_queue()
    if queue.active_job_id is None:
        queue.enqueue(job.id)
    return job


@router.post("/jobs/{job_id}/cancel", response_model=TrainingJobOut)
def cancel_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == JobStatus.pending.value:
        job.status = JobStatus.cancelled.value
        job.finished_at = job.finished_at or job.created_at
    elif job.status == JobStatus.running.value:
        job.cancel_requested = True
    elif job.status == JobStatus.paused.value:
        job.status = JobStatus.cancelled.value
    else:
        raise HTTPException(status_code=409, detail=f"cannot cancel job in status {job.status}")
    db.commit()
    db.refresh(job)
    return job


@router.post("/jobs/{job_id}/pause", response_model=TrainingJobOut)
def pause_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == JobStatus.running.value:
        job.pause_requested = True
    elif job.status == JobStatus.pending.value:
        job.status = JobStatus.paused.value
    else:
        raise HTTPException(status_code=409, detail=f"cannot pause job in status {job.status}")
    db.commit()
    db.refresh(job)
    return job


@router.post("/jobs/{job_id}/unpause", response_model=TrainingJobOut)
def unpause_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != JobStatus.paused.value:
        raise HTTPException(status_code=409, detail="job is not paused")
    job.status = JobStatus.pending.value
    db.commit()
    db.refresh(job)
    get_queue().kick()
    return job


@router.post("/jobs/{job_id}/resume", response_model=TrainingJobOut, status_code=201)
def resume_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    parent = db.get(TrainingJob, job_id)
    if not parent:
        raise HTTPException(status_code=404, detail="job not found")
    if not parent.best_weights:
        raise HTTPException(status_code=422, detail="no checkpoint available")

    verify = verify_checkpoint(job_id, parent.best_weights, db)
    if not verify.loadable:
        raise HTTPException(status_code=422, detail=verify.message)

    hw = detect_device()
    child = TrainingJob(
        name=f"{parent.name} (resume)",
        dataset_id=parent.dataset_id,
        status=JobStatus.pending.value,
        device=hw.device if hw.cuda_available else "cpu",
        epochs=parent.epochs,
        batch_size=parent.batch_size,
        imgsz=parent.imgsz,
        base_weights=parent.best_weights,
        resume_from=parent.best_weights,
    )
    db.add(child)
    db.commit()
    db.refresh(child)
    if get_queue().active_job_id is None:
        get_queue().enqueue(child.id)
    else:
        get_queue().kick()
    return child