from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, JobStatus, TrainingJob
from app.schemas import HardwareInfo, MetricPointOut, TrainingJobCreate, TrainingJobOut
from app.services.hardware import detect_device
from app.services.training_queue import get_queue

router = APIRouter(prefix="/training", tags=["training"])


@router.get("/hardware", response_model=HardwareInfo)
def hardware() -> HardwareInfo:
    return detect_device()


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


@router.post("/jobs/{job_id}/resume", response_model=TrainingJobOut, status_code=201)
def resume_job(job_id: int, db: Session = Depends(get_db)) -> TrainingJob:
    parent = db.get(TrainingJob, job_id)
    if not parent:
        raise HTTPException(status_code=404, detail="job not found")
    if not parent.best_weights:
        raise HTTPException(status_code=422, detail="no checkpoint available")

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
    get_queue().enqueue(child.id)
    return child