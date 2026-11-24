from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import MetricPoint, TrainingJob
from app.schemas import MetricPointOut

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/jobs/{job_id}/chart")
def experiment_chart(job_id: int, db: Session = Depends(get_db)) -> dict:
    """Minimal experiment history payload for frontend charts (stub-ready)."""
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    points = (
        db.query(MetricPoint)
        .filter(MetricPoint.job_id == job_id)
        .order_by(MetricPoint.epoch)
        .all()
    )
    return {
        "job_id": job_id,
        "status": job.status,
        "series": {
            "epochs": [p.epoch for p in points],
            "train_loss": [p.train_loss for p in points],
            "val_map50": [p.val_map50 for p in points],
            "val_map50_95": [p.val_map50_95 for p in points],
        },
        "metrics": job.metrics_json or {},
        "points": [MetricPointOut.model_validate(p).model_dump() for p in points],
    }