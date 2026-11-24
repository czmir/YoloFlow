from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dataset, MetricPoint, TrainingJob
from app.schemas import (
    ConfusionMatrixResponse,
    ExperimentCompareRequest,
    ExperimentCompareResponse,
    ExperimentSeries,
    ExperimentSummary,
    MetricPointOut,
)
from app.services.experiment_artifacts import read_confusion_matrix

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _metric(metrics: dict | None, key: str) -> float | None:
    if not metrics:
        return None
    v = metrics.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _summary(job: TrainingJob) -> ExperimentSummary:
    m = job.metrics_json or {}
    return ExperimentSummary(
        job_id=job.id,
        name=job.name,
        status=job.status,
        dataset_id=job.dataset_id,
        parent_job_id=job.parent_job_id,
        epochs=job.epochs,
        elapsed_seconds=_metric(m, "elapsed_seconds"),
        map50=_metric(m, "metrics/mAP50(B)"),
        map50_95=_metric(m, "metrics/mAP50-95(B)"),
        created_at=job.created_at,
    )


def _series(job: TrainingJob, points: list[MetricPoint]) -> ExperimentSeries:
    return ExperimentSeries(
        job_id=job.id,
        name=job.name,
        epochs=[p.epoch for p in points],
        train_loss=[p.train_loss for p in points],
        val_map50=[p.val_map50 for p in points],
        val_map50_95=[p.val_map50_95 for p in points],
    )


@router.get("", response_model=list[ExperimentSummary])
def list_experiments(db: Session = Depends(get_db)) -> list[ExperimentSummary]:
    jobs = db.query(TrainingJob).order_by(TrainingJob.created_at.desc()).all()
    return [_summary(j) for j in jobs]


@router.get("/jobs/{job_id}/chart")
def experiment_chart(job_id: int, db: Session = Depends(get_db)) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    points = (
        db.query(MetricPoint).filter(MetricPoint.job_id == job_id).order_by(MetricPoint.epoch).all()
    )
    return {
        "job_id": job_id,
        "status": job.status,
        "series": {
            "epochs": [p.epoch for p in points],
            "train_loss": [p.train_loss for p in points],
            "val_map50": [p.val_map50 for p in points],
            "val_map50_95": [p.val_map50_95 for p in points],
            "precision": [p.precision for p in points],
            "recall": [p.recall for p in points],
        },
        "metrics": job.metrics_json or {},
        "points": [MetricPointOut.model_validate(p).model_dump() for p in points],
    }


@router.post("/compare", response_model=ExperimentCompareResponse)
def compare_experiments(payload: ExperimentCompareRequest, db: Session = Depends(get_db)) -> ExperimentCompareResponse:
    jobs: list[TrainingJob] = []
    for jid in payload.job_ids:
        job = db.get(TrainingJob, jid)
        if not job:
            raise HTTPException(status_code=404, detail=f"job {jid} not found")
        jobs.append(job)

    summaries = [_summary(j) for j in jobs]
    series_list: list[ExperimentSeries] = []
    table: list[dict] = []
    for job in jobs:
        points = (
            db.query(MetricPoint).filter(MetricPoint.job_id == job.id).order_by(MetricPoint.epoch).all()
        )
        series_list.append(_series(job, points))
        ds = db.get(Dataset, job.dataset_id)
        table.append(
            {
                "job_id": job.id,
                "name": job.name,
                "dataset": ds.name if ds else None,
                "status": job.status,
                "epochs": job.epochs,
                "map50": _metric(job.metrics_json, "metrics/mAP50(B)"),
                "map50_95": _metric(job.metrics_json, "metrics/mAP50-95(B)"),
                "elapsed_seconds": _metric(job.metrics_json, "elapsed_seconds"),
            }
        )

    return ExperimentCompareResponse(jobs=summaries, series=series_list, metric_table=table)


@router.get("/jobs/{job_id}/confusion-matrix", response_model=ConfusionMatrixResponse)
def confusion_matrix(job_id: int, db: Session = Depends(get_db)) -> ConfusionMatrixResponse:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    raw = read_confusion_matrix(job)
    ds = db.get(Dataset, job.dataset_id)
    names: list[str] = []
    if ds and ds.stats_json:
        names = list(ds.stats_json.get("class_names") or [])
    if raw.get("available") and raw.get("image_path"):
        return ConfusionMatrixResponse(
            available=True,
            matrix=raw.get("matrix"),
            image_url=f"/api/v1/experiments/jobs/{job_id}/confusion-matrix/image",
            class_names=names,
        )
    return ConfusionMatrixResponse(
        available=bool(raw.get("available")),
        matrix=raw.get("matrix"),
        class_names=names,
    )


@router.get("/jobs/{job_id}/confusion-matrix/image")
def confusion_matrix_image(job_id: int, db: Session = Depends(get_db)) -> FileResponse:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    raw = read_confusion_matrix(job)
    path = raw.get("image_path")
    if not path or not Path(path).exists():
        norm = (Path(job.workspace_dir or "") / "train" / "confusion_matrix_normalized.png")
        if norm.exists():
            path = str(norm)
        else:
            raise HTTPException(status_code=404, detail="confusion matrix image not found")
    return FileResponse(path)