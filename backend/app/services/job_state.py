from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.models import JobStatus, TrainingJob

TERMINAL = {JobStatus.succeeded.value, JobStatus.failed.value, JobStatus.cancelled.value}


def can_cancel(job: TrainingJob) -> bool:
    return job.status in {
        JobStatus.pending.value,
        JobStatus.running.value,
        JobStatus.paused.value,
    }


def can_pause(job: TrainingJob) -> bool:
    return job.status in {JobStatus.pending.value, JobStatus.running.value}


def can_unpause(job: TrainingJob) -> bool:
    return job.status == JobStatus.paused.value


def can_resume_new_job(parent: TrainingJob) -> tuple[bool, str]:
    if not parent.best_weights:
        return False, "no checkpoint available"
    if not Path(parent.best_weights).exists():
        return False, "checkpoint file missing on disk"
    return True, "ok"


def apply_cancel(db: Session, job: TrainingJob) -> TrainingJob:
    if not can_cancel(job):
        raise ValueError(f"cannot cancel job in status {job.status}")
    if job.status == JobStatus.pending.value:
        job.status = JobStatus.cancelled.value
        job.finished_at = datetime.now(timezone.utc)
    elif job.status == JobStatus.paused.value:
        job.status = JobStatus.cancelled.value
        job.finished_at = datetime.now(timezone.utc)
    elif job.status == JobStatus.running.value:
        job.cancel_requested = True
    db.commit()
    db.refresh(job)
    return job


def apply_pause(db: Session, job: TrainingJob) -> TrainingJob:
    if not can_pause(job):
        raise ValueError(f"cannot pause job in status {job.status}")
    if job.status == JobStatus.pending.value:
        job.status = JobStatus.paused.value
    else:
        job.pause_requested = True
    db.commit()
    db.refresh(job)
    return job


def apply_unpause(db: Session, job: TrainingJob) -> TrainingJob:
    if not can_unpause(job):
        raise ValueError(f"cannot unpause job in status {job.status}")
    job.status = JobStatus.pending.value
    job.pause_requested = False
    db.commit()
    db.refresh(job)
    return job


def claim_pending_job(db: Session) -> TrainingJob | None:
    """Worker/API: next pending job, with resume checkpoint preflight."""
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.status == JobStatus.pending.value)
        .order_by(TrainingJob.created_at)
        .first()
    )
    if job is None:
        return None
    if job.resume_from and not Path(job.resume_from).exists():
        job.status = JobStatus.failed.value
        job.error_message = f"resume checkpoint missing: {job.resume_from}"
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
        return None
    return job