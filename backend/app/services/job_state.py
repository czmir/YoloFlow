from __future__ import annotations

import os
from uuid import uuid4
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from app.config import settings
from app.models import JobStatus, TrainingJob

TERMINAL = {JobStatus.succeeded.value, JobStatus.failed.value, JobStatus.cancelled.value}
_WORKER_INSTANCE = uuid4().hex


class JobCancelled(Exception):
    """Cooperative cancel requested by the user."""


class JobPaused(Exception):
    """Cooperative pause requested by the user."""


def worker_identity() -> str:
    return f"{settings.worker_name}:{os.getpid()}:{_WORKER_INSTANCE}"


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
        job.claimed_by = None
    elif job.status == JobStatus.paused.value:
        job.status = JobStatus.cancelled.value
        job.finished_at = datetime.now(timezone.utc)
        job.claimed_by = None
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
        job.claimed_by = None
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
    job.claimed_by = None
    db.commit()
    db.refresh(job)
    return job


def _preflight_resume(db: Session, job: TrainingJob) -> bool:
    if job.resume_from and not Path(job.resume_from).exists():
        job.status = JobStatus.failed.value
        job.error_message = f"resume checkpoint missing: {job.resume_from}"
        job.finished_at = datetime.now(timezone.utc)
        job.claimed_by = None
        db.commit()
        return False
    return True


def _cas_pending_to_running(db: Session, job_id: int, owner: str) -> bool:
    now = datetime.now(timezone.utc)
    result = db.execute(
        update(TrainingJob)
        .where(
            TrainingJob.id == job_id,
            TrainingJob.status == JobStatus.pending.value,
        )
        .values(
            status=JobStatus.running.value,
            started_at=now,
            claimed_by=owner,
            heartbeat_at=now,
        )
    )
    db.commit()
    return result.rowcount == 1


def claim_pending_job(db: Session, owner: str | None = None) -> TrainingJob | None:
    """Atomically claim the next pending job (compare-and-swap pending → running)."""
    owner = owner or worker_identity()
    while True:
        job = (
            db.query(TrainingJob)
            .filter(TrainingJob.status == JobStatus.pending.value)
            .order_by(TrainingJob.created_at)
            .first()
        )
        if job is None:
            return None
        if not _preflight_resume(db, job):
            continue
        job_id = job.id
        if not _cas_pending_to_running(db, job_id, owner):
            continue
        claimed = db.get(TrainingJob, job_id)
        return claimed


def claim_job_by_id(db: Session, job_id: int, owner: str | None = None) -> TrainingJob | None:
    """Claim a specific pending job. Returns None if another worker won the race."""
    owner = owner or worker_identity()
    job = db.get(TrainingJob, job_id)
    if job is None:
        return None
    if job.status == JobStatus.running.value and job.claimed_by == owner:
        return job
    if job.status != JobStatus.pending.value:
        return None
    if not _preflight_resume(db, job):
        return None
    if not _cas_pending_to_running(db, job_id, owner):
        return None
    return db.get(TrainingJob, job_id)


def touch_heartbeat(db: Session, job_id: int, owner: str | None = None) -> None:
    owner = owner or worker_identity()
    now = datetime.now(timezone.utc)
    db.execute(
        update(TrainingJob)
        .where(
            TrainingJob.id == job_id,
            TrainingJob.status == JobStatus.running.value,
            TrainingJob.claimed_by == owner,
            TrainingJob.heartbeat_at > now - timedelta(seconds=settings.job_lease_seconds),
        )
        .values(heartbeat_at=now)
        .execution_options(synchronize_session=False)
    )
    db.commit()


def lock_active_job(
    db: Session, job_id: int, owner: str, started_at: datetime
) -> TrainingJob | None:
    """Hold the SQLite write transaction while publishing changes for a live claim."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.job_lease_seconds)
    result = db.execute(
        update(TrainingJob)
        .where(
            TrainingJob.id == job_id,
            TrainingJob.status == JobStatus.running.value,
            TrainingJob.claimed_by == owner,
            TrainingJob.started_at == started_at,
            TrainingJob.heartbeat_at > cutoff,
        )
        .values(heartbeat_at=TrainingJob.heartbeat_at)
        .execution_options(synchronize_session=False)
    )
    if result.rowcount != 1:
        return None
    return db.get(TrainingJob, job_id, populate_existing=True)


def recover_orphan_jobs(db: Session, *, now: datetime | None = None) -> list[int]:
    """Re-queue running jobs whose lease expired (crash / restart).

    If `last.pt` exists, resume from that checkpoint; otherwise keep original weights.
    """
    now = now or datetime.now(timezone.utc)
    stale_before = now - timedelta(seconds=settings.job_lease_seconds)
    recovered: list[int] = []
    running = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.running.value).all()
    for job in running:
        hb = job.heartbeat_at
        if hb is not None and hb.tzinfo is None:
            hb = hb.replace(tzinfo=timezone.utc)
        if hb is not None and hb > stale_before:
            continue
        last_pt = None
        if job.workspace_dir:
            candidate = Path(job.workspace_dir) / "train" / "weights" / "last.pt"
            if candidate.exists():
                last_pt = str(candidate)
        values = dict(
            status=JobStatus.pending.value,
            cancel_requested=False,
            pause_requested=False,
            claimed_by=None,
            heartbeat_at=None,
            error_message="recovered after process restart",
        )
        if last_pt:
            values.update(resume_from=last_pt, base_weights=last_pt)
        job_id = job.id
        result = db.execute(
            update(TrainingJob)
            .where(
                TrainingJob.id == job_id,
                TrainingJob.status == JobStatus.running.value,
                TrainingJob.claimed_by == job.claimed_by,
                or_(TrainingJob.heartbeat_at.is_(None), TrainingJob.heartbeat_at <= stale_before),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 1:
            recovered.append(job_id)
    if recovered:
        db.commit()
    return recovered


def is_control_exception(exc: BaseException, typ: type[BaseException]) -> bool:
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, typ):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False
