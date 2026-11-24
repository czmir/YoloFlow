from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.database import SessionLocal, init_db
from app.main import app
from app.models import Dataset, JobStatus, TrainingJob
from app.services.job_state import apply_cancel, apply_pause, apply_unpause, can_resume_new_job, claim_pending_job

init_db()
client = TestClient(app)


def _dataset(db, suffix: str) -> Dataset:
    row = Dataset(
        name=f"test-ds-{suffix}",
        yaml_path="/tmp/data.yaml",
        root_dir="/tmp",
        stats_json={"class_names": ["a"]},
        validation_json={"valid": True, "errors": [], "warnings": []},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_cancel_pending_job():
    with SessionLocal() as db:
        ds = _dataset(db, "j1")
        job = TrainingJob(name="j1", dataset_id=ds.id, status=JobStatus.pending.value)
        db.add(job)
        db.commit()
        db.refresh(job)
        apply_cancel(db, job)
        assert job.status == JobStatus.cancelled.value


def test_pause_and_unpause_pending():
    with SessionLocal() as db:
        ds = _dataset(db, "j2")
        job = TrainingJob(name="j2", dataset_id=ds.id, status=JobStatus.pending.value)
        db.add(job)
        db.commit()
        db.refresh(job)
        apply_pause(db, job)
        assert job.status == JobStatus.paused.value
        apply_unpause(db, job)
        assert job.status == JobStatus.pending.value


def test_cannot_cancel_succeeded():
    with SessionLocal() as db:
        ds = _dataset(db, "j3")
        job = TrainingJob(name="j3", dataset_id=ds.id, status=JobStatus.succeeded.value)
        db.add(job)
        db.commit()
        db.refresh(job)
        with pytest.raises(ValueError):
            apply_cancel(db, job)


def test_claim_skips_missing_resume_checkpoint(tmp_path: Path):
    with SessionLocal() as db:
        db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).update(
            {TrainingJob.status: JobStatus.cancelled.value}
        )
        db.commit()
        ds = _dataset(db, "res-miss")
        job = TrainingJob(
            name="resume-missing",
            dataset_id=ds.id,
            status=JobStatus.pending.value,
            resume_from=str(tmp_path / "nope.pt"),
        )
        db.add(job)
        db.commit()
        claimed = claim_pending_job(db)
        assert claimed is None
        db.refresh(job)
        assert job.status == JobStatus.failed.value


def test_resume_new_job_requires_checkpoint(tmp_path: Path):
    with SessionLocal() as db:
        ds = _dataset(db, "parent")
        job = TrainingJob(name="parent", dataset_id=ds.id, status=JobStatus.succeeded.value)
        db.add(job)
        db.commit()
        ok, msg = can_resume_new_job(job)
        assert not ok

        ckpt = tmp_path / "best.pt"
        ckpt.write_bytes(b"fake")
        job.best_weights = str(ckpt)
        db.commit()
        ok, msg = can_resume_new_job(job)
        assert ok


def test_api_cancel_pending():
    with SessionLocal() as db:
        ds = _dataset(db, "api-cancel")
        job = TrainingJob(name="api-cancel", dataset_id=ds.id, status=JobStatus.pending.value)
        db.add(job)
        db.commit()
        jid = job.id
    r = client.post(f"/api/v1/training/jobs/{jid}/cancel")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"


def test_api_double_cancel_succeeded_is_409():
    with SessionLocal() as db:
        ds = _dataset(db, "done")
        job = TrainingJob(name="done", dataset_id=ds.id, status=JobStatus.succeeded.value)
        db.add(job)
        db.commit()
        jid = job.id
    r = client.post(f"/api/v1/training/jobs/{jid}/cancel")
    assert r.status_code == 409


def test_experiments_list_and_compare_empty():
    r = client.get("/api/v1/experiments")
    assert r.status_code == 200
    assert isinstance(r.json(), list)