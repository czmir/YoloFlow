from pathlib import Path

from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import Dataset, JobStatus, TrainingJob
from app.services.job_state import claim_pending_job

init_db()


def _ds(db: Session, tag: str) -> Dataset:
    row = Dataset(
        name=f"worker-ds-{tag}",
        yaml_path="/tmp/w.yaml",
        root_dir="/tmp",
        validation_json={"valid": True},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def test_claim_returns_resume_ready_job(tmp_path: Path):
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x" * 100)
    with SessionLocal() as db:
        db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).update(
            {TrainingJob.status: JobStatus.cancelled.value}
        )
        db.commit()
        ds = _ds(db, "resume")
        job = TrainingJob(
            name="worker-resume",
            dataset_id=ds.id,
            status=JobStatus.pending.value,
            resume_from=str(ckpt),
            base_weights=str(ckpt),
        )
        db.add(job)
        db.commit()
        claimed = claim_pending_job(db)
        assert claimed is not None
        assert claimed.id == job.id
        assert claimed.resume_from == str(ckpt)