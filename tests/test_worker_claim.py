from pathlib import Path

from app.database import SessionLocal, init_db
from app.models import JobStatus, TrainingJob
from app.services.job_state import claim_pending_job
from conftest import make_dataset

init_db()


def test_claim_returns_resume_ready_job(tmp_path: Path):
    ckpt = tmp_path / "last.pt"
    ckpt.write_bytes(b"x" * 100)
    with SessionLocal() as db:
        db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).update(
            {TrainingJob.status: JobStatus.cancelled.value},
            synchronize_session=False,
        )
        db.commit()
        ds = make_dataset(db, "worker-resume")
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
        assert claimed.status == JobStatus.running.value
        assert claimed.claimed_by is not None