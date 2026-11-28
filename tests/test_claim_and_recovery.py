import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.database import SessionLocal, init_db
from app.models import JobStatus, TrainingJob
from app.services.job_state import (
    JobCancelled,
    JobPaused,
    claim_pending_job,
    is_control_exception,
    lock_active_job,
    recover_orphan_jobs,
    touch_heartbeat,
)
from conftest import make_dataset

init_db()


def _clear_pending() -> None:
    with SessionLocal() as db:
        db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).update(
            {TrainingJob.status: JobStatus.cancelled.value},
            synchronize_session=False,
        )
        db.commit()


def test_claim_is_exclusive():
    _clear_pending()
    with SessionLocal() as db:
        ds = make_dataset(db, "cas")
        job = TrainingJob(name="cas", dataset_id=ds.id, status=JobStatus.pending.value)
        db.add(job)
        db.commit()
        job_id = job.id

    winners: list[int] = []
    lock = threading.Lock()

    def worker(owner: str) -> None:
        with SessionLocal() as db:
            claimed = claim_pending_job(db, owner=owner)
            if claimed is not None:
                with lock:
                    winners.append(claimed.id)

    threads = [threading.Thread(target=worker, args=(f"w{i}:1",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert winners == [job_id]
    with SessionLocal() as db:
        row = db.get(TrainingJob, job_id)
        assert row is not None
        assert row.status == JobStatus.running.value
        assert row.claimed_by is not None


def test_recover_orphan_running_job(tmp_path: Path):
    workspace = tmp_path / "job_ws"
    (workspace / "train" / "weights").mkdir(parents=True)
    last = workspace / "train" / "weights" / "last.pt"
    last.write_bytes(b"ckpt")
    with SessionLocal() as db:
        ds = make_dataset(db, "orphan")
        job = TrainingJob(
            name="orphan",
            dataset_id=ds.id,
            status=JobStatus.running.value,
            workspace_dir=str(workspace),
            claimed_by="dead:999999",
            heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        db.add(job)
        db.commit()
        jid = job.id
        recovered = recover_orphan_jobs(db)
    assert jid in recovered
    with SessionLocal() as db:
        row = db.get(TrainingJob, jid)
        assert row is not None
        assert row.status == JobStatus.pending.value
        assert row.resume_from == str(last)
        assert row.claimed_by is None


def test_fresh_heartbeat_not_recovered():
    with SessionLocal() as db:
        ds = make_dataset(db, "live")
        job = TrainingJob(
            name="live",
            dataset_id=ds.id,
            status=JobStatus.running.value,
            claimed_by="api:1",
            heartbeat_at=datetime.now(timezone.utc),
        )
        db.add(job)
        db.commit()
        jid = job.id
        recovered = recover_orphan_jobs(db)
    assert jid not in recovered
    with SessionLocal() as db:
        row = db.get(TrainingJob, jid)
        assert row is not None
        assert row.status == JobStatus.running.value


def test_control_exceptions_not_substring():
    wrapped = RuntimeError("training failed")
    wrapped.__cause__ = JobCancelled("job 1 cancelled by user")
    assert is_control_exception(wrapped, JobCancelled)
    assert not is_control_exception(RuntimeError("export cancelled by onnx"), JobCancelled)
    assert not is_control_exception(ValueError("paused video file"), JobPaused)
    assert is_control_exception(JobPaused("job 2 paused by user"), JobPaused)


def test_expired_worker_cannot_renew_lease():
    with SessionLocal() as db:
        ds = make_dataset(db, "expired-renewal")
        owner = "expired:123"
        job = TrainingJob(
            name="expired-renewal",
            dataset_id=ds.id,
            status=JobStatus.running.value,
            claimed_by=owner,
            heartbeat_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        )
        db.add(job)
        db.commit()
        job_id = job.id

        touch_heartbeat(db, job_id, owner=owner)

        assert job_id in recover_orphan_jobs(db)
        assert db.get(TrainingJob, job_id).status == JobStatus.pending.value


def test_old_attempt_cannot_publish_after_reclaim():
    _clear_pending()
    with SessionLocal() as db:
        dataset = make_dataset(db, "old-attempt")
        db.add(TrainingJob(name="old-attempt", dataset_id=dataset.id))
        db.commit()
        first = claim_pending_job(db, owner="same-worker")
        job_id, old_started_at = first.id, first.started_at
        first.heartbeat_at = datetime(2000, 1, 1, tzinfo=timezone.utc)
        db.commit()
        assert job_id in recover_orphan_jobs(db)
        current = claim_pending_job(db, owner="same-worker")
        assert current.id == job_id
        assert lock_active_job(db, job_id, "same-worker", old_started_at) is None
        assert lock_active_job(db, job_id, "same-worker", current.started_at) is not None


def test_acceptance_onnx_skip_is_failure():
    from app.services.acceptance_report import overall_exit_code

    assert (
        overall_exit_code(
            {"job": {"status": "succeeded"}, "onnx_compare": {"skipped": True, "reason": "onnx export missing"}}
        )
        == 3
    )
    assert overall_exit_code({"job": {"status": "succeeded"}, "onnx_compare": {"skipped": False}}) == 0
    assert overall_exit_code({"job": {"status": "failed"}}) == 2
