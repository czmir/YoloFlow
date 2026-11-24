"""Standalone training worker — polls SQLite; shares CAS claim with the API queue."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

os.chdir(ROOT)
os.environ.setdefault("YOLOFLOW_WORKER_NAME", "standalone")
os.environ.setdefault("YOLOFLOW_EMBEDDED_WORKER", "false")

from app.config import settings  # noqa: E402
from app.database import SessionLocal, init_db  # noqa: E402
from app.services.job_state import claim_pending_job, recover_orphan_jobs, worker_identity  # noqa: E402
from app.services.ultralytics_trainer import run_training_job, training_queue  # noqa: E402

training_queue.enabled = False


def poll_loop(interval: float | None = None) -> None:
    init_db()
    settings.worker_name = os.environ.get("YOLOFLOW_WORKER_NAME", "standalone")
    with SessionLocal() as db:
        recovered = recover_orphan_jobs(db)
        if recovered:
            print(f"worker: recovered orphan jobs {recovered}", flush=True)
    owner = worker_identity()
    interval = interval if interval is not None else settings.training_queue_poll_seconds
    while True:
        with SessionLocal() as db:
            job = claim_pending_job(db, owner=owner)
            job_id = job.id if job else None
        if job_id is not None:
            try:
                run_training_job(job_id)
            except Exception as exc:  # noqa: BLE001
                print(f"worker: job {job_id} failed: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    poll_loop()
