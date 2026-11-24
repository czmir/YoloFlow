"""Standalone training worker — polls SQLite for pending jobs including resume-from-checkpoint."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

os.chdir(ROOT)

from app.database import SessionLocal, init_db  # noqa: E402
from app.services.job_state import claim_pending_job  # noqa: E402
from app.services.ultralytics_trainer import run_training_job  # noqa: E402


def poll_loop(interval: float = 2.0) -> None:
    init_db()
    while True:
        with SessionLocal() as db:
            job = claim_pending_job(db)
            job_id = job.id if job else None
        if job_id is not None:
            try:
                run_training_job(job_id)
            except Exception as exc:  # noqa: BLE001
                print(f"worker: job {job_id} failed: {exc}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    poll_loop()