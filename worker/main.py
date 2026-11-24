"""Standalone training worker — polls SQLite for pending jobs (optional split from API)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

os.chdir(ROOT)

from app.database import init_db  # noqa: E402
from app.models import JobStatus, TrainingJob  # noqa: E402
from app.services.ultralytics_trainer import run_training_job  # noqa: E402
from app.database import SessionLocal  # noqa: E402


def poll_loop(interval: float = 2.0) -> None:
    init_db()
    active: int | None = None
    while True:
        if active is None:
            with SessionLocal() as db:
                job = (
                    db.query(TrainingJob)
                    .filter(TrainingJob.status == JobStatus.pending.value)
                    .order_by(TrainingJob.created_at)
                    .first()
                )
                if job:
                    active = job.id
        if active is not None:
            try:
                run_training_job(active)
            finally:
                active = None
        time.sleep(interval)


if __name__ == "__main__":
    poll_loop()