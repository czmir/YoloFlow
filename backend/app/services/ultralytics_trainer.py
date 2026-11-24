from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

from app.config import settings
from app.database import SessionLocal
from app.models import JobStatus, MetricPoint, TrainingJob
from app.services.onnx_export import export_to_onnx


class MetricsCallback:
    """Captures per-epoch metrics via Ultralytics trainer hooks."""

    def __init__(self, job_id: int):
        self.job_id = job_id

    def on_fit_epoch_end(self, trainer) -> None:  # type: ignore[no-untyped-def]
        metrics = trainer.metrics or {}
        tloss = None
        if trainer.loss_items is not None:
            try:
                tloss = float(sum(trainer.loss_items))
            except Exception:  # noqa: BLE001
                tloss = None
        lr = None
        try:
            lr = float(trainer.optimizer.param_groups[0]["lr"])
        except Exception:  # noqa: BLE001
            lr = None

        with SessionLocal() as db:
            db.add(
                MetricPoint(
                    job_id=self.job_id,
                    epoch=int(trainer.epoch) + 1,
                    train_loss=tloss,
                    val_map50=_safe_float(metrics.get("metrics/mAP50(B)")),
                    val_map50_95=_safe_float(metrics.get("metrics/mAP50-95(B)")),
                    precision=_safe_float(metrics.get("metrics/precision(B)")),
                    recall=_safe_float(metrics.get("metrics/recall(B)")),
                    lr=lr,
                )
            )
            db.commit()


def _safe_float(value) -> float | None:  # type: ignore[no-untyped-def]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_training_job(job_id: int) -> None:
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        job.status = JobStatus.running.value
        job.started_at = datetime.now(timezone.utc)
        workspace = settings.workspaces_dir / f"job_{job_id}"
        workspace.mkdir(parents=True, exist_ok=True)
        job.workspace_dir = str(workspace)
        log_path = workspace / "train.log"
        job.log_path = str(log_path)
        db.commit()
        dataset = job.dataset
        yaml_path = dataset.yaml_path
        device = job.device
        resume_path = job.resume_from
        epochs = job.epochs
        batch = job.batch_size
        imgsz = job.imgsz
        weights = job.base_weights

    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    class _Tee:
        def write(self, data: str) -> None:
            sys.__stdout__.write(data)
            log_file.write(data)
            log_file.flush()

        def flush(self) -> None:
            sys.__stdout__.flush()
            log_file.flush()

    old_stdout = sys.stdout
    sys.stdout = _Tee()  # type: ignore[assignment]

    callback = MetricsCallback(job_id)
    try:
        model = YOLO(resume_path or weights)
        model.add_callback("on_fit_epoch_end", callback.on_fit_epoch_end)

        train_kwargs = {
            "data": yaml_path,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "project": str(workspace),
            "name": "train",
            "exist_ok": True,
            "verbose": True,
            "workers": 0 if device == "cpu" else 2,
        }
        if resume_path:
            train_kwargs["resume"] = True

        t0 = time.perf_counter()
        results = model.train(**train_kwargs)
        elapsed = time.perf_counter() - t0

        best = workspace / "train" / "weights" / "best.pt"
        if not best.exists():
            best = workspace / "train" / "weights" / "last.pt"

        onnx_path = export_to_onnx(best, workspace / "model.onnx")

        metrics_summary = {}
        if hasattr(results, "results_dict"):
            metrics_summary = dict(results.results_dict)

        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            if job:
                job.status = JobStatus.succeeded.value
                job.finished_at = datetime.now(timezone.utc)
                job.best_weights = str(best) if best.exists() else None
                job.onnx_path = str(onnx_path) if onnx_path else None
                job.metrics_json = {
                    **metrics_summary,
                    "elapsed_seconds": elapsed,
                }
                db.commit()
    except Exception as exc:  # noqa: BLE001
        with SessionLocal() as db:
            job = db.get(TrainingJob, job_id)
            if job:
                job.status = JobStatus.failed.value
                job.finished_at = datetime.now(timezone.utc)
                job.error_message = str(exc)
                db.commit()
        raise
    finally:
        sys.stdout = old_stdout
        log_file.close()


class TrainingQueue:
    """Single-worker in-process queue; worker container uses the same code path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_id: int | None = None
        self._thread: threading.Thread | None = None

    def enqueue(self, job_id: int) -> None:
        with self._lock:
            if self._active_id is not None:
                return
            self._active_id = job_id
            self._thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            self._thread.start()

    def _run(self, job_id: int) -> None:
        try:
            run_training_job(job_id)
        finally:
            with self._lock:
                self._active_id = None
                self._thread = None
            self._dispatch_next()

    def _dispatch_next(self) -> None:
        with SessionLocal() as db:
            pending = (
                db.query(TrainingJob)
                .filter(TrainingJob.status == JobStatus.pending.value)
                .order_by(TrainingJob.created_at)
                .first()
            )
            if pending:
                self.enqueue(pending.id)

    @property
    def active_job_id(self) -> int | None:
        with self._lock:
            return self._active_id


training_queue = TrainingQueue()