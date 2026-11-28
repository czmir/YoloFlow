from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from app.config import settings
from app.database import SessionLocal
from app.models import JobStatus, MetricPoint, TrainingJob
from app.services.experiment_artifacts import collect_post_train_artifacts
from app.services.job_state import (
    JobCancelled,
    JobPaused,
    claim_job_by_id,
    claim_pending_job,
    is_control_exception,
    lock_active_job,
    touch_heartbeat,
    worker_identity,
)
from app.services.model_registry import register_job_model
from app.services.onnx_export import export_to_onnx
from app.services.security import UnsafePathError, resolve_weight_path
from app.services.stdio import redirect_stdio_to


class JobLeaseLost(Exception):
    """The training attempt no longer owns the job."""


class IsolatedDetectionTrainer(DetectionTrainer):
    def check_resume(self, overrides):
        super().check_resume(overrides)
        if self.resume:
            # Ultralytics restores the previous output directory from the checkpoint.
            self.args.save_dir = str(Path(overrides["project"]) / overrides["name"])


class JobControlCallback:
    def __init__(self, job_id: int, owner: str, started_at: datetime):
        self.job_id = job_id
        self.owner = owner
        self.started_at = started_at

    def on_train_epoch_end(self, trainer) -> None:  # type: ignore[no-untyped-def]
        with SessionLocal() as db:
            job = lock_active_job(db, self.job_id, self.owner, self.started_at)
            if not job:
                trainer.stop = True
                raise JobLeaseLost(f"job {self.job_id} lease expired")
            if job.cancel_requested:
                trainer.stop = True
                raise JobCancelled(f"job {self.job_id} cancelled by user")
            if job.pause_requested:
                trainer.stop = True
                raise JobPaused(f"job {self.job_id} paused by user")


class MetricsCallback:
    """Captures per-epoch metrics via Ultralytics trainer hooks."""

    def __init__(self, job_id: int, owner: str, started_at: datetime):
        self.job_id = job_id
        self.owner = owner
        self.started_at = started_at

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
            if lock_active_job(db, self.job_id, self.owner, self.started_at) is None:
                raise JobLeaseLost(f"job {self.job_id} lease expired")
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


def _heartbeat_loop(job_id: int, owner: str, stop: threading.Event) -> None:
    while not stop.wait(15.0):
        try:
            with SessionLocal() as db:
                touch_heartbeat(db, job_id, owner=owner)
        except Exception:  # noqa: BLE001
            return


def run_training_job(job_id: int) -> None:
    owner = worker_identity()
    with SessionLocal() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            return
        if job.status == JobStatus.pending.value:
            job = claim_job_by_id(db, job_id, owner=owner)
            if job is None:
                return
        elif job.status == JobStatus.running.value:
            if job.claimed_by and job.claimed_by != owner:
                return
        else:
            return
        started_at = job.started_at
        if started_at is None:
            return
        job = lock_active_job(db, job_id, owner, started_at)
        if job is None:
            return
        workspace = settings.workspaces_dir / f"job_{job_id}" / uuid4().hex
        workspace.mkdir(parents=True, exist_ok=True)
        job.workspace_dir = str(workspace)
        log_path = workspace / "train.log"
        job.log_path = str(log_path)
        job.claimed_by = owner
        db.commit()
        yaml_path = job.dataset.yaml_path
        device = job.device
        resume_path = job.resume_from
        epochs = job.epochs
        batch = job.batch_size
        imgsz = job.imgsz
        weights = job.base_weights

    try:
        weight_spec = resolve_weight_path(resume_path or weights)
    except (UnsafePathError, FileNotFoundError) as exc:
        with SessionLocal() as db:
            job = lock_active_job(db, job_id, owner, started_at)
            if job:
                job.status = JobStatus.failed.value
                job.finished_at = datetime.now(timezone.utc)
                job.error_message = f"refused weight path: {exc}"
                job.claimed_by = None
                db.commit()
        return

    hb_stop = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat_loop, args=(job_id, owner, hb_stop), daemon=True)
    hb_thread.start()

    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    metrics_cb = MetricsCallback(job_id, owner, started_at)
    control_cb = JobControlCallback(job_id, owner, started_at)
    try:
        with redirect_stdio_to(log_file, __import__("sys").__stdout__):
            model = YOLO(str(weight_spec))
            model.add_callback("on_fit_epoch_end", metrics_cb.on_fit_epoch_end)
            model.add_callback("on_train_epoch_end", control_cb.on_train_epoch_end)

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
            results = model.train(trainer=IsolatedDetectionTrainer, **train_kwargs)
            elapsed = time.perf_counter() - t0

            best = workspace / "train" / "weights" / "best.pt"
            if not best.exists():
                best = workspace / "train" / "weights" / "last.pt"

            onnx_path = export_to_onnx(best, workspace / "model.onnx")

            metrics_summary = {}
            if hasattr(results, "results_dict"):
                metrics_summary = dict(results.results_dict)

            with SessionLocal() as db:
                job = lock_active_job(db, job_id, owner, started_at)
                if job:
                    job.status = JobStatus.succeeded.value
                    job.finished_at = datetime.now(timezone.utc)
                    job.best_weights = str(best) if best.exists() else None
                    job.onnx_path = str(onnx_path) if onnx_path else None
                    job.claimed_by = None
                    artifacts = collect_post_train_artifacts(job)
                    job.metrics_json = {
                        **metrics_summary,
                        **artifacts,
                        "elapsed_seconds": elapsed,
                        "resumed_from": job.resume_from,
                    }
                    register_job_model(db, job)
                    db.commit()
    except Exception as exc:  # noqa: BLE001
        cancelled = is_control_exception(exc, JobCancelled)
        paused = is_control_exception(exc, JobPaused)
        with SessionLocal() as db:
            job = lock_active_job(db, job_id, owner, started_at)
            if job:
                if cancelled:
                    job.status = JobStatus.cancelled.value
                    job.finished_at = datetime.now(timezone.utc)
                elif paused:
                    job.status = JobStatus.paused.value
                    job.pause_requested = False
                    last = Path(job.workspace_dir or "") / "train" / "weights" / "last.pt"
                    if last.exists():
                        job.resume_from = str(last)
                    job.finished_at = None
                else:
                    job.status = JobStatus.failed.value
                    job.finished_at = datetime.now(timezone.utc)
                job.error_message = f"{type(exc).__name__}: {exc}"
                job.claimed_by = None
                db.commit()
        if cancelled or paused or is_control_exception(exc, JobLeaseLost):
            return
        raise
    finally:
        hb_stop.set()
        log_file.close()


class TrainingQueue:
    """Single in-process worker. Standalone `worker/main.py` uses the same claim CAS."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_id: int | None = None
        self._thread: threading.Thread | None = None
        self._starting = False
        self._pending_kick = False
        self.enabled = True

    def enqueue(self, job_id: int | None = None) -> None:
        """Compatibility: ignore explicit ids and claim via CAS."""
        _ = job_id
        self.kick()

    def _run(self, job_id: int) -> None:
        try:
            run_training_job(job_id)
        finally:
            with self._lock:
                self._active_id = None
                self._thread = None
            self.kick()

    def kick(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._active_id is not None or self._starting:
                self._pending_kick = True
                return
            self._starting = True
        job_id: int | None = None
        try:
            with SessionLocal() as db:
                pending = claim_pending_job(db)
                if pending:
                    job_id = pending.id
            if job_id is None:
                return
            with self._lock:
                self._active_id = job_id
                self._thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
                self._thread.start()
        finally:
            start_again = False
            with self._lock:
                self._starting = False
                start_again = self._pending_kick and self._active_id is None
                self._pending_kick = False
            if start_again:
                self.kick()

    @property
    def active_job_id(self) -> int | None:
        with self._lock:
            return self._active_id


training_queue = TrainingQueue()
