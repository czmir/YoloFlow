from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"
    paused = "paused"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    yaml_path: Mapped[str] = mapped_column(String(512))
    root_dir: Mapped[str] = mapped_column(String(512))
    stats_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    split_config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    jobs: Mapped[list[TrainingJob]] = relationship(back_populates="dataset")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    status: Mapped[str] = mapped_column(String(32), default=JobStatus.pending.value, index=True)
    device: Mapped[str] = mapped_column(String(32), default="cpu")
    epochs: Mapped[int] = mapped_column(Integer, default=3)
    batch_size: Mapped[int] = mapped_column(Integer, default=4)
    imgsz: Mapped[int] = mapped_column(Integer, default=320)
    base_weights: Mapped[str] = mapped_column(String(256), default="yolov8n.pt")
    resume_from: Mapped[str | None] = mapped_column(String(512), nullable=True)
    parent_job_id: Mapped[int | None] = mapped_column(ForeignKey("training_jobs.id"), nullable=True)
    workspace_dir: Mapped[str | None] = mapped_column(String(512), nullable=True)
    best_weights: Mapped[str | None] = mapped_column(String(512), nullable=True)
    onnx_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    log_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    pause_requested: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    dataset: Mapped[Dataset] = relationship(back_populates="jobs")
    metric_points: Mapped[list[MetricPoint]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class MetricPoint(Base):
    __tablename__ = "metric_points"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), index=True)
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_map50: Mapped[float | None] = mapped_column(Float, nullable=True)
    val_map50_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    lr: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    job: Mapped[TrainingJob] = relationship(back_populates="metric_points")


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("training_jobs.id"), nullable=True)
    backend: Mapped[str] = mapped_column(String(32))
    image_name: Mapped[str] = mapped_column(String(256))
    detections_json: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)