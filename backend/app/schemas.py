from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetStats(BaseModel):
    num_images_train: int = 0
    num_images_val: int = 0
    num_labels_train: int = 0
    num_labels_val: int = 0
    class_names: list[str] = Field(default_factory=list)
    class_counts_train: dict[str, int] = Field(default_factory=dict)
    class_counts_val: dict[str, int] = Field(default_factory=dict)
    class_counts_total: dict[str, int] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    path: str | None = None


class DatasetValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)


class SplitConfig(BaseModel):
    mode: Literal["ratio", "manual"] = "ratio"
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = 42
    train_rel: str = "images/train"
    val_rel: str = "images/val"
    manual_val_files: list[str] = Field(default_factory=list)


class DatasetCreateResponse(BaseModel):
    id: int
    name: str
    yaml_path: str
    stats: DatasetStats
    validation: DatasetValidation


class DatasetOut(BaseModel):
    id: int
    name: str
    yaml_path: str
    root_dir: str | None = None
    stats: DatasetStats | None = None
    validation: DatasetValidation | None = None
    split_config: SplitConfig | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ClassDistribution(BaseModel):
    class_names: list[str]
    train: dict[str, int]
    val: dict[str, int]
    total: dict[str, int]


class AnnotationBox(BaseModel):
    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float
    x1: float
    y1: float
    x2: float
    y2: float


class SampleAnnotation(BaseModel):
    dataset_id: int
    split: str
    file_name: str
    image_width: int
    image_height: int
    boxes: list[AnnotationBox]


class SplitRequest(BaseModel):
    mode: Literal["ratio", "manual"] = "ratio"
    val_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    seed: int = 42
    manual_val_files: list[str] = Field(default_factory=list)


class TrainingJobCreate(BaseModel):
    name: str
    dataset_id: int
    epochs: int = Field(default=3, ge=1, le=500)
    batch_size: int = Field(default=4, ge=1, le=128)
    imgsz: int = Field(default=320, ge=64, le=1280)
    base_weights: str = "yolov8n.pt"
    resume_from: str | None = None


class TrainingJobOut(BaseModel):
    id: int
    name: str
    dataset_id: int
    parent_job_id: int | None = None
    status: str
    device: str
    epochs: int
    batch_size: int
    imgsz: int
    base_weights: str
    resume_from: str | None
    workspace_dir: str | None
    best_weights: str | None
    onnx_path: str | None
    metrics_json: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    model_config = {"from_attributes": True}


class MetricPointOut(BaseModel):
    epoch: int
    train_loss: float | None
    val_map50: float | None
    val_map50_95: float | None
    precision: float | None
    recall: float | None
    lr: float | None
    recorded_at: datetime

    model_config = {"from_attributes": True}


class DetectionBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class InferenceResponse(BaseModel):
    backend: str
    latency_ms: float
    image_width: int
    image_height: int
    detections: list[DetectionBox]
    tiled: bool = False
    tile_count: int = 0


class OnnxCompareRequest(BaseModel):
    job_id: int | None = None
    weights_path: str | None = None
    image_path: str | None = None
    conf: float = 0.25
    iou: float = 0.45


class OnnxCompareResponse(BaseModel):
    pytorch: InferenceResponse
    onnx: InferenceResponse
    iou_overlap_mean: float
    count_diff: int


class HardwareInfo(BaseModel):
    device: str
    cuda_available: bool
    cuda_device_name: str | None = None
    cpu_count: int


class ResourceSnapshot(BaseModel):
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    active_jobs: int
    queued_jobs: int


class QueueJobItem(BaseModel):
    id: int
    name: str
    status: str
    position: int | None = None
    created_at: datetime


class TrainingQueueOut(BaseModel):
    active_job_id: int | None
    running: TrainingJobOut | None
    pending: list[QueueJobItem]
    paused: list[QueueJobItem]


class CheckpointInfo(BaseModel):
    path: str
    kind: str
    size_bytes: int


class CheckpointVerifyResult(BaseModel):
    path: str
    loadable: bool
    message: str


class ExperimentSummary(BaseModel):
    job_id: int
    name: str
    status: str
    dataset_id: int
    parent_job_id: int | None
    epochs: int
    elapsed_seconds: float | None
    map50: float | None
    map50_95: float | None
    created_at: datetime


class ExperimentCompareRequest(BaseModel):
    job_ids: list[int] = Field(min_length=2, max_length=4)


class ExperimentSeries(BaseModel):
    job_id: int
    name: str
    epochs: list[int]
    train_loss: list[float | None]
    val_map50: list[float | None]
    val_map50_95: list[float | None]


class ExperimentCompareResponse(BaseModel):
    jobs: list[ExperimentSummary]
    series: list[ExperimentSeries]
    metric_table: list[dict[str, Any]]


class ConfusionMatrixResponse(BaseModel):
    available: bool
    matrix: list[list[float]] | None = None
    image_url: str | None = None
    class_names: list[str] = Field(default_factory=list)