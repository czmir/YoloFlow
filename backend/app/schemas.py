from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetStats(BaseModel):
    num_images_train: int = 0
    num_images_val: int = 0
    num_labels_train: int = 0
    num_labels_val: int = 0
    class_names: list[str] = Field(default_factory=list)
    class_counts_train: dict[str, int] = Field(default_factory=dict)
    class_counts_val: dict[str, int] = Field(default_factory=dict)


class DatasetValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


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
    stats: DatasetStats | None = None
    validation: DatasetValidation | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


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