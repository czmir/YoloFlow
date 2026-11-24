export type JobStatus =
  | "pending"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface DetectionBox {
  class_id: number;
  class_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface InferenceResponse {
  backend: string;
  latency_ms: number;
  image_width: number;
  image_height: number;
  detections: DetectionBox[];
  tiled?: boolean;
  tile_count?: number;
}

export interface ResourceSnapshot {
  timestamp: number;
  cpu_percent: number;
  memory_percent: number;
  memory_used_mb: number;
  active_jobs: number;
  queued_jobs: number;
  active_job_id?: number | null;
}