import { useEffect, useRef, useState } from "react";
import { apiGet, apiPostForm } from "../api/client";
import type { InferenceResponse } from "../../../shared/typescript/types";

interface Job {
  id: number;
  name: string;
  status: string;
}

interface ModelVersion {
  id: number;
  job_id: number;
  version: string;
}

interface BackendComparison {
  pytorch: InferenceResponse;
  onnx: InferenceResponse;
  iou_overlap_mean: number;
}

export default function InferencePage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [models, setModels] = useState<ModelVersion[]>([]);
  const [jobId, setJobId] = useState<number | "">("");
  const [modelId, setModelId] = useState<number | "">("");
  const [file, setFile] = useState<File | null>(null);
  const [batchFiles, setBatchFiles] = useState<FileList | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [tiled, setTiled] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [batchCount, setBatchCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);
  const [comparison, setComparison] = useState<InferenceResponse[]>([]);
  const [overlap, setOverlap] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    apiGet<Job[]>("/training/jobs")
      .then((rows) => setJobs(rows.filter((j) => j.status === "succeeded")))
      .catch(() => setJobs([]));
    apiGet<ModelVersion[]>("/models").then(setModels).catch(() => setModels([]));
  }, []);

  useEffect(() => {
    if (!result || !previewRef.current || !canvasRef.current) return;
    const img = previewRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      ctx.lineWidth = 2;
      ctx.font = "14px sans-serif";
      for (const d of result.detections) {
        ctx.strokeStyle = "#2563eb";
        ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
        ctx.fillStyle = "#2563eb";
        ctx.fillText(`${d.class_name} ${(d.confidence * 100).toFixed(1)}%`, d.x1, Math.max(12, d.y1 - 4));
      }
    };

    if (img.complete) draw();
    else img.onload = draw;
  }, [result]);

  function appendModelFields(form: FormData) {
    if (modelId !== "") form.append("model_version_id", String(modelId));
    else if (jobId !== "") form.append("job_id", String(jobId));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    const form = new FormData();
    form.append("file", file);
    appendModelFields(form);
    form.append("tiled", String(tiled));
    try {
      const res = await apiPostForm<InferenceResponse>("/inference/detect", form);
      setResult(res);
      setBatchCount(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onCompare(backends: boolean) {
    if (!file) return;
    setComparing(true);
    setError(null);
    setComparison([]);
    setOverlap(null);
    const form = new FormData();
    form.append("file", file);
    appendModelFields(form);
    try {
      if (backends) {
        const run = await apiPostForm<BackendComparison>("/inference/compare-onnx", form);
        setComparison([run.pytorch, run.onnx]);
        setOverlap(run.iou_overlap_mean);
      } else {
        form.set("tile_size", "320");
        form.set("imgsz", "320");
        form.set("tiled", "false");
        const full = await apiPostForm<InferenceResponse>("/inference/detect", form);
        form.set("tiled", "true");
        const sliced = await apiPostForm<InferenceResponse>("/inference/detect", form);
        setComparison([full, sliced]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setComparing(false);
    }
  }

  async function onBatch() {
    if (!batchFiles?.length) return;
    const form = new FormData();
    for (const f of Array.from(batchFiles)) form.append("files", f);
    appendModelFields(form);
    form.append("tiled", String(tiled));
    try {
      const res = await apiPostForm<{ items: unknown[] }>("/inference/detect-batch", form);
      setBatchCount(res.items.length);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onVideo() {
    if (!videoFile) return;
    const form = new FormData();
    form.append("file", videoFile);
    appendModelFields(form);
    try {
      const res = await apiPostForm<{ items: unknown[] }>("/inference/detect-video", form);
      setBatchCount(res.items.length);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function onExport(fmt: "json" | "csv") {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    appendModelFields(form);
    form.append("format", fmt);
    form.append("tiled", String(tiled));
    const res = await fetch("/api/v1/inference/export-detections", { method: "POST", body: form });
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = fmt === "csv" ? "detections.csv" : "detections.json";
    a.click();
  }

  return (
    <div>
      <h2>Inference</h2>
      <div className="card">
        <form onSubmit={onSubmit}>
          <p>
            <select value={modelId} onChange={(e) => setModelId(e.target.value ? Number(e.target.value) : "")}>
              <option value="">Registry model…</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.version} (job #{m.job_id})
                </option>
              ))}
            </select>
            <select
              data-testid="infer-job"
              value={jobId}
              onChange={(e) => setJobId(e.target.value ? Number(e.target.value) : "")}
              style={{ marginLeft: 8 }}
            >
              <option value="">Or training job…</option>
              {jobs.map((j) => (
                <option key={j.id} value={j.id}>
                  #{j.id} {j.name}
                </option>
              ))}
            </select>
          </p>
          <p>
            <input
              data-testid="infer-file"
              type="file"
              accept="image/*"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </p>
          <p>
            <label>
              <input
                data-testid="infer-tiled"
                type="checkbox"
                checked={tiled}
                onChange={(e) => setTiled(e.target.checked)}
              />{" "}
              Tiled large-image + class-aware NMS
            </label>
          </p>
          <button type="submit" data-testid="infer-submit">
            Detect image
          </button>
          <button type="button" className="secondary" onClick={() => onExport("json")} style={{ marginLeft: 8 }}>
            Export JSON
          </button>
          <button type="button" className="secondary" onClick={() => onExport("csv")}>
            Export CSV
          </button>
          <p style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button type="button" disabled={!file || comparing} onClick={() => onCompare(false)}>
              Compare full image and slices
            </button>
            <button type="button" disabled={!file || comparing || (jobId === "" && modelId === "")} onClick={() => onCompare(true)}>
              Compare PyTorch and ONNX
            </button>
          </p>
          {comparing && <p role="status">Running comparison…</p>}
        </form>
        <p style={{ marginTop: 12 }}>
          <input type="file" accept="image/*" multiple onChange={(e) => setBatchFiles(e.target.files)} />
          <button type="button" onClick={onBatch} style={{ marginLeft: 8 }}>
            Batch detect
          </button>
        </p>
        <p>
          <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)} />
          <button type="button" onClick={onVideo} style={{ marginLeft: 8 }}>
            Video (sample frames)
          </button>
        </p>
        {batchCount !== null && <p className="mono">Processed {batchCount} frame(s)/image(s)</p>}
        {error && <p className="mono">{error}</p>}
      </div>
      {comparison.length > 0 && (
        <section className="card" aria-label="Inference comparison">
          <h3>Inference comparison</h3>
          <table>
            <thead><tr><th>Mode</th><th>Detections</th><th>Inference time</th><th>Passes</th></tr></thead>
            <tbody>
              {comparison.map((mode) => (
                <tr key={mode.backend}>
                  <td>{mode.backend}</td>
                  <td>{mode.detections.length}</td>
                  <td>{mode.latency_ms.toFixed(1)} ms</td>
                  <td>{mode.tiled ? mode.tile_count : 1}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {overlap !== null && <p>Mean best box overlap {overlap.toFixed(3)} IoU</p>}
        </section>
      )}
      {result && (
        <div className="card">
          <p className="mono" data-testid="infer-result">
            backend={result.backend} latency={result.latency_ms.toFixed(1)}ms detections={result.detections.length}
            {result.tiled ? ` tiles=${result.tile_count}` : ""}
          </p>
          <img ref={previewRef} src={file ? URL.createObjectURL(file) : ""} alt="preview" style={{ display: "none" }} />
          <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />
        </div>
      )}
    </div>
  );
}
