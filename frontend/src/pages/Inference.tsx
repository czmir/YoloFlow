import { useEffect, useRef, useState } from "react";
import { apiGet, apiPostForm } from "../api/client";
import type { InferenceResponse } from "../../../shared/typescript/types";

interface Job {
  id: number;
  name: string;
  status: string;
}

export default function InferencePage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState<number | "">("");
  const [file, setFile] = useState<File | null>(null);
  const [tiled, setTiled] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const previewRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    apiGet<Job[]>("/training/jobs")
      .then((rows) => setJobs(rows.filter((j) => j.status === "succeeded")))
      .catch(() => setJobs([]));
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

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setError(null);
    const form = new FormData();
    form.append("file", file);
    if (jobId !== "") form.append("job_id", String(jobId));
    form.append("tiled", String(tiled));
    try {
      const res = await apiPostForm<InferenceResponse>("/inference/detect", form);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <label style={{ display: "block" }}>
      <h2>Inference</h2>
      <div className="card">
        <form onSubmit={onSubmit}>
          <p>
            <label>
              Model job&nbsp;
              <select value={jobId} onChange={(e) => setJobId(e.target.value ? Number(e.target.value) : "")}>
                <option value="">latest succeeded job</option>
                {jobs.map((j) => (
                  <option key={j.id} value={j.id}>
                    #{j.id} {j.name}
                  </option>
                ))}
              </select>
            </label>
          </p>
          <p>
            <input type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </p>
          <p>
            <label>
              <input type="checkbox" checked={tiled} onChange={(e) => setTiled(e.target.checked)} /> Large-image tiling + NMS
            </label>
          </p>
          <button type="submit">Detect</button>
        </form>
        {error && <p className="mono">{error}</p>}
      </div>
      {result && (
        <div className="card">
          <p className="mono">
            backend={result.backend} latency={result.latency_ms.toFixed(1)}ms detections={result.detections.length}
            {result.tiled ? ` tiles=${result.tile_count}` : ""}
          </p>
          <div>
            <img ref={previewRef} src={file ? URL.createObjectURL(file) : ""} alt="preview" style={{ maxWidth: "100%", display: "none" }} />
            <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />
          </div>
        </div>
      )}
    </label>
  );
}