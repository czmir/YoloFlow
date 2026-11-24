import { useEffect, useState } from "react";
import { apiGet, apiPost } from "../api/client";

interface DatasetRow {
  id: number;
  name: string;
}

interface JobRow {
  id: number;
  name: string;
  status: string;
  device: string;
  epochs: number;
  metrics_json?: Record<string, unknown>;
}

export default function TrainingPage() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [form, setForm] = useState({ name: "demo-train", dataset_id: 0, epochs: 3, batch_size: 4, imgsz: 320 });

  const [queue, setQueue] = useState<{ pending: { id: number; name: string; position?: number }[]; active_job_id: number | null } | null>(null);

  async function refresh() {
    const [ds, js, q] = await Promise.all([
      apiGet<DatasetRow[]>("/datasets"),
      apiGet<JobRow[]>("/training/jobs"),
      apiGet<{ pending: { id: number; name: string; position?: number }[]; active_job_id: number | null }>("/training/queue"),
    ]);
    setQueue(q);
    setDatasets(ds);
    setJobs(js);
    if (ds.length && form.dataset_id === 0) setForm((f) => ({ ...f, dataset_id: ds[0].id }));
  }

  useEffect(() => {
    const id = setInterval(() => refresh().catch(console.error), 3000);
    refresh().catch(console.error);
    return () => clearInterval(id);
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    await apiPost("/training/jobs", form);
    await refresh();
  }

  async function cancelJob(id: number) {
    await apiPost(`/training/jobs/${id}/cancel`);
    await refresh();
  }

  async function pauseJob(id: number) {
    await apiPost(`/training/jobs/${id}/pause`);
    await refresh();
  }

  return (
    <div>
      <h2>Training</h2>
      {queue && (
        <div className="card mono">
          Queue: active #{queue.active_job_id ?? "-"} | pending {queue.pending.length}
          <ul>
            {queue.pending.map((p) => (
              <li key={p.id}>
                #{p.position} job {p.id} {p.name}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div className="card">
        <form onSubmit={submit}>
          <p>
            <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="job name" />
            <select value={form.dataset_id} onChange={(e) => setForm({ ...form, dataset_id: Number(e.target.value) })}>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name}
                </option>
              ))}
            </select>
            <input type="number" value={form.epochs} onChange={(e) => setForm({ ...form, epochs: Number(e.target.value) })} style={{ width: 70 }} />
            <button type="submit">Start job</button>
          </p>
        </form>
      </div>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Status</th>
              <th>Device</th>
              <th>Elapsed(s)</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id}>
                <td>{j.id}</td>
                <td>{j.name}</td>
                <td>{j.status}</td>
                <td>{j.device}</td>
                <td>{typeof j.metrics_json?.elapsed_seconds === "number" ? j.metrics_json.elapsed_seconds.toFixed(1) : "-"}</td>
                <td>
                  {(j.status === "pending" || j.status === "running") && (
                    <>
                      <button type="button" className="secondary" onClick={() => pauseJob(j.id)}>
                        Pause
                      </button>{" "}
                      <button type="button" className="secondary" onClick={() => cancelJob(j.id)}>
                        Cancel
                      </button>
                    </>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}