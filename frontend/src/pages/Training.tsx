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

  async function refresh() {
    const [ds, js] = await Promise.all([
      apiGet<DatasetRow[]>("/datasets"),
      apiGet<JobRow[]>("/training/jobs"),
    ]);
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

  return (
    <div>
      <h2>Training</h2>
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
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}