import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { apiGet } from "../api/client";

interface JobRow {
  id: number;
  name: string;
  status: string;
}

interface ChartPayload {
  series: {
    epochs: number[];
    train_loss: (number | null)[];
    val_map50: (number | null)[];
  };
}

export default function ExperimentsPage() {
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [jobId, setJobId] = useState<number | "">("");
  const [chart, setChart] = useState<ChartPayload | null>(null);

  useEffect(() => {
    apiGet<JobRow[]>("/training/jobs").then(setJobs).catch(console.error);
  }, []);

  useEffect(() => {
    if (jobId === "") return;
    apiGet<ChartPayload>(`/experiments/jobs/${jobId}/chart`).then(setChart).catch(console.error);
  }, [jobId]);

  const data =
    chart?.series.epochs.map((epoch, i) => ({
      epoch,
      train_loss: chart.series.train_loss[i],
      val_map50: chart.series.val_map50[i],
    })) ?? [];

  return (
    <div>
      <h2>Experiments</h2>
      <div className="card">
        <select value={jobId} onChange={(e) => setJobId(e.target.value ? Number(e.target.value) : "")}>
          <option value="">Select job</option>
          {jobs.map((j) => (
            <option key={j.id} value={j.id}>
              #{j.id} {j.name} ({j.status})
            </option>
          ))}
        </select>
      </div>
      {data.length > 0 && (
        <div className="card" style={{ height: 360 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="train_loss" stroke="#111" dot={false} />
              <Line type="monotone" dataKey="val_map50" stroke="#2563eb" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}