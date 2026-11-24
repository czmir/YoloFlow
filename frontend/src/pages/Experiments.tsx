import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiGet, apiPost } from "../api/client";

interface ExperimentSummary {
  job_id: number;
  name: string;
  status: string;
  map50: number | null;
  map50_95: number | null;
  elapsed_seconds: number | null;
}

interface ExperimentSeries {
  job_id: number;
  name: string;
  epochs: number[];
  train_loss: (number | null)[];
  val_map50: (number | null)[];
  val_map50_95: (number | null)[];
}

interface ComparePayload {
  jobs: ExperimentSummary[];
  series: ExperimentSeries[];
  metric_table: Record<string, unknown>[];
}

const COLORS = ["#111827", "#2563eb", "#16a34a", "#dc2626"];

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [compare, setCompare] = useState<ComparePayload | null>(null);
  const [singleId, setSingleId] = useState<number | "">("");
  const [cmUrl, setCmUrl] = useState<string | null>(null);

  useEffect(() => {
    apiGet<ExperimentSummary[]>("/experiments").then(setExperiments).catch(console.error);
  }, []);

  useEffect(() => {
    if (selected.length < 2) {
      setCompare(null);
      return;
    }
    apiPost<ComparePayload>("/experiments/compare", { job_ids: selected })
      .then(setCompare)
      .catch(console.error);
  }, [selected]);

  useEffect(() => {
    if (singleId === "") {
      setCmUrl(null);
      return;
    }
    apiGet<{ available: boolean; image_url?: string }>(`/experiments/jobs/${singleId}/confusion-matrix`)
      .then((r) => setCmUrl(r.available && r.image_url ? r.image_url : null))
      .catch(() => setCmUrl(null));
  }, [singleId]);

  const lossChartData = useMemo(() => {
    if (!compare) return [];
    const maxEpoch = Math.max(...compare.series.flatMap((s) => s.epochs), 0);
    const rows: Record<string, number | null>[] = [];
    for (let e = 1; e <= maxEpoch; e++) {
      const row: Record<string, number | null> = { epoch: e };
      for (const s of compare.series) {
        const idx = s.epochs.indexOf(e);
        row[`loss_${s.job_id}`] = idx >= 0 ? s.train_loss[idx] : null;
        row[`map50_${s.job_id}`] = idx >= 0 ? s.val_map50[idx] : null;
        row[`map95_${s.job_id}`] = idx >= 0 ? s.val_map50_95[idx] : null;
      }
      rows.push(row);
    }
    return rows;
  }, [compare]);

  function toggle(id: number) {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  }

  return (
    <div>
      <h2>Experiments</h2>
      <div className="card">
        <p>Select 2–4 runs to compare</p>
        <table>
          <thead>
            <tr>
              <th></th>
              <th>ID</th>
              <th>Name</th>
              <th>Status</th>
              <th>mAP50</th>
              <th>mAP50-95</th>
              <th>Time(s)</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((e) => (
              <tr key={e.job_id}>
                <td>
                  <input type="checkbox" checked={selected.includes(e.job_id)} onChange={() => toggle(e.job_id)} />
                </td>
                <td>{e.job_id}</td>
                <td>{e.name}</td>
                <td>{e.status}</td>
                <td>{e.map50?.toFixed(3) ?? "-"}</td>
                <td>{e.map50_95?.toFixed(3) ?? "-"}</td>
                <td>{e.elapsed_seconds?.toFixed(1) ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {compare && compare.metric_table.length > 0 && (
        <div className="card">
          <h3>Metric table</h3>
          <table>
            <thead>
              <tr>
                <th>Job</th>
                <th>Dataset</th>
                <th>mAP50</th>
                <th>mAP50-95</th>
                <th>Elapsed</th>
              </tr>
            </thead>
            <tbody>
              {compare.metric_table.map((row) => (
                <tr key={String(row.job_id)}>
                  <td>
                    #{String(row.job_id)} {String(row.name)}
                  </td>
                  <td>{String(row.dataset ?? "-")}</td>
                  <td>{row.map50 != null ? Number(row.map50).toFixed(3) : "-"}</td>
                  <td>{row.map50_95 != null ? Number(row.map50_95).toFixed(3) : "-"}</td>
                  <td>{row.elapsed_seconds != null ? Number(row.elapsed_seconds).toFixed(1) : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {lossChartData.length > 0 && compare && (
        <>
          <div className="card" style={{ height: 320 }}>
            <h3>Train loss</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                {compare.series.map((s, i) => (
                  <Line
                    key={s.job_id}
                    type="monotone"
                    dataKey={`loss_${s.job_id}`}
                    name={`#${s.job_id} loss`}
                    stroke={COLORS[i % COLORS.length]}
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="card" style={{ height: 320 }}>
            <h3>Val mAP50 / mAP50-95</h3>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={lossChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="epoch" />
                <YAxis />
                <Tooltip />
                <Legend />
                {compare.series.map((s, i) => (
                  <Line
                    key={`m50-${s.job_id}`}
                    type="monotone"
                    dataKey={`map50_${s.job_id}`}
                    name={`#${s.job_id} mAP50`}
                    stroke={COLORS[i % COLORS.length]}
                    dot={false}
                  />
                ))}
                {compare.series.map((s, i) => (
                  <Line
                    key={`m95-${s.job_id}`}
                    type="monotone"
                    dataKey={`map95_${s.job_id}`}
                    name={`#${s.job_id} mAP50-95`}
                    stroke={COLORS[i % COLORS.length]}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      <div className="card">
        <h3>Confusion matrix</h3>
        <select value={singleId} onChange={(e) => setSingleId(e.target.value ? Number(e.target.value) : "")}>
          <option value="">Select a finished job</option>
          {experiments
            .filter((e) => e.status === "succeeded")
            .map((e) => (
              <option key={e.job_id} value={e.job_id}>
                #{e.job_id} {e.name}
              </option>
            ))}
        </select>
        {cmUrl && <img src={cmUrl} alt="confusion matrix" style={{ display: "block", maxWidth: "100%", marginTop: 12 }} />}
        {singleId !== "" && !cmUrl && <p className="mono">No confusion matrix artifact for this job yet (available after training).</p>}
      </div>
    </div>
  );
}