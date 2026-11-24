import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { apiGet, apiPost } from "../api/client";

interface ValidationIssue {
  severity: string;
  code: string;
  message: string;
}

interface DatasetRow {
  id: number;
  name: string;
  yaml_path: string;
  stats?: {
    num_images_train: number;
    num_images_val: number;
    class_names: string[];
    class_counts_total?: Record<string, number>;
  };
  validation?: { valid: boolean; errors: string[]; warnings: string[]; issues?: ValidationIssue[] };
}

export default function DatasetsPage() {
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [zipName, setZipName] = useState("");
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [selectedId, setSelectedId] = useState<number | "">("");
  const [valRatio, setValRatio] = useState(0.2);
  const [report, setReport] = useState<DatasetRow["validation"] | null>(null);
  const [distribution, setDistribution] = useState<Record<string, number>>({});
  const [msg, setMsg] = useState<string | null>(null);

  const selected = rows.find((r) => r.id === selectedId);

  async function refresh() {
    setRows(await apiGet<DatasetRow[]>("/datasets"));
  }

  useEffect(() => {
    refresh().catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setReport(null);
      setDistribution({});
      return;
    }
    apiGet<DatasetRow["validation"]>(`/datasets/${selectedId}/report`).then(setReport).catch(console.error);
    apiGet<{ total: Record<string, number> }>(`/datasets/${selectedId}/class-distribution`)
      .then((d) => setDistribution(d.total))
      .catch(console.error);
  }, [selectedId, rows]);

  async function importYaml(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    try {
      const q = new URLSearchParams({ path });
      if (name) q.set("name", name);
      const res = await fetch(`/api/v1/datasets/import-yaml?${q}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setMsg("imported from yaml");
      await refresh();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    }
  }

  async function scanFolder() {
    setMsg(null);
    try {
      const q = new URLSearchParams({ path });
      if (name) q.set("name", name);
      const res = await fetch(`/api/v1/datasets/scan-folder?${q}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setMsg("imported from folder scan");
      await refresh();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    }
  }

  async function uploadZip(e: React.FormEvent) {
    e.preventDefault();
    if (!zipFile || !zipName) return;
    setMsg(null);
    const form = new FormData();
    form.append("file", zipFile);
    form.append("name", zipName);
    try {
      const res = await fetch("/api/v1/datasets/upload-zip", { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      setMsg("uploaded zip");
      await refresh();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    }
  }

  async function runSplit() {
    if (!selectedId) return;
    setMsg(null);
    try {
      await apiPost(`/datasets/${selectedId}/split`, { mode: "ratio", val_ratio: valRatio, seed: 42 });
      setMsg("split applied");
      await refresh();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    }
  }

  async function revalidate() {
    if (!selectedId) return;
    await fetch(`/api/v1/datasets/${selectedId}/revalidate`, { method: "POST" });
    await refresh();
    setMsg("revalidated");
  }

  return (
    <div>
      <h2>Datasets</h2>
      <div className="card">
        <h3>Import</h3>
        <form onSubmit={importYaml}>
          <p>
            <input placeholder="path to data.yaml or dataset root" value={path} onChange={(e) => setPath(e.target.value)} style={{ width: "55%" }} />
            <input placeholder="name" value={name} onChange={(e) => setName(e.target.value)} />
            <button type="submit">Import YAML</button>
            <button type="button" className="secondary" onClick={() => scanFolder()} style={{ marginLeft: 8 }}>
              Scan folder
            </button>
          </p>
        </form>
        <form onSubmit={uploadZip}>
          <p>
            <input placeholder="dataset name" value={zipName} onChange={(e) => setZipName(e.target.value)} />
            <input type="file" accept=".zip" onChange={(e) => setZipFile(e.target.files?.[0] ?? null)} />
            <button type="submit">Upload ZIP</button>
          </p>
        </form>
        {msg && <p className="mono">{msg}</p>}
      </div>

      <div className="card">
        <table>
          <thead>
            <tr>
              <th></th>
              <th>ID</th>
              <th>Name</th>
              <th>Train</th>
              <th>Val</th>
              <th>Valid</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>
                  <input type="radio" checked={selectedId === r.id} onChange={() => setSelectedId(r.id)} />
                </td>
                <td>{r.id}</td>
                <td>{r.name}</td>
                <td>{r.stats?.num_images_train ?? "-"}</td>
                <td>{r.stats?.num_images_val ?? "-"}</td>
                <td>{r.validation?.valid ? "yes" : "no"}</td>
                <td>
                  <Link to={`/datasets/${r.id}/annotations`}>Annotate</Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="card">
          <h3>{selected.name} — validation & split</h3>
          <p>
            <button type="button" onClick={revalidate}>
              Revalidate
            </button>
            <label style={{ marginLeft: 16 }}>
              val ratio{" "}
              <input
                type="number"
                min={0.05}
                max={0.5}
                step={0.05}
                value={valRatio}
                onChange={(e) => setValRatio(Number(e.target.value))}
                style={{ width: 64 }}
              />
            </label>
            <button type="button" onClick={runSplit} style={{ marginLeft: 8 }}>
              Apply split
            </button>
          </p>
          {report && (
            <>
              <p className="mono">errors: {report.errors.length} | warnings: {report.warnings?.length ?? 0}</p>
              <ul className="mono" style={{ maxHeight: 160, overflow: "auto" }}>
                {(report.issues ?? []).map((i, idx) => (
                  <li key={idx}>
                    [{i.severity}] {i.code}: {i.message}
                  </li>
                ))}
              </ul>
            </>
          )}
          {Object.keys(distribution).length > 0 && (
            <table>
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Labels (total)</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(distribution).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td>{v}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}