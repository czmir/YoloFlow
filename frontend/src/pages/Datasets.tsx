import { useEffect, useState } from "react";
import { apiGet } from "../api/client";

interface DatasetRow {
  id: number;
  name: string;
  yaml_path: string;
  stats?: {
    num_images_train: number;
    num_images_val: number;
    class_names: string[];
  };
  validation?: { valid: boolean; errors: string[] };
}

export default function DatasetsPage() {
  const [rows, setRows] = useState<DatasetRow[]>([]);
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  async function refresh() {
    setRows(await apiGet<DatasetRow[]>("/datasets"));
  }

  useEffect(() => {
    refresh().catch(console.error);
  }, []);

  async function importYaml(e: React.FormEvent) {
    e.preventDefault();
    setMsg(null);
    try {
      const q = new URLSearchParams({ path });
      if (name) q.set("name", name);
      const res = await fetch(`/api/v1/datasets/import-yaml?${q}`, { method: "POST" });
      if (!res.ok) throw new Error(await res.text());
      setMsg("imported");
      await refresh();
    } catch (err) {
      setMsg(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div>
      <h2>Datasets</h2>
      <div className="card">
        <form onSubmit={importYaml}>
          <p>
            <input placeholder="absolute path to data.yaml" value={path} onChange={(e) => setPath(e.target.value)} style={{ width: "60%" }} />
            <input placeholder="name (optional)" value={name} onChange={(e) => setName(e.target.value)} />
            <button type="submit">Import YAML</button>
          </p>
        </form>
        {msg && <p className="mono">{msg}</p>}
      </div>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Name</th>
              <th>Train imgs</th>
              <th>Val imgs</th>
              <th>Classes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.id}</td>
                <td>{r.name}</td>
                <td>{r.stats?.num_images_train ?? "-"}</td>
                <td>{r.stats?.num_images_val ?? "-"}</td>
                <td>{r.stats?.class_names?.join(", ") ?? "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}