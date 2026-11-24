import { useEffect, useState } from "react";
import type { ResourceSnapshot } from "../../../shared/typescript/types";

export default function ResourcesPage() {
  const [snap, setSnap] = useState<ResourceSnapshot | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/resources`);
    ws.onmessage = (ev) => setSnap(JSON.parse(ev.data));
    return () => ws.close();
  }, []);

  return (
    <div>
      <h2>Resource monitor</h2>
      <div className="card mono">
        {!snap && <p>connecting...</p>}
        {snap && (
          <ul>
            <li>CPU: {snap.cpu_percent.toFixed(1)}%</li>
            <li>Memory: {snap.memory_percent.toFixed(1)}% ({snap.memory_used_mb.toFixed(0)} MB)</li>
            <li>Active jobs: {snap.active_jobs}</li>
            <li>Queued jobs: {snap.queued_jobs}</li>
            <li>Active job id: {snap.active_job_id ?? "-"}</li>
          </ul>
        )}
      </div>
    </div>
  );
}