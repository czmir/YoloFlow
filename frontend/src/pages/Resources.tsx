import { useEffect, useRef, useState } from "react";
import type { ResourceSnapshot } from "../../../shared/typescript/types";

export default function ResourcesPage() {
  const [snap, setSnap] = useState<ResourceSnapshot | null>(null);
  const [status, setStatus] = useState<"connecting" | "open" | "closed">("connecting");
  const retryRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    function connect() {
      if (cancelled) return;
      setStatus("connecting");
      const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/resources`);
      wsRef.current = ws;
      ws.onopen = () => {
        retryRef.current = 0;
        setStatus("open");
        ws.send("ping");
      };
      ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.type === "pong") return;
          setSnap(data);
        } catch {
          /* ignore */
        }
      };
      ws.onclose = () => {
        setStatus("closed");
        const delay = Math.min(1000 * 2 ** retryRef.current, 15000);
        retryRef.current += 1;
        timer = setTimeout(connect, delay);
      };
      ws.onerror = () => ws.close();
    }

    connect();
    const heartbeat = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) wsRef.current.send("ping");
    }, 15000);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      clearInterval(heartbeat);
      wsRef.current?.close();
    };
  }, []);

  return (
    <div>
      <h2>Resource monitor</h2>
      <div className="card mono">
        <p>WebSocket: {status}</p>
        {!snap && <p>waiting for metrics…</p>}
        {snap && (
          <ul>
            <li>CPU: {snap.cpu_percent.toFixed(1)}%</li>
            <li>Memory: {snap.memory_percent.toFixed(1)}% ({snap.memory_used_mb.toFixed(0)} MB)</li>
            <li>Active jobs: {snap.active_jobs}</li>
            <li>Queued jobs: {snap.queued_jobs}</li>
            <li>Paused jobs: {snap.paused_jobs ?? 0}</li>
            <li>Active job id: {snap.active_job_id ?? "-"}</li>
          </ul>
        )}
      </div>
    </div>
  );
}