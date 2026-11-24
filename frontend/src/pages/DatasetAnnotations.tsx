import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { apiGet } from "../api/client";

interface SampleItem {
  file_name: string;
  split: string;
}

interface AnnotationBox {
  class_id: number;
  class_name: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface SampleAnnotation {
  file_name: string;
  image_width: number;
  image_height: number;
  boxes: AnnotationBox[];
}

export default function DatasetAnnotationsPage() {
  const { id } = useParams();
  const [search] = useSearchParams();
  const datasetId = Number(id);
  const [split, setSplit] = useState(search.get("split") || "train");
  const [items, setItems] = useState<SampleItem[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [ann, setAnn] = useState<SampleAnnotation | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!datasetId) return;
    apiGet<{ items: SampleItem[] }>(`/datasets/${datasetId}/samples?split=${split}&limit=32`)
      .then((r) => {
        setItems(r.items);
        if (r.items.length) setSelected(r.items[0].file_name);
      })
      .catch(console.error);
  }, [datasetId, split]);

  useEffect(() => {
    if (!datasetId || !selected) return;
    apiGet<SampleAnnotation>(`/datasets/${datasetId}/samples/${encodeURIComponent(selected)}/annotations?split=${split}`)
      .then(setAnn)
      .catch(console.error);
  }, [datasetId, selected, split]);

  useEffect(() => {
    if (!ann || !imgRef.current || !canvasRef.current) return;
    const img = imgRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const draw = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);
      ctx.lineWidth = 2;
      for (const b of ann.boxes) {
        ctx.strokeStyle = "#16a34a";
        ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);
        ctx.fillStyle = "#16a34a";
        ctx.font = "12px sans-serif";
        ctx.fillText(b.class_name, b.x1, Math.max(14, b.y1 - 2));
      }
    };
    if (img.complete) draw();
    else img.onload = draw;
  }, [ann]);

  const imageUrl =
    datasetId && selected
      ? `/api/v1/datasets/${datasetId}/samples/${encodeURIComponent(selected)}/image?split=${split}`
      : "";

  return (
    <div>
      <p>
        <Link to="/datasets">← Datasets</Link>
      </p>
      <h2>Annotations — dataset #{datasetId}</h2>
      <div className="card">
        <label>
          Split&nbsp;
          <select value={split} onChange={(e) => setSplit(e.target.value)}>
            <option value="train">train</option>
            <option value="val">val</option>
          </select>
        </label>
        <select value={selected} onChange={(e) => setSelected(e.target.value)} style={{ marginLeft: 8, minWidth: 200 }}>
          {items.map((it) => (
            <option key={it.file_name} value={it.file_name}>
              {it.file_name}
            </option>
          ))}
        </select>
      </div>
      {ann && (
        <div className="card">
          <p className="mono">
            {ann.file_name} — {ann.image_width}×{ann.image_height}, boxes={ann.boxes.length}
          </p>
          <img ref={imgRef} src={imageUrl} alt={selected} style={{ display: "none" }} />
          <canvas ref={canvasRef} style={{ maxWidth: "100%" }} />
        </div>
      )}
    </div>
  );
}