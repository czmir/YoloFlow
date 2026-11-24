from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def export_to_onnx(weights: Path, output_path: Path) -> Path | None:
    if not weights.exists():
        return None
    model = YOLO(str(weights))
    out = model.export(format="onnx", simplify=False, opset=12)
    out_path = Path(out)
    if out_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.replace(output_path)
        return output_path
    return out_path