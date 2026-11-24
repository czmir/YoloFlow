"""Standalone ONNX Runtime inference microservice (YF-7)."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from ultralytics import YOLO

MODEL_PATH = os.getenv("YOLOFLOW_ORT_MODEL", "yolov8n.onnx")

app = FastAPI(title="YoloFlow ORT Inference")


@app.on_event("startup")
def _load():
    if Path(MODEL_PATH).exists():
        app.state.model = YOLO(MODEL_PATH)
    else:
        app.state.model = None


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "loaded": app.state.model is not None}


@app.post("/v1/detect")
async def detect(file: UploadFile = File(...), conf: float = Form(0.25)):
    if app.state.model is None:
        return {"error": "model not loaded", "path": MODEL_PATH}
    suffix = Path(file.filename or "img.jpg").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        path = Path(tmp.name)
    try:
        results = app.state.model.predict(source=str(path), conf=conf, verbose=False)
        dets = []
        names = app.state.model.names
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                dets.append(
                    {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, str(cls_id)),
                        "confidence": float(box.conf[0]),
                        "xyxy": [float(x) for x in box.xyxy[0].tolist()],
                    }
                )
        return {"detections": dets, "count": len(dets)}
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8001")))