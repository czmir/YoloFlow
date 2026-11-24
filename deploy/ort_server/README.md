# YoloFlow ORT inference service (YF-7)

Lightweight FastAPI wrapper around an Ultralytics-loadable `.onnx` model.

## Run with Docker Compose profile

```bash
# Point at an exported ONNX from a training job
export YOLOFLOW_ORT_MODEL=/app/data/workspaces/job_1/model.onnx
docker compose --profile ort up ort-inference
```

## curl examples

```bash
curl -s http://127.0.0.1:8001/health | jq

curl -s -X POST http://127.0.0.1:8001/v1/detect \
  -F "file=@/path/to/image.jpg" \
  -F "conf=0.25" | jq
```

## Local (no Docker)

```bash
export YOLOFLOW_ORT_MODEL=./data/workspaces/job_1/model.onnx
pip install fastapi uvicorn ultralytics onnxruntime
python deploy/ort_server/main.py
```