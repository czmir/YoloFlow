from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/deploy", tags=["deploy"])


@router.get("/status")
def deploy_status() -> dict:
    """Skeleton deployment service — ONNX Runtime serving hooks land in YF-7."""
    return {
        "service": "yoloflow-inference",
        "status": "skeleton",
        "message": "Main API inference at /api/v1/inference/*; standalone ORT at :8001 (docker compose --profile ort).",
        "endpoints": {
            "health": "/api/v1/deploy/status",
            "infer": "/api/v1/inference/detect",
            "ort_health": "http://127.0.0.1:8001/health",
            "ort_detect": "http://127.0.0.1:8001/v1/detect",
        },
    }