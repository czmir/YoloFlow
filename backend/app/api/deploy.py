from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/deploy", tags=["deploy"])


@router.get("/status")
def deploy_status() -> dict:
    """Skeleton deployment service — ONNX Runtime serving hooks land in YF-7."""
    return {
        "service": "yoloflow-inference",
        "status": "skeleton",
        "message": "Use POST /api/v1/inference/detect for now; standalone ORT service in YF-7.",
        "endpoints": {
            "health": "/api/v1/deploy/status",
            "infer": "/api/v1/inference/detect",
        },
    }