from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import JobStatus, TrainingJob
from app.services.hardware import resource_snapshot
from app.services.ultralytics_trainer import training_queue

router = APIRouter()


@router.websocket("/ws/resources")
async def resources_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            with SessionLocal() as db:
                active = (
                    db.query(TrainingJob).filter(TrainingJob.status == JobStatus.running.value).count()
                )
                queued = (
                    db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).count()
                )
            snap = resource_snapshot(active_jobs=active, queued_jobs=queued)
            payload = snap.model_dump()
            payload["active_job_id"] = training_queue.active_job_id
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return