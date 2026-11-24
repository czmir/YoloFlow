from __future__ import annotations

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, WebSocketException

from app.config import settings
from app.database import SessionLocal
from app.models import JobStatus, TrainingJob
from app.services.hardware import resource_snapshot
from app.services.ultralytics_trainer import training_queue

logger = logging.getLogger(__name__)
router = APIRouter()

WS_INTERVAL_S = 1.0
MAX_BACKOFF_S = 30.0


def _snapshot_payload() -> dict:
    with SessionLocal() as db:
        active = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.running.value).count()
        queued = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).count()
        paused = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.paused.value).count()
    snap = resource_snapshot(active_jobs=active, queued_jobs=queued)
    payload = snap.model_dump()
    payload["paused_jobs"] = paused
    payload["active_job_id"] = training_queue.active_job_id
    payload["server_time"] = time.time()
    payload["app"] = settings.app_name
    return payload


async def _sender(websocket: WebSocket, stop: asyncio.Event) -> None:
    while not stop.is_set():
        payload = _snapshot_payload()
        try:
            await websocket.send_text(json.dumps(payload))
        except (WebSocketDisconnect, WebSocketException):
            stop.set()
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("ws send failed: %s", exc)
            stop.set()
            return
        await asyncio.sleep(WS_INTERVAL_S)


async def _receiver(websocket: WebSocket, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            msg = await websocket.receive_text()
        except (WebSocketDisconnect, WebSocketException):
            stop.set()
            return
        if msg.strip().lower() == "ping":
            try:
                await websocket.send_text(json.dumps({"type": "pong", "server_time": time.time()}))
            except Exception:  # noqa: BLE001
                stop.set()
                return


@router.websocket("/ws/resources")
async def resources_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    stop = asyncio.Event()
    send_task = asyncio.create_task(_sender(websocket, stop))
    recv_task = asyncio.create_task(_receiver(websocket, stop))
    try:
        await stop.wait()
    finally:
        stop.set()
        send_task.cancel()
        recv_task.cancel()
        await asyncio.gather(send_task, recv_task, return_exceptions=True)