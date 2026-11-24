from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.config import settings
from app.database import SessionLocal
from app.models import JobStatus, TrainingJob
from app.services.hardware import resource_snapshot
from app.services.ultralytics_trainer import training_queue

logger = logging.getLogger(__name__)
router = APIRouter()

WS_INTERVAL_S = 1.0
_connections: set[WebSocket] = set()


def _snapshot_payload() -> dict[str, Any]:
    with SessionLocal() as db:
        active = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.running.value).count()
        queued = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.pending.value).count()
        paused = db.query(TrainingJob).filter(TrainingJob.status == JobStatus.paused.value).count()
    snap = resource_snapshot(active_jobs=active, queued_jobs=queued)
    payload = snap.model_dump()
    payload["paused_jobs"] = paused
    payload["active_job_id"] = training_queue.active_job_id
    payload["server_time"] = time.time()
    payload["heartbeat"] = True
    payload["app"] = settings.app_name
    return payload


async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


async def _sender(websocket: WebSocket, stop: asyncio.Event) -> None:
    try:
        while not stop.is_set():
            try:
                await _send_json(websocket, _snapshot_payload())
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("ws send failed: %s", exc)
                return
            try:
                await asyncio.wait_for(stop.wait(), timeout=WS_INTERVAL_S)
            except TimeoutError:
                continue
    except asyncio.CancelledError:
        return
    finally:
        stop.set()


async def _receiver(websocket: WebSocket, stop: asyncio.Event) -> None:
    try:
        while not stop.is_set():
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                return
            except RuntimeError:
                return
            except asyncio.CancelledError:
                return
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                return
            text = message.get("text")
            if isinstance(text, str) and text.strip().lower() == "ping":
                try:
                    await _send_json(websocket, {"type": "pong", "server_time": time.time()})
                except Exception:  # noqa: BLE001
                    return
    finally:
        stop.set()


async def _close_ws(websocket: WebSocket) -> None:
    if websocket.client_state != WebSocketState.CONNECTED:
        return
    with contextlib.suppress(Exception):
        await websocket.close()


@router.websocket("/ws/resources")
async def resources_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    _connections.add(websocket)
    stop = asyncio.Event()
    send_task = asyncio.create_task(_sender(websocket, stop), name="ws-resources-send")
    recv_task = asyncio.create_task(_receiver(websocket, stop), name="ws-resources-recv")
    try:
        await stop.wait()
    except asyncio.CancelledError:
        stop.set()
    finally:
        stop.set()
        _connections.discard(websocket)
        await _close_ws(websocket)
        for task in (send_task, recv_task):
            if not task.done():
                task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.gather(send_task, recv_task, return_exceptions=True)


def active_connection_count() -> int:
    return len(_connections)
