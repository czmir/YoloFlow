from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import TrainingJob

router = APIRouter(prefix="/training", tags=["logs"])


@router.get("/jobs/{job_id}/logs")
def read_job_logs(
    job_id: int,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=8000, ge=1, le=500_000),
    db: Session = Depends(get_db),
) -> dict:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    log_path = job.log_path
    if not log_path or not Path(log_path).exists():
        return {
            "job_id": job_id,
            "content": "",
            "offset": offset,
            "size": 0,
            "next_offset": offset,
            "complete": True,
        }
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    size = len(text)
    chunk = text[offset : offset + limit]
    next_off = offset + len(chunk)
    return {
        "job_id": job_id,
        "content": chunk,
        "offset": offset,
        "size": size,
        "next_offset": next_off,
        "complete": next_off >= size,
    }


@router.get("/jobs/{job_id}/logs/stream")
async def stream_job_logs(
    job_id: int,
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    job = db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    log_path = Path(job.log_path or "")

    async def generate():
        pos = offset
        while True:
            with SessionLocal() as session:
                row = session.get(TrainingJob, job_id)
                status = row.status if row else "unknown"
            if not log_path.exists():
                yield "data: [waiting for log file]\n\n"
                await asyncio.sleep(1.0)
                continue
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > pos:
                chunk = text[pos:]
                pos = len(text)
                for line in chunk.splitlines(keepends=True):
                    yield f"data: {line.rstrip()}\n\n"
            await asyncio.sleep(1.0)
            if status in {"succeeded", "failed", "cancelled"} and pos >= len(text):
                yield "event: done\ndata: eof\n\n"
                break

    return StreamingResponse(generate(), media_type="text/event-stream")