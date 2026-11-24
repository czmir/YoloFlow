from __future__ import annotations

import os
from contextlib import asynccontextmanager

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, deploy, experiments, inference, logs, models, training
from app.config import settings
from app.database import SessionLocal, init_db
from app.services.job_state import recover_orphan_jobs
from app.services.ultralytics_trainer import training_queue
from app.ws import resources


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    training_queue.enabled = settings.embedded_worker
    with SessionLocal() as db:
        recover_orphan_jobs(db)
    if settings.embedded_worker:
        training_queue.kick()
    yield
    training_queue.enabled = False


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = settings.api_prefix
app.include_router(datasets.router, prefix=api)
app.include_router(training.router, prefix=api)
app.include_router(logs.router, prefix=api)
app.include_router(inference.router, prefix=api)
app.include_router(models.router, prefix=api)
app.include_router(experiments.router, prefix=api)
app.include_router(deploy.router, prefix=api)
app.include_router(resources.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}