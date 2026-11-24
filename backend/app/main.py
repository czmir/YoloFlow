from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, deploy, experiments, inference, training
from app.config import settings
from app.database import init_db
from app.ws import resources


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    init_db()
    yield


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
app.include_router(inference.router, prefix=api)
app.include_router(experiments.router, prefix=api)
app.include_router(deploy.router, prefix=api)
app.include_router(resources.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}