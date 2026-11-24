from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Dataset
from app.schemas import DatasetCreateResponse, DatasetOut
from app.services.dataset_validator import import_dataset_from_yaml

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)) -> list[Dataset]:
    return db.query(Dataset).order_by(Dataset.created_at.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    return ds


@router.post("/import-yaml", response_model=DatasetCreateResponse)
def import_yaml(path: str, name: str | None = None, db: Session = Depends(get_db)) -> DatasetCreateResponse:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="yaml path not found")
    ds_name, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name=name)
    existing = db.query(Dataset).filter(Dataset.name == ds_name).first()
    if existing:
        raise HTTPException(status_code=409, detail="dataset name already exists")
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.model_dump())

    row = Dataset(
        name=ds_name,
        yaml_path=str(resolved),
        root_dir=str(resolved.parent),
        stats_json=stats.model_dump(),
        validation_json=validation.model_dump(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return DatasetCreateResponse(
        id=row.id,
        name=row.name,
        yaml_path=row.yaml_path,
        stats=stats,
        validation=validation,
    )


@router.post("/upload-zip", response_model=DatasetCreateResponse)
async def upload_zip(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: Session = Depends(get_db),
) -> DatasetCreateResponse:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="expected a .zip archive")

    target = settings.uploads_dir / f"{name}_{file.filename}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    extract_dir = settings.datasets_dir / name
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target) as zf:
        zf.extractall(extract_dir)

    yaml_candidates = list(extract_dir.rglob("data.yaml")) + list(extract_dir.rglob("dataset.yaml"))
    if not yaml_candidates:
        raise HTTPException(status_code=422, detail="no data.yaml found in archive")

    ds_name, resolved, stats, validation = import_dataset_from_yaml(yaml_candidates[0], name=name)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.model_dump())

    row = Dataset(
        name=ds_name,
        yaml_path=str(resolved),
        root_dir=str(resolved.parent),
        stats_json=stats.model_dump(),
        validation_json=validation.model_dump(),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return DatasetCreateResponse(
        id=row.id,
        name=row.name,
        yaml_path=row.yaml_path,
        stats=stats,
        validation=validation,
    )