from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models import Dataset
from app.schemas import (
    ClassDistribution,
    DatasetCreateResponse,
    DatasetOut,
    DatasetStats,
    DatasetValidation,
    SampleAnnotation,
    SplitConfig,
    SplitRequest,
)
from app.services.dataset_annotations import list_sample_images, load_annotations
from app.services.dataset_split import apply_split
from app.services.dataset_validator import (
    discover_yaml_in_tree,
    import_dataset_from_yaml,
    scan_yolo_folder,
    validate_yolo_dataset,
)
from app.services.security import (
    UnsafePathError,
    enforce_upload_limit,
    safe_extract_zip,
    sanitize_filename,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _persist_dataset(
    db: Session,
    name: str,
    yaml_path: Path,
    stats,
    validation,
    split_config: SplitConfig | None = None,
) -> Dataset:
    existing = db.query(Dataset).filter(Dataset.name == name).first()
    if existing:
        raise HTTPException(status_code=409, detail="dataset name already exists")

    row = Dataset(
        name=name,
        yaml_path=str(yaml_path.resolve()),
        root_dir=str(yaml_path.resolve().parent),
        stats_json=stats.model_dump(),
        validation_json=validation.model_dump(),
        split_config_json=split_config.model_dump() if split_config else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _to_out(ds: Dataset) -> DatasetOut:
    return DatasetOut(
        id=ds.id,
        name=ds.name,
        yaml_path=ds.yaml_path,
        root_dir=ds.root_dir,
        stats=DatasetStats(**ds.stats_json) if ds.stats_json else None,
        validation=DatasetValidation(**ds.validation_json) if ds.validation_json else None,
        split_config=SplitConfig(**ds.split_config_json) if ds.split_config_json else None,
        created_at=ds.created_at,
    )


def _refresh_dataset(db: Session, ds: Dataset) -> tuple[DatasetStats, DatasetValidation]:
    stats, validation = validate_yolo_dataset(Path(ds.yaml_path))
    ds.stats_json = stats.model_dump()
    ds.validation_json = validation.model_dump()
    db.commit()
    db.refresh(ds)
    return stats, validation


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)) -> list[DatasetOut]:
    return [_to_out(d) for d in db.query(Dataset).order_by(Dataset.created_at.desc()).all()]


@router.get("/{dataset_id}", response_model=DatasetOut)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)) -> DatasetOut:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    return _to_out(ds)


@router.post("/import-yaml", response_model=DatasetCreateResponse)
def import_yaml(path: str, name: str | None = None, db: Session = Depends(get_db)) -> DatasetCreateResponse:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail="yaml path not found")
    ds_name, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name=name)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.model_dump())
    row = _persist_dataset(db, ds_name, resolved, stats, validation)
    return DatasetCreateResponse(id=row.id, name=row.name, yaml_path=row.yaml_path, stats=stats, validation=validation)


@router.post("/scan-folder", response_model=DatasetCreateResponse)
def scan_folder(path: str, name: str | None = None, db: Session = Depends(get_db)) -> DatasetCreateResponse:
    folder = Path(path)
    yaml_path, notes = scan_yolo_folder(folder)
    if yaml_path is None:
        raise HTTPException(status_code=422, detail={"notes": notes})
    ds_name = name or folder.name
    _, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name=ds_name)
    if not validation.valid:
        raise HTTPException(status_code=422, detail={"validation": validation.model_dump(), "notes": notes})
    row = _persist_dataset(db, ds_name, resolved, stats, validation)
    return DatasetCreateResponse(id=row.id, name=row.name, yaml_path=row.yaml_path, stats=stats, validation=validation)


@router.post("/upload-zip", response_model=DatasetCreateResponse)
async def upload_zip(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: Session = Depends(get_db),
) -> DatasetCreateResponse:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="expected a .zip archive")
    try:
        zip_name = sanitize_filename(file.filename)
        ds_key = sanitize_filename(name)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    target = settings.uploads_dir / f"{ds_key}_{zip_name}"
    written = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with target.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            try:
                enforce_upload_limit(written)
            except UnsafePathError as exc:
                out.close()
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            out.write(chunk)
    _ = max_bytes

    extract_dir = settings.datasets_dir / ds_key
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(target) as zf:
            if zf.testzip() is not None:
                raise HTTPException(status_code=400, detail="corrupt zip archive")
            safe_extract_zip(zf, extract_dir)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="invalid zip archive") from exc
    except UnsafePathError as exc:
        shutil.rmtree(extract_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    yaml_path = discover_yaml_in_tree(extract_dir)
    if not yaml_path and (extract_dir / "coco128").exists():
        yaml_path = discover_yaml_in_tree(extract_dir / "coco128")
    if not yaml_path:
        raise HTTPException(status_code=422, detail="no data.yaml found in archive")

    ds_name, resolved, stats, validation = import_dataset_from_yaml(yaml_path, name=ds_key)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.model_dump())

    row = _persist_dataset(db, ds_name, resolved, stats, validation)
    return DatasetCreateResponse(id=row.id, name=row.name, yaml_path=row.yaml_path, stats=stats, validation=validation)


@router.post("/{dataset_id}/revalidate", response_model=DatasetOut)
def revalidate(dataset_id: int, db: Session = Depends(get_db)) -> DatasetOut:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    _refresh_dataset(db, ds)
    return _to_out(ds)


@router.get("/{dataset_id}/report", response_model=DatasetValidation)
def validation_report(dataset_id: int, db: Session = Depends(get_db)) -> DatasetValidation:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    if not ds.validation_json:
        _, validation = validate_yolo_dataset(Path(ds.yaml_path))
        return validation
    return DatasetValidation(**ds.validation_json)


@router.get("/{dataset_id}/class-distribution", response_model=ClassDistribution)
def class_distribution(dataset_id: int, db: Session = Depends(get_db)) -> ClassDistribution:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    stats = DatasetStats(**(ds.stats_json or {}))
    return ClassDistribution(
        class_names=stats.class_names,
        train=stats.class_counts_train,
        val=stats.class_counts_val,
        total=_merge_counts(stats.class_counts_train, stats.class_counts_val),
    )


@router.post("/{dataset_id}/split", response_model=DatasetOut)
def split_dataset(dataset_id: int, payload: SplitRequest, db: Session = Depends(get_db)) -> DatasetOut:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    cfg = SplitConfig(
        mode=payload.mode,
        val_ratio=payload.val_ratio,
        seed=payload.seed,
        manual_val_files=payload.manual_val_files,
    )
    apply_split(Path(ds.yaml_path), cfg)
    ds.split_config_json = cfg.model_dump()
    stats, validation = _refresh_dataset(db, ds)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.model_dump())
    return _to_out(ds)


@router.get("/{dataset_id}/samples")
def samples(
    dataset_id: int,
    split: str = Query(default="train"),
    limit: int = Query(default=24, ge=1, le=100),
    db: Session = Depends(get_db),
):
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    items = list_sample_images(Path(ds.yaml_path), split=split, limit=limit)
    return {"dataset_id": dataset_id, "items": items}


@router.get("/{dataset_id}/samples/{file_name}/annotations", response_model=SampleAnnotation)
def sample_annotations(
    dataset_id: int,
    file_name: str,
    split: str = Query(default="train"),
    db: Session = Depends(get_db),
) -> SampleAnnotation:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    try:
        file_name = sanitize_filename(file_name)
        ann = load_annotations(Path(ds.yaml_path), file_name, split)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    ann.dataset_id = dataset_id
    return ann


@router.get("/{dataset_id}/samples/{file_name}/image")
def sample_image(
    dataset_id: int,
    file_name: str,
    split: str = Query(default="train"),
    db: Session = Depends(get_db),
) -> FileResponse:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="dataset not found")
    try:
        file_name = sanitize_filename(file_name)
        ann = load_annotations(Path(ds.yaml_path), file_name, split)
    except UnsafePathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    root = Path(ds.root_dir or Path(ds.yaml_path).parent)
    with Path(ds.yaml_path).open(encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f) or {}
    rel = cfg.get("train" if split == "train" else "val", f"images/{split}")
    from app.services.dataset_validator import _resolve_path
    from app.services.security import is_within

    img_dir = _resolve_path(Path(cfg.get("path", root)), str(rel))
    img_path = (img_dir / file_name).resolve()
    if not img_path.exists() or not is_within(img_path, img_dir.resolve()):
        raise HTTPException(status_code=404, detail="image not found")
    _ = ann
    return FileResponse(img_path)