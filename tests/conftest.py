import uuid

import pytest
from sqlalchemy.orm import Session

from app.database import SessionLocal, init_db
from app.models import Dataset

init_db()


@pytest.fixture()
def db() -> Session:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def make_dataset(db: Session, tag: str = "t") -> Dataset:
    row = Dataset(
        name=f"ds-{tag}-{uuid.uuid4().hex[:10]}",
        yaml_path="/tmp/data.yaml",
        root_dir="/tmp",
        stats_json={"class_names": ["obj"]},
        validation_json={"valid": True, "errors": [], "warnings": []},
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row