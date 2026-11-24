from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


def _sqlite_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        return url
    return f"sqlite:///{url}"


engine = create_engine(
    _sqlite_url(settings.database_url),
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _):  # type: ignore[no-untyped-def]
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Base(DeclarativeBase):
    pass


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    from app import models  # noqa: F401

    settings.ensure_dirs()
    db_path = settings.database_url.replace("sqlite:///", "")
    Path = __import__("pathlib").Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _migrate_sqlite()


def _migrate_sqlite() -> None:
    from sqlalchemy import text

    stmts = [
        "ALTER TABLE datasets ADD COLUMN split_config_json JSON",
        "ALTER TABLE training_jobs ADD COLUMN cancel_requested BOOLEAN DEFAULT 0",
        "ALTER TABLE training_jobs ADD COLUMN pause_requested BOOLEAN DEFAULT 0",
        "ALTER TABLE training_jobs ADD COLUMN parent_job_id INTEGER",
        "ALTER TABLE training_jobs ADD COLUMN claimed_by VARCHAR(128)",
        "ALTER TABLE training_jobs ADD COLUMN heartbeat_at DATETIME",
    ]
    # model_versions / onnx_export_records created via create_all on fresh DB
    with engine.begin() as conn:
        for stmt in stmts:
            try:
                conn.execute(text(stmt))
            except Exception:  # noqa: BLE001
                pass