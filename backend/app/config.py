from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _REPO_ROOT / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="YOLOFLOW_")

    app_name: str = "YoloFlow"
    api_prefix: str = "/api/v1"
    database_url: str = f"sqlite:///{_DATA_ROOT / 'yoloflow.db'}"
    data_root: Path = _DATA_ROOT
    datasets_dir: Path = _DATA_ROOT / "datasets"
    workspaces_dir: Path = _DATA_ROOT / "workspaces"
    uploads_dir: Path = _DATA_ROOT / "uploads"
    max_upload_mb: int = 512
    training_queue_poll_seconds: float = 1.0
    default_model: str = "yolov8n.pt"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]

    def ensure_dirs(self) -> None:
        for path in (
            self.data_root,
            self.datasets_dir,
            self.workspaces_dir,
            self.uploads_dir,
            self.data_root / "runs",
            self.data_root / "models",
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()