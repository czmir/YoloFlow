import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.security import (
    UnsafePathError,
    inspect_zip,
    resolve_weight_path,
    safe_extract_zip,
    sanitize_filename,
)

client = TestClient(app)


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_zip_path_traversal_rejected(tmp_path: Path):
    payload = _zip_bytes({"../../evil.txt": b"nope", "ok/a.txt": b"x"})
    dest = tmp_path / "out"
    dest.mkdir()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with pytest.raises(UnsafePathError, match="traversal"):
            inspect_zip(zf)
        with pytest.raises(UnsafePathError):
            safe_extract_zip(zf, dest)
    assert not (tmp_path / "evil.txt").exists()


def test_zip_absolute_entry_rejected(tmp_path: Path):
    payload = _zip_bytes({"/tmp/abs.txt": b"x"})
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        with pytest.raises(UnsafePathError):
            inspect_zip(zf)


def test_safe_extract_keeps_members_inside(tmp_path: Path):
    payload = _zip_bytes({"images/a.txt": b"hi", "labels/a.txt": b"0 0.5 0.5 0.1 0.1"})
    dest = tmp_path / "ds"
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        safe_extract_zip(zf, dest)
    assert (dest / "images" / "a.txt").read_text() == "hi"
    assert not (tmp_path / "a.txt").exists()


def test_upload_zip_traversal_http():
    payload = _zip_bytes({"../escape.txt": b"x"})
    r = client.post(
        "/api/v1/datasets/upload-zip",
        files={"file": ("bad.zip", payload, "application/zip")},
        data={"name": "zip-trav"},
    )
    assert r.status_code == 400
    assert "traversal" in r.text.lower() or "unsafe" in r.text.lower()


def test_weight_path_rejects_escape(tmp_path: Path):
    outside = tmp_path / "hack.pt"
    outside.write_bytes(b"x")
    with pytest.raises(UnsafePathError):
        resolve_weight_path(outside)


def test_builtin_weight_name_allowed():
    path = resolve_weight_path("yolov8n.pt")
    assert path.name == "yolov8n.pt"


def test_weight_under_data_root_ok(tmp_path: Path, monkeypatch):
    models = settings.data_root / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / "custom.pt"
    target.write_bytes(b"not-a-real-checkpoint")
    resolved = resolve_weight_path(target)
    assert resolved == target.resolve()


def test_sanitize_filename():
    assert sanitize_filename("a.jpg") == "a.jpg"
    with pytest.raises(UnsafePathError):
        sanitize_filename("../etc/passwd")
    with pytest.raises(UnsafePathError):
        sanitize_filename("a/b.jpg")


def test_sample_image_rejects_traversal():
    r = client.get("/api/v1/datasets/1/samples/..%2F..%2Fetc%2Fpasswd/image")
    assert r.status_code in {400, 404}
