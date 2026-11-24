from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

from app.config import settings

# Ultralytics hub checkpoints we allow by filename (downloaded by the library).
_BUILTIN_WEIGHTS = re.compile(
    r"^(?:yolov8|yolov9|yolov10|yolo11|yolo12)[nslmx](?:-[a-z0-9]+)?\.(?:pt|onnx)$",
    re.IGNORECASE,
)
_WEIGHT_SUFFIXES = {".pt", ".onnx"}
_UNSAFE_ZIP_PARTS = {"", ".", ".."}


class UnsafePathError(ValueError):
    """Path is outside the allowed data roots or otherwise rejected."""


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def data_roots() -> list[Path]:
    return [
        settings.data_root.resolve(),
        settings.datasets_dir.resolve(),
        settings.workspaces_dir.resolve(),
        settings.uploads_dir.resolve(),
        (settings.data_root / "models").resolve(),
    ]


def safe_join(root: Path, *parts: str) -> Path:
    """Join then require the result to stay under root (no traversal)."""
    root = root.resolve()
    candidate = root.joinpath(*parts)
    resolved = candidate.resolve()
    if not is_within(resolved, root) and resolved != root:
        raise UnsafePathError(f"path escapes root: {candidate}")
    return resolved


def sanitize_filename(name: str) -> str:
    base = Path(name).name
    if not base or base in _UNSAFE_ZIP_PARTS or "/" in name or "\\" in name:
        raise UnsafePathError("invalid file name")
    if base != name:
        raise UnsafePathError("path components not allowed in file name")
    return base


def is_builtin_weight_name(name: str) -> bool:
    return bool(_BUILTIN_WEIGHTS.match(Path(name).name))


def resolve_weight_path(spec: str | Path, *, must_exist: bool = False) -> Path:
    """Allow Ultralytics builtin names or files under the data root."""
    raw = Path(str(spec))
    if not raw.is_absolute() and is_builtin_weight_name(raw.name) and len(raw.parts) == 1:
        return raw
    resolved = raw.expanduser().resolve()
    if resolved.suffix.lower() not in _WEIGHT_SUFFIXES:
        raise UnsafePathError(f"unsupported weight type: {resolved.suffix}")
    if not any(is_within(resolved, root) for root in data_roots()):
        raise UnsafePathError("weight path is outside the data directory")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def zip_member_path(info: zipfile.ZipInfo) -> Path:
    name = info.filename.replace("\\", "/")
    if info.is_dir() or name.endswith("/"):
        rel = Path(name)
    else:
        rel = Path(name)
    if rel.is_absolute() or name.startswith("/") or name.startswith("\\"):
        raise UnsafePathError(f"absolute zip entry: {info.filename}")
    if ".." in rel.parts:
        raise UnsafePathError(f"zip path traversal: {info.filename}")
    if any(part in _UNSAFE_ZIP_PARTS or part.startswith("/") for part in rel.parts):
        raise UnsafePathError(f"unsafe zip entry: {info.filename}")
    return rel


def inspect_zip(archive: zipfile.ZipFile) -> tuple[int, int]:
    """Return (file_count, uncompressed_bytes) after rejecting traversal."""
    files = 0
    total = 0
    for info in archive.infolist():
        rel = zip_member_path(info)
        if info.is_dir() or str(rel).endswith("/"):
            continue
        files += 1
        total += max(0, int(info.file_size))
        if files > settings.max_zip_files:
            raise UnsafePathError(f"zip has more than {settings.max_zip_files} files")
        if total > settings.max_zip_uncompressed_mb * 1024 * 1024:
            raise UnsafePathError(f"zip uncompressed size exceeds {settings.max_zip_uncompressed_mb} MB")
    return files, total


def safe_extract_zip(archive: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    inspect_zip(archive)
    for info in archive.infolist():
        rel = zip_member_path(info)
        target = (dest / rel).resolve()
        if not is_within(target, dest) and target != dest:
            raise UnsafePathError(f"zip path traversal: {info.filename}")
        if info.is_dir() or info.filename.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as src, target.open("wb") as out:
            remaining = settings.max_zip_uncompressed_mb * 1024 * 1024
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                remaining -= len(chunk)
                if remaining < 0:
                    raise UnsafePathError("zip uncompressed size limit exceeded during extract")
                out.write(chunk)


def enforce_upload_limit(nbytes: int) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if nbytes > max_bytes:
        raise UnsafePathError(f"upload exceeds {settings.max_upload_mb} MB")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
