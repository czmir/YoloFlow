#!/usr/bin/env python3
"""Redact secrets, emails, and machine-local usernames from acceptance artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"x-access-token:[^@\s\"']+", re.I), "x-access-token:<redacted>"),
    (re.compile(r"\b(ghp|ghs|github_pat)_[A-Za-z0-9_]+"), "<redacted-token>"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-]+", re.I), "Bearer <redacted>"),
    (re.compile(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", re.I), "<redacted-email>"),
    (re.compile(r"/home/[^/\s\"']+"), "/home/<user>"),
    (re.compile(r"/Users/[^/\s\"']+"), "/Users/<user>"),
    (re.compile(r"[A-Za-z]:\\Users\\[^\\s\"']+", re.I), r"C:\\Users\\<user>"),
]


def sanitize_text(text: str, repo_root: Path | None = None) -> str:
    root = str((repo_root or ROOT).resolve())
    text = text.replace(root, "<repo>")
    text = text.replace(root.replace("\\", "/"), "<repo>")
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


def sanitize_json_obj(obj, repo_root: Path | None = None):
    if isinstance(obj, str):
        return sanitize_text(obj, repo_root)
    if isinstance(obj, list):
        return [sanitize_json_obj(x, repo_root) for x in obj]
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v, repo_root) for k, v in obj.items()}
    return obj


def write_json(path: Path, obj, repo_root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = sanitize_json_obj(obj, repo_root)
    path.write_text(json.dumps(clean, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str, repo_root: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sanitize_text(text, repo_root), encoding="utf-8")
