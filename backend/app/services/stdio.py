from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TextIO


class TeeStream(io.TextIOBase):
    """Write to several text streams. Not a process-wide singleton."""

    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:  # type: ignore[override]
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:  # type: ignore[override]
        for stream in self._streams:
            stream.flush()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._streams[0], "encoding", "utf-8") or "utf-8"

    def isatty(self) -> bool:  # type: ignore[override]
        return False


@contextmanager
def redirect_stdio_to(*streams: TextIO) -> Iterator[None]:
    """Temporarily tee stdout/stderr. Restores even if Ultralytics rebinds them."""
    tee = TeeStream(*streams)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
