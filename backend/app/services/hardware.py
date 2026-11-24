from __future__ import annotations

import os

import psutil
import torch

from app.schemas import HardwareInfo


def detect_device() -> HardwareInfo:
    cuda = torch.cuda.is_available()
    device = "cuda:0" if cuda else "cpu"
    name = torch.cuda.get_device_name(0) if cuda else None
    return HardwareInfo(
        device=device,
        cuda_available=cuda,
        cuda_device_name=name,
        cpu_count=os.cpu_count() or 1,
    )


def resource_snapshot(active_jobs: int, queued_jobs: int):
    from app.schemas import ResourceSnapshot
    import time

    mem = psutil.virtual_memory()
    return ResourceSnapshot(
        timestamp=time.time(),
        cpu_percent=psutil.cpu_percent(interval=None),
        memory_percent=mem.percent,
        memory_used_mb=mem.used / (1024 * 1024),
        active_jobs=active_jobs,
        queued_jobs=queued_jobs,
    )