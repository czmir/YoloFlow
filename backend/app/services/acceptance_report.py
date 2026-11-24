from __future__ import annotations


def overall_exit_code(report: dict) -> int:
    """Train success is not overall success if ONNX compare was skipped."""
    compare = report.get("onnx_compare") or {}
    if compare.get("skipped"):
        return 3
    job = report.get("job") or {}
    if job.get("status") != "succeeded":
        return 2
    return 0
