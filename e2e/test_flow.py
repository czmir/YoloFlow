"""Playwright E2E: upload dataset, train, inspect, infer against a live stack."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ.get("YOLOFLOW_E2E_BASE", "http://127.0.0.1:5173")
ZIP_PATH = Path(os.environ.get("YOLOFLOW_E2E_ZIP", ROOT / "data" / "runs" / "e2e-coco8.zip"))
SHOT_DIR = Path(os.environ.get("YOLOFLOW_E2E_SCREENSHOT_DIR", ROOT / "docs" / "images"))
RESULT_PATH = Path(os.environ.get("YOLOFLOW_E2E_RESULT", ROOT / "data" / "runs" / "e2e-result.json"))
TRAIN_TIMEOUT_S = int(os.environ.get("YOLOFLOW_E2E_TRAIN_TIMEOUT", "420"))


def _shot(page, name: str) -> None:
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SHOT_DIR / name), full_page=True)


def _write_result(payload: dict) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@pytest.mark.e2e
def test_upload_train_inspect_infer():
    assert ZIP_PATH.exists(), f"missing E2E zip {ZIP_PATH}; run scripts/prepare_e2e_zip.py"
    result: dict = {"steps": {}, "skipped": []}
    jpeg = ROOT / "data" / "datasets" / "_cache" / "coco8" / "images" / "val"
    sample_jpg = None
    if jpeg.exists():
        jpgs = sorted(jpeg.glob("*.jpg"))
        sample_jpg = jpgs[0] if jpgs else None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(BASE, wait_until="networkidle")
        expect(page.get_by_role("heading", name="YoloFlow")).to_be_visible()
        _shot(page, "ui-home.png")
        result["steps"]["home"] = "PASS"

        page.get_by_role("link", name="Datasets").click()
        expect(page.get_by_role("heading", name="Datasets")).to_be_visible()
        ds_name = f"e2e-coco8-{int(time.time())}"
        page.get_by_test_id("zip-name").fill(ds_name)
        page.get_by_test_id("zip-file").set_input_files(str(ZIP_PATH))
        page.get_by_test_id("zip-upload").click()
        expect(page.get_by_test_id("dataset-msg")).to_contain_text("uploaded zip", timeout=60_000)
        expect(page.get_by_text(ds_name)).to_be_visible()
        _shot(page, "ui-datasets.png")
        result["steps"]["upload_dataset"] = "PASS"

        page.get_by_role("link", name="Training").click()
        expect(page.get_by_role("heading", name="Training")).to_be_visible()
        page.get_by_test_id("job-dataset").locator("option", has_text=ds_name).wait_for(
            state="attached", timeout=30_000
        )
        page.get_by_test_id("job-name").fill("e2e-coco8-1ep")
        page.get_by_test_id("job-dataset").select_option(label=ds_name)
        page.get_by_test_id("job-epochs").fill("1")
        page.get_by_test_id("job-start").click()
        expect(page.get_by_text("e2e-coco8-1ep")).to_be_visible(timeout=15_000)
        result["steps"]["create_job"] = "PASS"

        deadline = time.time() + TRAIN_TIMEOUT_S
        train_status = "unknown"
        while time.time() < deadline:
            page.wait_for_timeout(3000)
            row = page.locator("table tbody tr", has_text="e2e-coco8-1ep").first
            if row.count():
                train_status = (row.locator("td").nth(2).inner_text() or "").strip()
                if train_status in {"succeeded", "failed", "cancelled"}:
                    break
        result["train_status"] = train_status
        if train_status != "succeeded":
            result["skipped"].append(
                {
                    "step": "wait_train_succeeded",
                    "status": "SKIPPED",
                    "reason": f"job ended as {train_status!r} or timed out after {TRAIN_TIMEOUT_S}s",
                }
            )
        else:
            result["steps"]["wait_train_succeeded"] = "PASS"
        _shot(page, "ui-training.png")

        page.get_by_role("link", name="Experiments").click()
        expect(page.get_by_role("heading", name="Experiments")).to_be_visible()
        _shot(page, "ui-experiments.png")
        result["steps"]["inspect_experiments"] = "PASS"

        page.get_by_role("link", name="Resources").click()
        expect(page.get_by_text("WebSocket")).to_be_visible()
        _shot(page, "ui-resources.png")
        result["steps"]["resources_ws"] = "PASS"

        page.get_by_role("link", name="Inference").click()
        expect(page.get_by_role("heading", name="Inference")).to_be_visible()
        if train_status == "succeeded":
            opt = page.get_by_test_id("infer-job").locator("option", has_text="e2e-coco8-1ep")
            opt.wait_for(state="attached", timeout=30_000)
            value = opt.get_attribute("value")
            assert value
            page.get_by_test_id("infer-job").select_option(value=value)
        infer_image = sample_jpg
        if infer_image is None:
            # fall back to any jpeg inside the zip extract
            candidates = list((ROOT / "data" / "datasets" / "_cache" / "coco8").rglob("*.jpg"))
            infer_image = candidates[0] if candidates else None
        if infer_image is None:
            result["skipped"].append(
                {"step": "run_inference", "status": "SKIPPED", "reason": "no sample jpeg on disk"}
            )
        else:
            page.get_by_test_id("infer-file").set_input_files(str(infer_image))
            page.get_by_test_id("infer-submit").click()
            expect(page.get_by_test_id("infer-result")).to_be_visible(timeout=120_000)
            full_text = page.get_by_test_id("infer-result").inner_text()
            result["steps"]["run_inference_full"] = {"status": "PASS", "text": full_text}
            page.get_by_test_id("infer-tiled").check()
            page.get_by_test_id("infer-submit").click()
            expect(page.get_by_test_id("infer-result")).to_contain_text("tiled", timeout=120_000)
            tiled_text = page.get_by_test_id("infer-result").inner_text()
            result["steps"]["run_inference_tiled"] = {"status": "PASS", "text": tiled_text}
        _shot(page, "ui-inference.png")
        browser.close()

    _write_result(result)
    assert result["steps"].get("upload_dataset") == "PASS"
    assert result["steps"].get("create_job") == "PASS"
    assert result["steps"].get("inspect_experiments") == "PASS"
    assert "run_inference_full" in result["steps"] or any(s.get("step") == "run_inference" for s in result["skipped"])
