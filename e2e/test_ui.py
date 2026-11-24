"""Playwright E2E of main UI routes. Run with the stack up (see scripts/acceptance_e2e.sh)."""

from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("playwright")
from playwright.sync_api import expect, sync_playwright

BASE = os.environ.get("YOLOFLOW_E2E_BASE", "http://127.0.0.1:5173")


@pytest.mark.e2e
def test_main_nav_and_pages():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(BASE, wait_until="networkidle")
        expect(page.get_by_role("heading", name="YoloFlow")).to_be_visible()
        for label in ("Inference", "Datasets", "Training", "Experiments", "Resources"):
            page.get_by_role("link", name=label).click()
            page.wait_for_timeout(300)
            expect(page.locator("main")).to_be_visible()
        page.get_by_role("link", name="Resources").click()
        expect(page.get_by_text(re.compile("WebSocket"))).to_be_visible()
        page.get_by_role("link", name="Datasets").click()
        expect(page.get_by_role("heading", name="Datasets")).to_be_visible()
        page.get_by_role("link", name="Training").click()
        expect(page.get_by_role("heading", name="Training")).to_be_visible()
        page.get_by_role("link", name="Inference").click()
        expect(page.get_by_role("heading", name="Inference")).to_be_visible()
        browser.close()
