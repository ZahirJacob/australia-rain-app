"""End-to-end test in a real browser: the page's JavaScript must fetch
Open-Meteo itself and the result must render.

Why this exists: every other test calls the Python endpoints directly (as
gradio_client does) and never executes the page's JavaScript. The browser
fetch feature once shipped completely dead while all of those passed.

Runs with Playwright's headless Chromium. Skips - never fails - when the
browser is not installed or Open-Meteo is unreachable, so it cannot block a
merge for reasons outside this repository. Set APP_URL to test a server that
is already running (e.g. the deployed site) instead of starting one.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")
requests = pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parent.parent
PAST_DATE = "2016-06-10"  # archive data: stable, never "not yet available"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _open_meteo_reachable() -> bool:
    try:
        r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                         params={"latitude": -33.87, "longitude": 151.21, "start_date": PAST_DATE,
                                 "end_date": PAST_DATE, "hourly": "temperature_2m", "timezone": "UTC"}, timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def app_url():
    url = os.environ.get("APP_URL")
    if url:
        yield url.rstrip("/") + "/"
        return
    port = _free_port()
    env = {**os.environ, "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_SERVER_PORT": str(port)}
    proc = subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/"
    try:
        for _ in range(60):
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("app did not start within 60 s")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def page(app_url):
    if not _open_meteo_reachable():
        pytest.skip("Open-Meteo not reachable from this machine")
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # browser binaries not installed
            pytest.skip(f"Chromium not available: {str(exc)[:80]}")
        page = browser.new_page()
        page.set_default_timeout(60_000)
        yield page
        browser.close()


def test_browser_fetches_open_meteo_and_result_renders(page, app_url):
    open_meteo_calls: list[str] = []
    page.on("response", lambda r: open_meteo_calls.append(f"{r.status} {r.url[:60]}") if "open-meteo.com" in r.url else None)
    page.goto(app_url, wait_until="networkidle")

    page.get_by_role("textbox", name=re.compile("Date")).first.fill(PAST_DATE)
    page.get_by_role("button", name="Fetch weather & predict").click()
    page.wait_for_function(f"document.body.innerText.includes('Inputs for Sydney on {PAST_DATE}')", timeout=60_000)

    text = page.inner_text("body")
    note = text[text.index("Inputs for"):][:300]
    assert "fetched by your browser" in note, f"server fallback was used instead of the browser fetch: {note}"
    assert "archive" in note
    assert re.search(r"P\(rain tomorrow\) = \d+\.\d%", text), "probability line missing"
    assert any(c.startswith("200 ") for c in open_meteo_calls), f"browser made no Open-Meteo request: {open_meteo_calls}"


def test_browser_error_path_shows_message(page, app_url):
    page.goto(app_url, wait_until="networkidle")
    page.get_by_role("textbox", name=re.compile("Date")).first.fill("2031-01-01")
    page.get_by_role("button", name="Fetch weather & predict").click()
    page.wait_for_function("document.body.innerText.includes('future')", timeout=60_000)
