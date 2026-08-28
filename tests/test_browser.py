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

from rainapp.i18n import TEXTS

ES_FETCH = TEXTS["es"]["fetch"]
ES_BROWSER_NOTE = TEXTS["es"]["src_browser"].split("{src}")[1].strip(" —")   # e.g. "obtenido por tu navegador"
ES_BAR = TEXTS["es"]["bar"]

playwright = pytest.importorskip("playwright.sync_api")
requests = pytest.importorskip("requests")

ROOT = Path(__file__).resolve().parent.parent
PAST_DATE = "2016-06-10"  # archive data: stable, never "not yet available"


def open_app(page, url: str, button_text: str, date_label: str):
    """Load the page and wait until it is really ready.

    `wait_until="networkidle"` never fires: the page-load language hook keeps
    Gradio's event stream open. Waiting for the (relabelled) button text proves
    the load hook ran, and waiting for the date box to be populated avoids a
    race where the page's default value overwrites a value typed too early.
    """
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function(f"document.body.innerText.includes({button_text!r})", timeout=30_000)
    date_box = page.get_by_role("textbox", name=re.compile(date_label)).first
    page.wait_for_function("(sel) => { const e = document.querySelector(sel); return !!e && e.value.length === 10; }",
                           arg="input[type=text], textarea", timeout=30_000)
    return date_box


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
def browser(app_url):
    if not _open_meteo_reachable():
        pytest.skip("Open-Meteo not reachable from this machine")
    with playwright.sync_playwright() as p:  # one Playwright per module: nesting is not allowed
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as exc:  # browser binaries not installed
            pytest.skip(f"Chromium not available: {str(exc)[:80]}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    page.set_default_timeout(60_000)
    yield page
    context.close()


def test_browser_fetches_open_meteo_and_result_renders(page, app_url):
    open_meteo_calls: list[str] = []
    page.on("response", lambda r: open_meteo_calls.append(f"{r.status} {r.url[:60]}") if "open-meteo.com" in r.url else None)
    date_box = open_app(page, app_url, "Fetch weather & predict", "Date")
    date_box.fill(PAST_DATE)
    page.get_by_role("button", name="Fetch weather & predict").click()
    page.wait_for_function(f"document.body.innerText.includes('Inputs for Sydney on {PAST_DATE}')", timeout=60_000)

    text = page.inner_text("body")
    note = text[text.index("Inputs for"):][:300]
    assert "fetched by your browser" in note, f"server fallback was used instead of the browser fetch: {note}"
    assert "archive" in note
    assert re.search(r"P\(rain tomorrow\) = \d+\.\d%", text), "probability line missing"
    assert any(c.startswith("200 ") for c in open_meteo_calls), f"browser made no Open-Meteo request: {open_meteo_calls}"


def test_browser_error_path_shows_message(page, app_url):
    date_box = open_app(page, app_url, "Fetch weather & predict", "Date")
    date_box.fill("2031-01-01")
    page.get_by_role("button", name="Fetch weather & predict").click()
    page.wait_for_function("document.body.innerText.includes('future')", timeout=60_000)


def test_spanish_browser_gets_spanish_ui_and_lang_override(browser, app_url):
    context = browser.new_context(locale="es-AR")
    page = context.new_page()
    page.set_default_timeout(60_000)
    try:
        date_box = open_app(page, app_url, ES_FETCH, "Fecha")
        date_box.fill(PAST_DATE)
        page.get_by_role("button", name=ES_FETCH).click()
        page.wait_for_function(f"document.body.innerText.includes({ES_BROWSER_NOTE!r})", timeout=60_000)
        text = page.inner_text("body")
        assert ES_BAR in text and f"Entradas para Sydney el {PAST_DATE}" in text
        # explicit override beats the browser language
        open_app(page, app_url + "?lang=en", "Fetch weather & predict", "Date")
    finally:
        context.close()
