"""Gradio UI for the Australian rain predictor.

Run locally:  python app.py   (then open http://127.0.0.1:7860)
On Hugging Face Spaces this file is the entry point.
"""

from __future__ import annotations

import json
import math
from typing import Any

import gradio as gr
import pandas as pd

from rainapp import load_default_predictor
from rainapp.i18n import ABOUT, DEFAULT_LANGUAGE, FIELD_HELP, LANGUAGES, pick_language, t
from rainapp.predictor import CATEGORICAL_VOCAB, COMPASS_POINTS, INPUT_COLUMNS
from rainapp.stations import STATION_NAMES, STATIONS, TIMEZONES
from rainapp.weather_source import (ARCHIVE_DELAY_DAYS, ARCHIVE_URL, COMPASS, FORECAST_PAST_LIMIT_DAYS,
                                    FORECAST_URL, HOURLY, NoDataError, WeatherSourceError, _coerce_date,
                                    fetch_day_with_source, map_payload, station_today)

PREDICTOR = load_default_predictor()
THRESHOLD = PREDICTOR.threshold

EDITABLE = [c for c in INPUT_COLUMNS if c not in ("Date", "Location")]
SOURCES = ("archive", "forecast")


def _lang(lang: str | None) -> str:
    return lang if lang in LANGUAGES else DEFAULT_LANGUAGE

# The visitor's browser fetches Open-Meteo directly (its own IP and quota;
# the free tier is rate-limited per IP, and a shared hosting IP exhausts it).
# The server-side fetch remains as the fallback when the browser call fails.
# Gradio runs this `js` function BEFORE the Python handler of the same event and
# feeds its returned list to the handler as its inputs: (station, date, payload).
# (A separate fn=None JS event chained with .then() never fires the .then in
# Gradio 5.50 - verified with a minimal app - so both must be on one event.)
_BROWSER_CFG = json.dumps({
    "stations": STATIONS, "timezones": TIMEZONES, "hourly": ",".join(HOURLY),
    "archiveDelayDays": ARCHIVE_DELAY_DAYS, "forecastPastLimitDays": FORECAST_PAST_LIMIT_DAYS,
    "archiveUrl": ARCHIVE_URL, "forecastUrl": FORECAST_URL, "timeoutMs": 15000,
})
BROWSER_FETCH_JS = r"""
async (station, date, previousPayload, lang) => {
  const cfg = __CFG__;
  const done = (payload) => [station, date, payload, lang];
  const fetchJson = async (base, prevIso, dayIso, coords) => {
    const url = base + "?latitude=" + coords[0] + "&longitude=" + coords[1] + "&timezone=UTC"
      + "&hourly=" + cfg.hourly + "&start_date=" + prevIso + "&end_date=" + dayIso;
    const r = await fetch(url, { signal: AbortSignal.timeout(cfg.timeoutMs) });
    if (!r.ok) return null;
    return await r.json();
  };
  const hasData = (j) => !!(j && j.hourly && Array.isArray(j.hourly.temperature_2m)
                            && j.hourly.temperature_2m.some((v) => v !== null));
  try {
    const coords = cfg.stations[station];
    if (!coords || !/^\s*\d{4}-\d{2}-\d{2}\s*$/.test(date)) return done("");
    const dayIso = date.trim();
    const day = new Date(dayIso + "T00:00:00Z");
    if (isNaN(day) || day.toISOString().slice(0, 10) !== dayIso) return done("");
    // "today" at the station, same rule as the server (station_today)
    const todayIso = new Date().toLocaleDateString("en-CA", { timeZone: cfg.timezones[station] });
    const ageDays = Math.round((new Date(todayIso + "T00:00:00Z").getTime() - day.getTime()) / 86400000);
    if (ageDays < 0) return done("");
    const prevIso = new Date(day.getTime() - 86400000).toISOString().slice(0, 10);
    let source = ageDays >= cfg.archiveDelayDays ? "archive" : "forecast";
    let body = await fetchJson(source === "archive" ? cfg.archiveUrl : cfg.forecastUrl, prevIso, dayIso, coords);
    if (source === "archive" && !hasData(body) && ageDays <= cfg.forecastPastLimitDays) {
      source = "forecast";
      body = await fetchJson(cfg.forecastUrl, prevIso, dayIso, coords);
    }
    if (!hasData(body)) return done("");
    return done(JSON.stringify({ source: source, body: body }));
  } catch (e) { return done(""); }
}
""".replace("__CFG__", _BROWSER_CFG)


def _choices(column: str) -> list[str]:
    """Dropdown choices for a categorical column, derived from the predictor's vocabulary."""
    vocab = CATEGORICAL_VOCAB[column]
    ordered = list(COMPASS) if vocab == COMPASS_POINTS else sorted(vocab, reverse=True)  # Yes before No
    return [""] + ordered


# ------------------------------------------------------------------ helpers
def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def _blank_to_none(v: Any) -> Any:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    text = str(v).strip()
    return text or None


def _valid_date(date: str, lang: str) -> str:
    """ISO date string or gr.Error; shared by every prediction path."""
    try:
        return _coerce_date(date).isoformat()
    except WeatherSourceError as exc:
        raise gr.Error(t(lang, "err_date")) from exc


def _table_headers(lang: str) -> list[str]:
    return [t(lang, "col_field"), t(lang, "col_value"), t(lang, "col_unit")]


def _record_to_table(record: dict[str, Any], lang: str) -> pd.DataFrame:
    h = _table_headers(lang)
    return pd.DataFrame({h[0]: EDITABLE, h[1]: [_fmt(record[c]) for c in EDITABLE],
                         h[2]: [FIELD_HELP[lang][c] for c in EDITABLE]})


def _table_to_record(table: pd.DataFrame, station: str, date: str, lang: str) -> dict[str, Any]:
    if table is None or len(table) != len(EDITABLE) or table.shape[1] < 2 or set(table.iloc[:, 0]) != set(EDITABLE):
        raise gr.Error(t(lang, "err_no_table"))
    values = dict(zip(table.iloc[:, 0], table.iloc[:, 1]))
    record: dict[str, Any] = {"Date": date, "Location": station}
    for column in EDITABLE:
        record[column] = _blank_to_none(values.get(column))
    return record


def _result(record: dict[str, Any], lang: str) -> tuple[dict[str, float], str]:
    """Label value + markdown verdict.

    The verdict must come from the model's threshold, not from which
    probability is larger, so the Label shows a single bar for P(rain) and the
    Yes/No decision is stated in the markdown.
    """
    out = PREDICTOR.predict_one(record)
    p = out["probability"]
    verdict = t(lang, "verdict_rain" if out["rain_tomorrow"] == "Yes" else "verdict_dry")
    why = t(lang, "why") if abs(THRESHOLD - 0.5) > 1e-9 else ""
    summary = f"## {verdict}\n" + t(lang, "prob_line", p=f"{p:.1%}", thr=f"{THRESHOLD:.1%}", why=why)
    return {t(lang, "bar"): p}, summary


# ------------------------------------------------------------------ actions
def _predict_or_error(record: dict[str, Any], lang: str):
    try:
        return _result(record, lang)
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc


def _weather_error(exc: WeatherSourceError, station: str, date: str, lang: str) -> gr.Error:
    msg = str(exc)
    if isinstance(exc, NoDataError):
        return gr.Error(t(lang, "err_nodata", station=station, date=date))
    if "future" in msg:
        return gr.Error(t(lang, "err_future"))
    if "ISO" in msg:
        return gr.Error(t(lang, "err_date"))
    return gr.Error(t(lang, "err_weather", msg=msg))


COORD_TOLERANCE_DEG = 0.2  # Open-Meteo snaps to its grid cell (~0.1-0.25 deg)


def _record_from_browser(station: str, date: str, payload_text: str):
    """(record, source) from the JSON the visitor's browser fetched, or None if unusable.

    The payload is untrusted input: it must be the {source, body} wrapper the
    page's JS produces, its coordinates must match the requested station's
    grid cell, its hourly times must cover the requested date, and the date
    must not be in the future. Anything else -> None -> server-side fetch.
    """
    if not payload_text or not str(payload_text).strip():
        return None
    try:
        wrapper = json.loads(payload_text)
        source, body = wrapper["source"], wrapper["body"]
        if source not in SOURCES or not isinstance(body, dict) or not isinstance(body.get("hourly"), dict):
            return None
        day = _coerce_date(date)
        if day > station_today(station):
            return None
        lat, lon = STATIONS[station]
        if abs(float(body["latitude"]) - lat) > COORD_TOLERANCE_DEG or abs(float(body["longitude"]) - lon) > COORD_TOLERANCE_DEG:
            return None
        times = body.get("hourly", {}).get("time") or []
        if not any(str(t).startswith(day.isoformat()) for t in times):
            return None
        record = map_payload(body, station, day)
    except (ValueError, TypeError, KeyError, AttributeError, IndexError, WeatherSourceError):
        return None
    return record, source


def fetch_and_predict(station: str, date: str, browser_payload: str = "", lang: str = DEFAULT_LANGUAGE):
    lang = _lang(lang)
    from_browser = _record_from_browser(station, date, browser_payload)
    if from_browser is not None:
        record, source = from_browser
        source_text = t(lang, "src_browser", src=t(lang, f"src_{source}"))
    else:
        try:
            record, source = fetch_day_with_source(station, date)
        except WeatherSourceError as exc:
            raise _weather_error(exc, station, date, lang) from exc
        source_text = t(lang, f"src_{source}")
    label, summary = _predict_or_error(record, lang)
    note = t(lang, "note", station=station, date=record["Date"], src=source_text)
    return label, summary, _record_to_table(record, lang), note


def repredict(station: str, date: str, table: pd.DataFrame, lang: str = DEFAULT_LANGUAGE):
    lang = _lang(lang)
    return _predict_or_error(_table_to_record(table, station, _valid_date(date, lang), lang), lang)


def manual_predict(station, date, *values):
    """values = the EDITABLE fields, optionally followed by the language."""
    if len(values) == len(EDITABLE) + 1:
        *values, lang = values
    else:
        lang = DEFAULT_LANGUAGE
    lang = _lang(lang)
    record = {"Date": _valid_date(date, lang), "Location": station}
    for column, v in zip(EDITABLE, values):
        record[column] = _blank_to_none(v)
    return _predict_or_error(record, lang)


def default_date(station: str) -> str:
    return station_today(station).isoformat()


def sync_date(station: str, current: str) -> str:
    """When the station changes, only fill the date if the user left it blank."""
    return current if (current or "").strip() else default_date(station)


# ------------------------------------------------------------------ layout
from rainapp.i18n import TEXTS
E = TEXTS["en"]


def _field_label(column: str, lang: str) -> str:
    return f"{column} ({FIELD_HELP[lang][column]})"


with gr.Blocks(title="Australia rain tomorrow") as demo:
    lang_state = gr.State(DEFAULT_LANGUAGE)
    title_md = gr.Markdown(E["title"])
    lang_md = gr.Markdown(E["lang_note"])

    with gr.Tab(E["tab_station"]) as tab_station:
        with gr.Row():
            station = gr.Dropdown(choices=list(STATION_NAMES), value="Sydney", label=E["station"])
            date = gr.Textbox(value="", label=E["date_local"])  # filled by the page-load hook
            go = gr.Button(E["fetch"], variant="primary")
        with gr.Row():
            label = gr.Label(label=E["prob_label"])
            summary = gr.Markdown()
        note = gr.Markdown()
        browser_payload = gr.Textbox(visible=False)
        with gr.Accordion(E["accordion"], open=False) as accordion:
            table = gr.Dataframe(headers=[E["col_field"], E["col_value"], E["col_unit"]], datatype=["str", "str", "str"],
                                 interactive=True, label=E["table_label"], row_count=(len(EDITABLE), "fixed"))
            redo = gr.Button(E["repredict"])

        station.change(sync_date, [station, date], date)
        go.click(fetch_and_predict, [station, date, browser_payload, lang_state], [label, summary, table, note], js=BROWSER_FETCH_JS)
        redo.click(repredict, [station, date, table, lang_state], [label, summary])

    with gr.Tab(E["tab_manual"]) as tab_manual:
        manual_md = gr.Markdown(E["manual_intro"])
        with gr.Row():
            m_station = gr.Dropdown(choices=list(STATION_NAMES), value="Sydney", label=E["station"])
            m_date = gr.Textbox(value="", label=E["date"])  # filled by the page-load hook
        inputs = []
        with gr.Row():
            for chunk in (EDITABLE[:7], EDITABLE[7:14], EDITABLE[14:]):
                with gr.Column():
                    for column in chunk:
                        if column in CATEGORICAL_VOCAB:
                            inputs.append(gr.Dropdown(choices=_choices(column), value="", label=_field_label(column, "en")))
                        else:
                            inputs.append(gr.Textbox(value="", label=_field_label(column, "en")))
        m_station.change(sync_date, [m_station, m_date], m_date)
        m_go = gr.Button(E["predict"], variant="primary")
        with gr.Row():
            m_label = gr.Label(label=E["prob_label"])
            m_summary = gr.Markdown()
        m_go.click(manual_predict, [m_station, m_date, *inputs, lang_state], [m_label, m_summary])

    with gr.Tab(E["tab_about"]) as tab_about:
        about_md = gr.Markdown(ABOUT["en"])

    # ---- page-load hook: language chosen once, every visible text relabelled,
    # and the default date set. Everything arrives in ONE update, so nothing
    # a user types afterwards can be overwritten by a late default value
    # (a `value=lambda` default is fetched separately and lands late).
    RELABEL = [  # (component, text key, attribute)
        (title_md, "title", "value"), (lang_md, "lang_note", "value"),
        (tab_station, "tab_station", "label"), (tab_manual, "tab_manual", "label"), (tab_about, "tab_about", "label"),
        (station, "station", "label"), (go, "fetch", "value"),
        (label, "prob_label", "label"), (accordion, "accordion", "label"), (table, "table_label", "label"),
        (redo, "repredict", "value"), (manual_md, "manual_intro", "value"),
        (m_station, "station", "label"), (m_go, "predict", "value"),
        (m_label, "prob_label", "label"),
    ]

    def apply_language(request: gr.Request):
        lang = pick_language(request.headers.get("accept-language"), request.query_params.get("lang"))
        today = default_date("Sydney")
        updates = [gr.update(**{attr: t(lang, key)}) for _, key, attr in RELABEL]
        updates.append(gr.update(label=t(lang, "date_local"), value=today))
        updates.append(gr.update(label=t(lang, "date"), value=today))
        updates.append(gr.update(headers=_table_headers(lang)))
        updates.append(gr.update(value=ABOUT[lang]))
        updates.extend(gr.update(label=_field_label(column, lang)) for column in EDITABLE)
        return [lang, *updates]

    demo.load(apply_language, None, [lang_state, *[c for c, _, _ in RELABEL], date, m_date, table, about_md, *inputs])

if __name__ == "__main__":
    demo.launch()
