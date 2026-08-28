"""Offline tests for the Gradio app's helper functions (no server, no network)."""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

import app
from rainapp.predictor import INPUT_COLUMNS

SYDNEY = {"Date": "2015-06-10", "Location": "Sydney", "Humidity3pm": 90, "RainToday": "Yes"}


def test_result_verdict_follows_model_threshold_not_argmax(monkeypatch):
    """p = 0.55 is above 50% but below the 59.6% threshold: must read as No rain."""
    monkeypatch.setattr(app.PREDICTOR, "predict_one",
                        lambda rec: {"rain_tomorrow": "No", "probability": 0.55, "threshold": app.THRESHOLD})
    label, summary = app._result(SYDNEY, "en")
    assert label == {"P(rain tomorrow)": 0.55}
    assert "No rain" in summary and "55.0%" in summary


def test_result_real_prediction():
    label, summary = app._result(SYDNEY, "en")
    assert 0 <= label["P(rain tomorrow)"] <= 1
    assert ("Rain expected" in summary) == (label["P(rain tomorrow)"] >= app.THRESHOLD)


def test_table_roundtrip_preserves_values_and_blanks():
    record = {c: None for c in INPUT_COLUMNS}
    record.update(SYDNEY)
    table = app._record_to_table(record, "en")
    assert list(table["Field"]) == app.EDITABLE and len(table) == 20
    back = app._table_to_record(table, "Sydney", "2015-06-10", "en")
    assert back["Humidity3pm"] == "90" and back["RainToday"] == "Yes"
    assert back["MinTemp"] is None and back["Date"] == "2015-06-10" and back["Location"] == "Sydney"


def test_repredict_from_edited_table_matches_predict_one():
    record = {c: None for c in INPUT_COLUMNS}
    record.update(SYDNEY)
    table = app._record_to_table(record, "en")
    table.loc[table["Field"] == "Humidity3pm", "Value"] = "20"
    label, _ = app.repredict("Sydney", "2015-06-10", table)
    expected = app.PREDICTOR.predict_one({**SYDNEY, "Humidity3pm": 20})["probability"]
    assert label["P(rain tomorrow)"] == pytest.approx(expected, abs=1e-9)


def test_manual_predict_with_blanks():
    values = ["" for _ in app.EDITABLE]
    values[app.EDITABLE.index("Humidity3pm")] = "95"
    label, summary = app.manual_predict("Hobart", "2015-07-01", *values)
    assert 0 <= label["P(rain tomorrow)"] <= 1
    values[app.EDITABLE.index("MinTemp")] = "13.6C"  # unparseable text is treated as missing
    app.manual_predict("Hobart", " 2015-07-01 ", *values)  # surrounding whitespace tolerated


def test_invalid_or_blank_date_is_an_error_everywhere():
    import gradio as gr
    values = ["" for _ in app.EDITABLE]
    for bad in ("10/06/2015", "", "yesterday"):
        with pytest.raises(gr.Error):
            app.manual_predict("Hobart", bad, *values)
        with pytest.raises(gr.Error):
            app.repredict("Hobart", bad, app._record_to_table({c: None for c in INPUT_COLUMNS} | SYDNEY, "en"))


def test_repredict_on_empty_table_is_an_error():
    import gradio as gr
    with pytest.raises(gr.Error, match="Fetch weather"):
        app.repredict("Sydney", "2015-06-10", pd.DataFrame(columns=["Field", "Value", "Unit / meaning"]))


def test_sync_date_keeps_user_typed_date():
    assert app.sync_date("Perth", "2015-06-10") == "2015-06-10"
    assert app.sync_date("Perth", "  ") == app.default_date("Perth")


def test_dropdown_choices_come_from_predictor_vocabulary():
    assert app._choices("RainToday") == ["", "Yes", "No"]
    assert app._choices("WindDir9am")[1:] == list(app.COMPASS)


def test_default_date_is_station_local_iso():
    d = app.default_date("Perth")
    assert len(d) == 10 and d[4] == "-" and d[7] == "-"


# ---- browser-side fetch ------------------------------------------------------

FIXTURE_TEXT = (Path(__file__).parent / "fixtures" / "open_meteo_utc_sydney_2016-06-10.json").read_text()
WRAPPED = json.dumps({"source": "archive", "body": json.loads(FIXTURE_TEXT)})


def test_browser_payload_is_used_without_server_fetch(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("server fetch must not be called when the browser payload is valid")
    monkeypatch.setattr(app, "fetch_day_with_source", boom)
    label, summary, table, note = app.fetch_and_predict("Sydney", "2016-06-10", WRAPPED)
    assert "fetched by your browser" in note and "archive" in note
    assert label["P(rain tomorrow)"] == pytest.approx(0.2604687511920929, abs=1e-6)


def test_browser_payload_keeps_endpoint_provenance():
    note = app.fetch_and_predict("Sydney", "2016-06-10", json.dumps({"source": "forecast", "body": json.loads(FIXTURE_TEXT)}))[3]
    assert "short-range forecast" in note and "fetched by your browser" in note


def test_unusable_browser_payloads_fall_back_to_server(monkeypatch):
    calls = []
    def fake(station, date, **k):
        calls.append(station)
        return app.map_payload(json.loads(FIXTURE_TEXT), station, dt.date(2016, 6, 10)), "archive"
    monkeypatch.setattr(app, "fetch_day_with_source", fake)
    body = json.loads(FIXTURE_TEXT)
    wrong_station = json.dumps({"source": "archive", "body": {**body, "latitude": -37.81, "longitude": 144.96}})  # Melbourne
    bad = ["", "not json", FIXTURE_TEXT,                                   # raw body without wrapper
           '{"source": "archive", "body": {"hourly": {}}}',
           '{"source": "hacked", "body": %s}' % FIXTURE_TEXT,
           wrong_station,
           "[]", "null", '"abc"', '{"source": "archive", "body": []}',
           '{"source": "archive", "body": {"latitude": -33.85, "longitude": 151.2, "hourly": "x"}}',
           '{"source": "archive", "body": {"latitude": -33.85, "longitude": 151.2, "hourly": {"time": ["2016-06-10T00:00"], "temperature_2m": {"a": 1}}}}',
           '{"source": "archive", "body": {"latitude": -33.85, "longitude": 151.2, "hourly": {"time": ["2016-06-10T00:00"], "wind_direction_10m": ["abc"]}}}']
    for payload in bad:
        _, _, _, note = app.fetch_and_predict("Sydney", "2016-06-10", payload)
        assert "fetched by your browser" not in note
    # payload for another day than requested
    _, _, _, note = app.fetch_and_predict("Sydney", "2016-06-12", WRAPPED)
    assert "fetched by your browser" not in note
    assert len(calls) == len(bad) + 1


def test_future_date_with_browser_payload_is_rejected():
    import gradio as gr
    with pytest.raises(gr.Error, match="future"):
        app.fetch_and_predict("Sydney", "2031-01-01", WRAPPED)


def test_browser_js_embeds_config():
    assert '"Sydney": [' in app.BROWSER_FETCH_JS and "precipitation" in app.BROWSER_FETCH_JS
    assert "Australia/Sydney" in app.BROWSER_FETCH_JS and "AbortSignal.timeout" in app.BROWSER_FETCH_JS
    assert app.ARCHIVE_URL in app.BROWSER_FETCH_JS and app.FORECAST_URL in app.BROWSER_FETCH_JS


def test_browser_js_is_valid_javascript_and_returns_input_list():
    esprima = pytest.importorskip("esprima")
    esprima.parseScript("const f = " + app.BROWSER_FETCH_JS + ";")
    # Gradio feeds the JS return list to the Python handler as (station, date, payload)
    assert "async (station, date, previousPayload, lang)" in app.BROWSER_FETCH_JS
    assert "[station, date, payload, lang]" in app.BROWSER_FETCH_JS


# ---- i18n ----------------------------------------------------------------------

from rainapp import i18n


def test_language_dictionaries_have_identical_keys():
    assert set(i18n.TEXTS["en"]) == set(i18n.TEXTS["es"])
    assert set(i18n.FIELD_HELP["en"]) == set(i18n.FIELD_HELP["es"]) == set(app.EDITABLE)
    assert set(i18n.ABOUT) == {"en", "es"}
    for key in i18n.TEXTS["en"]:  # same format placeholders in both languages
        import re
        assert set(re.findall(r"{(\w+)}", i18n.TEXTS["en"][key])) == set(re.findall(r"{(\w+)}", i18n.TEXTS["es"][key])), key


def test_pick_language():
    assert i18n.pick_language("es-AR,es;q=0.9,en;q=0.8") == "es"
    assert i18n.pick_language("en-US,en;q=0.9") == "en"
    assert i18n.pick_language("fr-FR,fr;q=0.9") == "en"          # unsupported -> default
    assert i18n.pick_language(None) == "en"
    assert i18n.pick_language("en-US", "es") == "es"             # ?lang= wins
    assert i18n.pick_language("es-AR", "EN") == "en"
    assert i18n.pick_language("es-AR", "de") == "es"             # unsupported override ignored


def test_spanish_outputs_end_to_end():
    label, summary, table, note = app.fetch_and_predict("Sydney", "2016-06-10", WRAPPED, "es")
    assert list(label) == ["P(lluvia mañana)"] and "mañana" in summary and "umbral" in summary
    assert i18n.TEXTS["es"]["src_browser"].split("{src}")[1].strip(" —") in note and "Entradas para **Sydney**" in note
    assert list(table.columns) == ["Campo", "Valor", "Unidad / significado"]
    values = ["" for _ in app.EDITABLE]
    _, es_summary = app.manual_predict("Hobart", "2015-07-01", *values, "es")
    _, en_summary = app.manual_predict("Hobart", "2015-07-01", *values)          # 20 values, no language -> English
    assert "lluvia" in es_summary and "rain" in en_summary


def test_spanish_error_messages():
    import gradio as gr
    with pytest.raises(gr.Error, match="AAAA-MM-DD"):
        app.manual_predict("Hobart", "10/06/2015", *["" for _ in app.EDITABLE], "es")
    with pytest.raises(gr.Error, match="futuro"):
        app.fetch_and_predict("Sydney", "2031-01-01", WRAPPED, "es")
    with pytest.raises(gr.Error, match="primero"):
        app.repredict("Sydney", "2015-06-10", pd.DataFrame(columns=["Campo", "Valor", "Unidad"]), "es")


def test_unknown_language_falls_back_to_english():
    _, summary, _, note = app.fetch_and_predict("Sydney", "2016-06-10", WRAPPED, "de")
    assert "rain" in summary and "fetched by your browser" in note
