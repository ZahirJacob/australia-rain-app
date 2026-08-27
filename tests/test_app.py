"""Offline tests for the Gradio app's helper functions (no server, no network)."""

import pandas as pd
import pytest

import app
from rainapp.predictor import INPUT_COLUMNS

SYDNEY = {"Date": "2015-06-10", "Location": "Sydney", "Humidity3pm": 90, "RainToday": "Yes"}


def test_result_verdict_follows_model_threshold_not_argmax(monkeypatch):
    """p = 0.55 is above 50% but below the 59.6% threshold: must read as No rain."""
    monkeypatch.setattr(app.PREDICTOR, "predict_one",
                        lambda rec: {"rain_tomorrow": "No", "probability": 0.55, "threshold": app.THRESHOLD})
    label, summary = app._result(SYDNEY)
    assert label == {"P(rain tomorrow)": 0.55}
    assert "No rain" in summary and "55.0%" in summary


def test_result_real_prediction():
    label, summary = app._result(SYDNEY)
    assert 0 <= label["P(rain tomorrow)"] <= 1
    assert ("Rain expected" in summary) == (label["P(rain tomorrow)"] >= app.THRESHOLD)


def test_table_roundtrip_preserves_values_and_blanks():
    record = {c: None for c in INPUT_COLUMNS}
    record.update(SYDNEY)
    table = app._record_to_table(record)
    assert list(table["Field"]) == app.EDITABLE and len(table) == 20
    back = app._table_to_record(table, "Sydney", "2015-06-10")
    assert back["Humidity3pm"] == "90" and back["RainToday"] == "Yes"
    assert back["MinTemp"] is None and back["Date"] == "2015-06-10" and back["Location"] == "Sydney"


def test_repredict_from_edited_table_matches_predict_one():
    record = {c: None for c in INPUT_COLUMNS}
    record.update(SYDNEY)
    table = app._record_to_table(record)
    table.loc[table["Field"] == "Humidity3pm", "Value"] = "20"
    label, _ = app.repredict("Sydney", "2015-06-10", table)
    expected = app.PREDICTOR.predict_one({**SYDNEY, "Humidity3pm": 20})["probability"]
    assert label["P(rain tomorrow)"] == pytest.approx(expected, abs=1e-9)


def test_manual_predict_with_blanks_and_invalid_date_error():
    values = ["" for _ in app.EDITABLE]
    values[app.EDITABLE.index("Humidity3pm")] = "95"
    label, summary = app.manual_predict("Hobart", "2015-07-01", *values)
    assert 0 <= label["P(rain tomorrow)"] <= 1
    values[app.EDITABLE.index("MinTemp")] = "[1,2]"  # a string, not a list: coerced to missing, no crash
    app.manual_predict("Hobart", "2015-07-01", *values)


def test_default_date_is_station_local_iso():
    d = app.default_date("Perth")
    assert len(d) == 10 and d[4] == "-" and d[7] == "-"
