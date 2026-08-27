"""Offline tests for the Open-Meteo adapter (fixture recorded 2026-08-27)."""

import datetime as dt
import json
from pathlib import Path

import pytest

from rainapp import weather_source as ws
from rainapp.predictor import INPUT_COLUMNS
from rainapp.stations import STATIONS

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "open_meteo_sydney_2016-06-10.json").read_text())


def test_compass_conversion():
    assert ws.degrees_to_compass(0) == "N"
    assert ws.degrees_to_compass(360) == "N"
    assert ws.degrees_to_compass(298) == "WNW"
    assert ws.degrees_to_compass(11.24) == "N" and ws.degrees_to_compass(11.26) == "NNE"
    assert ws.degrees_to_compass(None) is None


def test_oktas_conversion():
    assert ws.percent_to_oktas(0) == 0 and ws.percent_to_oktas(100) == 8
    assert ws.percent_to_oktas(37) == 3 and ws.percent_to_oktas(2) == 0
    assert ws.percent_to_oktas(None) is None


def test_map_response_produces_full_record():
    rec = ws.map_response(FIXTURE, "Sydney", dt.date(2016, 6, 10))
    assert set(rec) == set(INPUT_COLUMNS)
    assert rec["Date"] == "2016-06-10" and rec["Location"] == "Sydney"
    assert rec["Pressure9am"] == 1017.6 and rec["WindDir9am"] == "WNW"
    assert rec["RainToday"] == "No" and rec["Rainfall"] == 0.0
    assert rec["Evaporation"] is None  # deliberately missing by default
    assert rec["Sunshine"] == pytest.approx(35257.32 / 3600)
    assert rec["WindGustSpeed"] == 34.6 and rec["WindGustDir"] == "W"
    assert rec["Cloud9am"] == 0.0 and rec["Cloud3pm"] == 3.0


def test_map_response_et0_option():
    rec = ws.map_response(FIXTURE, "Sydney", dt.date(2016, 6, 10), evaporation="et0")
    assert rec["Evaporation"] == 2.28


def test_rain_today_rule_uses_one_mm_threshold():
    payload = json.loads(json.dumps(FIXTURE))
    payload["daily"]["precipitation_sum"] = [1.0]
    assert ws.map_response(payload, "Sydney", dt.date(2016, 6, 10))["RainToday"] == "No"
    payload["daily"]["precipitation_sum"] = [1.1]
    assert ws.map_response(payload, "Sydney", dt.date(2016, 6, 10))["RainToday"] == "Yes"


def test_map_response_wrong_day_raises():
    with pytest.raises(ws.WeatherSourceError):
        ws.map_response(FIXTURE, "Sydney", dt.date(2016, 6, 11))


def test_fetch_day_rejects_future_and_unknown_station():
    with pytest.raises(ws.WeatherSourceError):
        ws.fetch_day("Sydney", "2030-01-01", today=dt.date(2026, 8, 27))
    with pytest.raises(ws.WeatherSourceError):
        ws.fetch_day("Atlantis", "2016-06-10", today=dt.date(2026, 8, 27))


def test_stations_table():
    assert len(STATIONS) == 49
    lat, lon = STATIONS["Richmond"]
    assert -34.5 < lat < -33 and 150 < lon < 151.5  # NSW, not Victoria
    # Norfolk Island (168E) is the only station outside mainland+Tasmania bounds
    assert all(-45 < lat < -10 and 110 < lon < 170 for lat, lon in STATIONS.values())
