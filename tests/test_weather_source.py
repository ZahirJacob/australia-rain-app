"""Offline tests for the Open-Meteo adapter (UTC hourly fixture recorded 2026-08-27)."""

import datetime as dt
import json
from pathlib import Path

import joblib
import pytest

from rainapp import weather_source as ws
from rainapp.predictor import COMPASS_POINTS, INPUT_COLUMNS
from rainapp.stations import STATIONS, TIMEZONES

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "open_meteo_utc_sydney_2016-06-10.json").read_text())
DAY = dt.date(2016, 6, 10)
TODAY = dt.date(2026, 8, 27)


def test_compass_conversion():
    assert ws.degrees_to_compass(0) == "N" and ws.degrees_to_compass(360) == "N"
    assert ws.degrees_to_compass(298) == "WNW"
    assert ws.degrees_to_compass(11.24) == "N" and ws.degrees_to_compass(11.26) == "NNE"
    assert ws.degrees_to_compass(None) is None
    assert set(ws.COMPASS) == COMPASS_POINTS


def test_oktas_conversion():
    assert ws.percent_to_oktas(0) == 0 and ws.percent_to_oktas(100) == 8
    assert ws.percent_to_oktas(37) == 3 and ws.percent_to_oktas(2) == 0
    assert ws.percent_to_oktas(None) is None


def test_map_payload_produces_full_record():
    rec = ws.map_payload(FIXTURE, "Sydney", DAY)
    assert set(rec) == set(INPUT_COLUMNS)
    assert rec["Date"] == "2016-06-10" and rec["Location"] == "Sydney"
    assert rec["Evaporation"] is None  # deliberately missing by default
    assert rec["RainToday"] in {"Yes", "No"} and rec["Rainfall"] >= 0
    assert rec["WindGustDir"] in ws.COMPASS and rec["WindDir9am"] in ws.COMPASS
    assert 0 <= rec["Cloud9am"] <= 8 and 900 < rec["Pressure9am"] < 1100
    assert rec["MinTemp"] <= rec["Temp9am"] <= rec["MaxTemp"] + 0.01


def test_local_time_conversion_is_true_local_not_fixed_offset():
    """UTC label 2016-06-09T23:00 is 09:00 AEST (winter, UTC+10) in Sydney."""
    series = ws._LocalHourly(FIXTURE, "Sydney")
    idx = FIXTURE["hourly"]["time"].index("2016-06-09T23:00")
    assert series.at(DAY, 9, "temperature_2m") == FIXTURE["hourly"]["temperature_2m"][idx]
    # In January (AEDT, UTC+11) the same local hour maps to 22:00 UTC instead.
    payload = {"hourly": {"time": ["2016-01-09T22:00", "2016-01-09T23:00"], "temperature_2m": [21.0, 22.0]}}
    summer = ws._LocalHourly(payload, "Sydney")
    assert summer.at(dt.date(2016, 1, 10), 9, "temperature_2m") == 21.0
    assert summer.at(dt.date(2016, 1, 10), 10, "temperature_2m") == 22.0


def test_rainfall_uses_9am_to_9am_window():
    """BoM Rainfall(D) is the 24 h to 09:00 on D; afternoon rain on D must not count."""
    times = [f"2016-06-{d:02d}T{h:02d}:00" for d in (9, 10) for h in range(24)]
    payload = {"hourly": {"time": times, "precipitation": [0.0] * 48, "temperature_2m": [15.0] * 48,
                          "pressure_msl": [1010.0] * 48}}
    # 2016-06-10T05:00 UTC == 15:00 AEST on 10 June: afternoon rain, belongs to tomorrow's window
    payload["hourly"]["precipitation"][times.index("2016-06-10T05:00")] = 7.0
    rec = ws.map_payload(payload, "Sydney", DAY)
    assert rec["Rainfall"] == 0.0 and rec["RainToday"] == "No"
    # 2016-06-09T20:00 UTC == 06:00 AEST on 10 June: inside the 24 h to 09:00
    payload["hourly"]["precipitation"][times.index("2016-06-09T20:00")] = 1.5
    rec = ws.map_payload(payload, "Sydney", DAY)
    assert rec["Rainfall"] == 1.5 and rec["RainToday"] == "Yes"


def test_rain_today_rule_uses_one_mm_threshold():
    times = [f"2016-06-{d:02d}T{h:02d}:00" for d in (9, 10) for h in range(24)]
    base = {"time": times, "precipitation": [0.0] * 48, "temperature_2m": [15.0] * 48}
    for total, expected in ((1.0, "No"), (1.01, "Yes")):
        p = json.loads(json.dumps(base)); p["precipitation"][times.index("2016-06-09T20:00")] = total
        assert ws.map_payload({"hourly": p}, "Sydney", DAY)["RainToday"] == expected


def test_et0_option_and_validation():
    assert ws.map_payload(FIXTURE, "Sydney", DAY, evaporation="et0")["Evaporation"] > 0
    with pytest.raises(ws.WeatherSourceError):
        ws.map_payload(FIXTURE, "Sydney", DAY, evaporation="ET0")


def test_all_null_day_raises_no_data():
    payload = json.loads(json.dumps(FIXTURE))
    for key in ws.HOURLY:
        payload["hourly"][key] = [None] * len(payload["hourly"]["time"])
    with pytest.raises(ws.NoDataError):
        ws.map_payload(payload, "Sydney", DAY)


def test_input_validation_raises_only_weather_source_error():
    with pytest.raises(ws.WeatherSourceError):
        ws.fetch_day("Sydney", "2030-01-01", today=TODAY)          # future
    with pytest.raises(ws.WeatherSourceError):
        ws.fetch_day("Atlantis", "2016-06-10", today=TODAY)        # unknown station
    with pytest.raises(ws.WeatherSourceError):
        ws.fetch_day("Sydney", "10/06/2016", today=TODAY)          # non-ISO
    assert ws._coerce_date(dt.datetime(2016, 6, 10, 14, 30)) == DAY  # datetime accepted
    assert ws._coerce_date(" 2016-06-10 ") == DAY                    # whitespace tolerated


def test_stations_table_matches_frozen_bundle():
    assert len(STATIONS) == 49 and set(TIMEZONES) == set(STATIONS)
    pre = joblib.load(Path(__file__).parent.parent / "model" / "preprocessor.joblib")
    bundle = pre.named_steps["weather_features"].location_coordinates
    assert set(bundle) == set(STATIONS)
    for name, coords in bundle.items():
        if name != "Richmond":
            assert tuple(coords) == STATIONS[name], name
    lat, lon = STATIONS["Richmond"]
    assert -34.5 < lat < -33 and 150 < lon < 151.5  # NSW override, not Victoria
