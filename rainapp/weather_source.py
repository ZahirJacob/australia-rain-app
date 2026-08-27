"""Fetch one day of observations for a station from Open-Meteo and map it to
the 22 input columns the model expects.

Open-Meteo (https://open-meteo.com) is free, needs no API key, and serves both
an archive (ERA5 reanalysis, ~5-day delay) and a forecast endpoint that also
returns the current day. Its values are *model/reanalysis* estimates on a grid,
not the Bureau of Meteorology instrument readings the model was trained on.
`scripts/evaluate_api_shift.py` measures what that costs.

Column mapping (BoM semantics -> Open-Meteo field):
    MinTemp / MaxTemp      daily temperature_2m_min / _max
    Rainfall               daily precipitation_sum (mm)
    RainToday              Rainfall > 1 mm  (the dataset's rule)
    Evaporation            NOT AVAILABLE. Pan evaporation has no equivalent;
                           ET0 is systematically ~1/2-1/3 of it, so we leave it
                           missing and let the preprocessor impute (as it did
                           for 16 stations in training). Set
                           `evaporation="et0"` to use ET0 anyway.
    Sunshine               daily sunshine_duration (s) / 3600
    WindGustSpeed / Dir    hourly wind_gusts_10m: max over the day and the
                           wind direction at that hour, as 16-point compass
    *9am / *3pm            hourly values at local 09:00 / 15:00
    Cloud9am / Cloud3pm    cloud_cover % -> oktas = round(pct / 100 * 8)
    Pressure9am / 3pm      pressure_msl (hPa; BoM also reports MSL)
    WindSpeed9am / 3pm     wind_speed_10m (km/h)
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Mapping

import requests

from .stations import STATIONS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_DELAY_DAYS = 6  # the archive lags real time by ~5 days

HOURLY = (
    "temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
)
DAILY = (
    "temperature_2m_min", "temperature_2m_max", "precipitation_sum",
    "sunshine_duration", "et0_fao_evapotranspiration",
)
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


class WeatherSourceError(RuntimeError):
    pass


def degrees_to_compass(degrees: float | None) -> str | None:
    if degrees is None or (isinstance(degrees, float) and math.isnan(degrees)):
        return None
    return COMPASS[int((float(degrees) / 22.5) + 0.5) % 16]


def percent_to_oktas(percent: float | None) -> float | None:
    if percent is None:
        return None
    return float(min(8, max(0, round(float(percent) / 100 * 8))))


def _at(values: list, index: int):
    try:
        v = values[index]
    except IndexError:
        return None
    return None if v is None else v


def map_response(payload: Mapping[str, Any], station: str, date: dt.date,
                 evaporation: str = "missing") -> dict[str, Any]:
    """Turn one Open-Meteo JSON payload (one day, hourly+daily) into a record."""
    hourly, daily = payload.get("hourly") or {}, payload.get("daily") or {}
    times = hourly.get("time") or []
    day = date.isoformat()
    hours = [i for i, t in enumerate(times) if t.startswith(day)]
    if len(hours) != 24:
        raise WeatherSourceError(f"expected 24 hourly rows for {day}, got {len(hours)}")
    h9, h15 = hours[9], hours[15]
    dtimes = daily.get("time") or []
    if day not in dtimes:
        raise WeatherSourceError(f"no daily row for {day}")
    d = dtimes.index(day)

    gusts = [_at(hourly.get("wind_gusts_10m", []), i) for i in hours]
    valid = [(g, i) for g, i in zip(gusts, hours) if g is not None]
    if valid:
        gust_max, gust_hour = max(valid)
        gust_dir = degrees_to_compass(_at(hourly.get("wind_direction_10m", []), gust_hour))
    else:
        gust_max, gust_dir = None, None

    rainfall = _at(daily.get("precipitation_sum", []), d)
    sunshine = _at(daily.get("sunshine_duration", []), d)
    et0 = _at(daily.get("et0_fao_evapotranspiration", []), d)
    hv = lambda key, i: _at(hourly.get(key, []), i)  # noqa: E731

    return {
        "Date": day,
        "Location": station,
        "MinTemp": _at(daily.get("temperature_2m_min", []), d),
        "MaxTemp": _at(daily.get("temperature_2m_max", []), d),
        "Rainfall": rainfall,
        "Evaporation": et0 if evaporation == "et0" else None,
        "Sunshine": None if sunshine is None else sunshine / 3600.0,
        "WindGustDir": gust_dir,
        "WindGustSpeed": gust_max,
        "WindDir9am": degrees_to_compass(hv("wind_direction_10m", h9)),
        "WindDir3pm": degrees_to_compass(hv("wind_direction_10m", h15)),
        "WindSpeed9am": hv("wind_speed_10m", h9),
        "WindSpeed3pm": hv("wind_speed_10m", h15),
        "Humidity9am": hv("relative_humidity_2m", h9),
        "Humidity3pm": hv("relative_humidity_2m", h15),
        "Pressure9am": hv("pressure_msl", h9),
        "Pressure3pm": hv("pressure_msl", h15),
        "Cloud9am": percent_to_oktas(hv("cloud_cover", h9)),
        "Cloud3pm": percent_to_oktas(hv("cloud_cover", h15)),
        "Temp9am": hv("temperature_2m", h9),
        "Temp3pm": hv("temperature_2m", h15),
        "RainToday": None if rainfall is None else ("Yes" if rainfall > 1.0 else "No"),
    }


def _request(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise WeatherSourceError(f"weather API unreachable: {exc}") from exc
    if response.status_code != 200:
        raise WeatherSourceError(f"weather API returned {response.status_code}: {response.text[:200]}")
    return response.json()


def fetch_day(station: str, date: dt.date | str, *, evaporation: str = "missing",
              timeout: float = 15.0, today: dt.date | None = None) -> dict[str, Any]:
    """Return the model's 22-column record for `station` on `date`.

    Past dates older than ~6 days come from the archive endpoint; recent dates
    and today come from the forecast endpoint (which mixes observations with
    short-range forecast for the hours not yet elapsed).
    """
    if station not in STATIONS:
        raise WeatherSourceError(f"unknown station {station!r}")
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    today = today or dt.date.today()
    if date > today:
        raise WeatherSourceError("date is in the future; the model needs the day's observations")
    lat, lon = STATIONS[station]
    params: dict[str, Any] = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "hourly": ",".join(HOURLY), "daily": ",".join(DAILY),
        "start_date": date.isoformat(), "end_date": date.isoformat(),
    }
    url = ARCHIVE_URL if (today - date).days >= ARCHIVE_DELAY_DAYS else FORECAST_URL
    payload = _request(url, params, timeout)
    return map_response(payload, station, date, evaporation=evaporation)


def fetch_range(station: str, start: dt.date, end: dt.date, *, evaporation: str = "missing",
                timeout: float = 60.0) -> list[dict[str, Any]]:
    """Archive-only bulk fetch (one request) for evaluation scripts."""
    lat, lon = STATIONS[station]
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "hourly": ",".join(HOURLY), "daily": ",".join(DAILY),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    }
    payload = _request(ARCHIVE_URL, params, timeout)
    records = []
    day = start
    while day <= end:
        try:
            records.append(map_response(payload, station, day, evaporation=evaporation))
        except WeatherSourceError:
            pass  # a day missing from the archive is skipped
        day += dt.timedelta(days=1)
    return records
