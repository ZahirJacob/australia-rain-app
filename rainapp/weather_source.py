"""Fetch one day of observations for a station from Open-Meteo and map it to
the 22 input columns the model expects.

Open-Meteo (https://open-meteo.com) is free, needs no API key, and serves both
an archive (ERA5 reanalysis, ~5-7 day delay) and a forecast endpoint that also
covers the recent past and today. Its values are *model/reanalysis* estimates
on a grid, not the Bureau of Meteorology instrument readings the model was
trained on. `scripts/evaluate_api_shift.py` measures what that costs.

Time windows follow BoM daily-observation conventions, which the training data
uses (verified: RainTomorrow(D) == RainToday(D+1) on every row pair, so the
label window starts at 09:00 today):

    Rainfall(D), Evaporation(D), MinTemp(D)   24 h ending 09:00 local on D
    MaxTemp(D), Sunshine(D), WindGust*(D)     local calendar day D
    *9am / *3pm                               local 09:00 / 15:00 on D
    RainToday                                 Rainfall > 1 mm (the dataset's rule)

Everything is aggregated from *hourly* data requested in UTC and converted to
the station's true local time with zoneinfo, because Open-Meteo applies one
fixed UTC offset per response (no daylight-saving changes).

Other unit conversions: cloud_cover % -> oktas (round(pct/100*8)); wind
direction degrees -> 16-point compass; sunshine s -> h. Pan `Evaporation` has
no Open-Meteo equivalent: it is left missing by default (the preprocessor
imputes it, as it did for 16 stations in training); `evaporation="et0"` uses
FAO reference evapotranspiration instead, a related but smaller quantity.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings
from typing import Any, Literal, Mapping
from zoneinfo import ZoneInfo

import requests

from .stations import STATIONS, TIMEZONES

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_DELAY_DAYS = 7      # archive lags real time by ~5-7 days
FORECAST_PAST_LIMIT_DAYS = 90

HOURLY = (
    "temperature_2m", "relative_humidity_2m", "pressure_msl", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "precipitation", "sunshine_duration", "et0_fao_evapotranspiration",
)
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
EvaporationMode = Literal["missing", "et0"]
_EVAPORATION_MODES = ("missing", "et0")


class WeatherSourceError(RuntimeError):
    """Any failure of the weather source; the only exception type raised here."""


class NoDataError(WeatherSourceError):
    """The API answered, but has no observations for that station/day (yet)."""


# ----------------------------------------------------------------- helpers
def degrees_to_compass(degrees: float | None) -> str | None:
    if degrees is None or (isinstance(degrees, float) and math.isnan(degrees)):
        return None
    return COMPASS[int((float(degrees) / 22.5) + 0.5) % 16]


def percent_to_oktas(percent: float | None) -> float | None:
    if percent is None or (isinstance(percent, float) and math.isnan(percent)):
        return None
    return float(min(8, max(0, round(float(percent) / 100 * 8))))


def _coerce_date(date: dt.date | dt.datetime | str) -> dt.date:
    if isinstance(date, dt.datetime):
        return date.date()
    if isinstance(date, dt.date):
        return date
    try:
        return dt.date.fromisoformat(str(date).strip())
    except ValueError as exc:
        raise WeatherSourceError(f"date must be ISO YYYY-MM-DD, got {date!r}") from exc


def _check_station(station: str) -> None:
    if station not in STATIONS:
        raise WeatherSourceError(f"unknown station {station!r}")


def _check_evaporation(mode: str) -> None:
    if mode not in _EVAPORATION_MODES:
        raise WeatherSourceError(f"evaporation must be one of {_EVAPORATION_MODES}, got {mode!r}")


def station_today(station: str) -> dt.date:
    """Today's calendar date at the station (not the server)."""
    return dt.datetime.now(ZoneInfo(TIMEZONES[station])).date()


def _params(station: str, start: dt.date, end: dt.date) -> dict[str, Any]:
    lat, lon = STATIONS[station]
    return {
        "latitude": lat, "longitude": lon, "timezone": "UTC",
        "hourly": ",".join(HOURLY),
        # one extra day before `start` so the 09:00-to-09:00 windows are complete
        "start_date": (start - dt.timedelta(days=1)).isoformat(),
        "end_date": end.isoformat(),
    }


def _request(url: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code != 200:
            raise WeatherSourceError(f"weather API returned {response.status_code}: {response.text[:200]}")
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherSourceError(f"weather API unreachable: {exc}") from exc
    except ValueError as exc:  # includes JSONDecodeError
        raise WeatherSourceError(f"weather API returned non-JSON content: {exc}") from exc
    if not isinstance(payload, dict) or "hourly" not in payload:
        raise WeatherSourceError("weather API response has no hourly block")
    return payload


class _LocalHourly:
    """Hourly series re-keyed by station-local naive timestamps 'YYYY-MM-DDTHH:00'."""

    def __init__(self, payload: Mapping[str, Any], station: str):
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        tz = ZoneInfo(TIMEZONES[station])
        self.rows: dict[str, dict[str, Any]] = {}
        for i, label in enumerate(times):
            utc = dt.datetime.fromisoformat(label).replace(tzinfo=dt.timezone.utc)
            local = utc.astimezone(tz).replace(tzinfo=None)
            row = {}
            for key in HOURLY:
                values = hourly.get(key) or []
                v = values[i] if i < len(values) else None
                row[key] = None if v is None or (isinstance(v, float) and math.isnan(v)) else v
            self.rows[local.strftime("%Y-%m-%dT%H:00")] = row

    def at(self, day: dt.date, hour: int, key: str):
        return self.rows.get(f"{day.isoformat()}T{hour:02d}:00", {}).get(key)

    def window(self, start: dt.datetime, end: dt.datetime, key: str) -> list[float]:
        """Values whose (preceding-hour) label t satisfies start < t <= end."""
        out = []
        t = start + dt.timedelta(hours=1)
        while t <= end:
            v = self.rows.get(t.strftime("%Y-%m-%dT%H:00"), {}).get(key)
            if v is not None:
                out.append(v)
            t += dt.timedelta(hours=1)
        return out


def map_payload(payload: Mapping[str, Any], station: str, date: dt.date,
                evaporation: EvaporationMode = "missing") -> dict[str, Any]:
    """Build the model's record for `date` from an hourly UTC payload."""
    _check_station(station)
    _check_evaporation(evaporation)
    series = _LocalHourly(payload, station)
    nine_today = dt.datetime.combine(date, dt.time(9))
    nine_yesterday = nine_today - dt.timedelta(days=1)
    day_start = dt.datetime.combine(date, dt.time(0))
    day_end = day_start + dt.timedelta(hours=23)

    def agg(fn, values):
        return fn(values) if values else None

    rainfall = agg(sum, series.window(nine_yesterday, nine_today, "precipitation"))
    min_temp = agg(min, series.window(nine_yesterday, nine_today, "temperature_2m"))
    max_temp = agg(max, series.window(day_start - dt.timedelta(hours=1), day_end, "temperature_2m"))
    sunshine = agg(sum, series.window(day_start - dt.timedelta(hours=1), day_end, "sunshine_duration"))
    et0 = agg(sum, series.window(nine_yesterday, nine_today, "et0_fao_evapotranspiration"))

    gust_speed, gust_dir = None, None
    for hour in range(24):
        g = series.at(date, hour, "wind_gusts_10m")
        if g is not None and (gust_speed is None or g > gust_speed):
            gust_speed, gust_dir = g, degrees_to_compass(series.at(date, hour, "wind_direction_10m"))

    at = series.at
    record = {
        "Date": date.isoformat(),
        "Location": station,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": None if rainfall is None else round(rainfall, 2),
        "Evaporation": (None if et0 is None else round(et0, 2)) if evaporation == "et0" else None,
        "Sunshine": None if sunshine is None else round(sunshine / 3600.0, 2),
        "WindGustDir": gust_dir,
        "WindGustSpeed": gust_speed,
        "WindDir9am": degrees_to_compass(at(date, 9, "wind_direction_10m")),
        "WindDir3pm": degrees_to_compass(at(date, 15, "wind_direction_10m")),
        "WindSpeed9am": at(date, 9, "wind_speed_10m"),
        "WindSpeed3pm": at(date, 15, "wind_speed_10m"),
        "Humidity9am": at(date, 9, "relative_humidity_2m"),
        "Humidity3pm": at(date, 15, "relative_humidity_2m"),
        "Pressure9am": at(date, 9, "pressure_msl"),
        "Pressure3pm": at(date, 15, "pressure_msl"),
        "Cloud9am": percent_to_oktas(at(date, 9, "cloud_cover")),
        "Cloud3pm": percent_to_oktas(at(date, 15, "cloud_cover")),
        "Temp9am": at(date, 9, "temperature_2m"),
        "Temp3pm": at(date, 15, "temperature_2m"),
        "RainToday": None if rainfall is None else ("Yes" if rainfall > 1.0 else "No"),
    }
    core = (record["Temp9am"], record["Pressure9am"], record["MaxTemp"], record["Rainfall"])
    if all(v is None for v in core):
        raise NoDataError(f"no observations for {station} on {date.isoformat()} (archive lag?)")
    return record


# ------------------------------------------------------------------ public
def fetch_day_with_source(station: str, date: dt.date | dt.datetime | str, *,
                          evaporation: EvaporationMode = "missing", timeout: float = 15.0,
                          today: dt.date | None = None) -> tuple[dict[str, Any], str]:
    """Like fetch_day, also returning which endpoint served the data:
    'archive' (ERA5 reanalysis) or 'forecast' (observations + short-range
    forecast for the hours not yet elapsed)."""
    _check_station(station)
    _check_evaporation(evaporation)
    date = _coerce_date(date)
    today = today or station_today(station)
    if date > today:
        raise WeatherSourceError("date is in the future; the model needs the day's observations")
    age = (today - date).days
    params = _params(station, date, date)
    if age >= ARCHIVE_DELAY_DAYS:
        try:
            return map_payload(_request(ARCHIVE_URL, params, timeout), station, date, evaporation), "archive"
        except NoDataError:
            if age > FORECAST_PAST_LIMIT_DAYS:
                raise
    return map_payload(_request(FORECAST_URL, params, timeout), station, date, evaporation), "forecast"


def fetch_day(station: str, date: dt.date | dt.datetime | str, *,
              evaporation: EvaporationMode = "missing", timeout: float = 15.0,
              today: dt.date | None = None) -> dict[str, Any]:
    """Return the model's 22-column record for `station` on `date` (station-local).

    Dates older than ~7 days come from the archive; more recent ones (and
    today) from the forecast endpoint. If the archive has no data yet, the
    forecast endpoint is tried as a fallback (see fetch_day_with_source).
    """
    return fetch_day_with_source(station, date, evaporation=evaporation, timeout=timeout, today=today)[0]


def fetch_range(station: str, start: dt.date, end: dt.date, *,
                evaporation: EvaporationMode = "missing", timeout: float = 60.0) -> list[dict[str, Any]]:
    """Archive-only bulk fetch (one request) for evaluation scripts.

    Days without data are skipped with a warning; any other failure raises.
    """
    _check_station(station)
    _check_evaporation(evaporation)
    start, end = _coerce_date(start), _coerce_date(end)
    payload = _request(ARCHIVE_URL, _params(station, start, end), timeout)
    records, skipped = [], []
    day = start
    while day <= end:
        try:
            records.append(map_payload(payload, station, day, evaporation))
        except NoDataError:
            skipped.append(day.isoformat())
        day += dt.timedelta(days=1)
    if skipped:
        warnings.warn(f"{station}: {len(skipped)} day(s) without data skipped ({skipped[0]} .. {skipped[-1]})")
    return records
