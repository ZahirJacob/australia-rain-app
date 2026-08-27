"""Gradio UI for the Australian rain predictor.

Run locally:  python app.py   (then open http://127.0.0.1:7860)
On Hugging Face Spaces this file is the entry point.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import gradio as gr
import pandas as pd

from rainapp import load_default_predictor
from rainapp.predictor import INPUT_COLUMNS, NUMERIC_COLUMNS
from rainapp.stations import STATION_NAMES
from rainapp.weather_source import COMPASS, WeatherSourceError, fetch_day, station_today

PREDICTOR = load_default_predictor()
THRESHOLD = PREDICTOR.threshold

FIELD_HELP = {
    "MinTemp": "°C, 24 h to 9 am", "MaxTemp": "°C, calendar day", "Rainfall": "mm, 24 h to 9 am",
    "Evaporation": "mm (pan), 24 h to 9 am", "Sunshine": "hours", "WindGustDir": "16-point compass",
    "WindGustSpeed": "km/h", "WindDir9am": "compass", "WindDir3pm": "compass",
    "WindSpeed9am": "km/h", "WindSpeed3pm": "km/h", "Humidity9am": "%", "Humidity3pm": "%",
    "Pressure9am": "hPa (MSL)", "Pressure3pm": "hPa (MSL)", "Cloud9am": "oktas 0-8",
    "Cloud3pm": "oktas 0-8", "Temp9am": "°C", "Temp3pm": "°C", "RainToday": "Yes/No (> 1 mm)",
}
EDITABLE = [c for c in INPUT_COLUMNS if c not in ("Date", "Location")]
DIRECTION_FIELDS = ("WindGustDir", "WindDir9am", "WindDir3pm")


# ------------------------------------------------------------------ helpers
def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def _record_to_table(record: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"Field": EDITABLE, "Value": [_fmt(record[c]) for c in EDITABLE],
                         "Unit / meaning": [FIELD_HELP[c] for c in EDITABLE]})


def _table_to_record(table: pd.DataFrame, station: str, date: str) -> dict[str, Any]:
    values = dict(zip(table["Field"], table["Value"]))
    record: dict[str, Any] = {"Date": date, "Location": station}
    for column in EDITABLE:
        v = values.get(column, "")
        v = None if v is None or str(v).strip() == "" else str(v).strip()
        record[column] = v
    return record


def _result(record: dict[str, Any]) -> tuple[dict[str, float], str]:
    """Label value + markdown verdict.

    The verdict must come from the model's threshold (59.6%), not from which
    probability is larger, so the Label shows a single bar for P(rain) and the
    Yes/No decision is stated in the markdown.
    """
    out = PREDICTOR.predict_one(record)
    p = out["probability"]
    if out["rain_tomorrow"] == "Yes":
        verdict = "🌧️ Rain expected tomorrow"
    else:
        verdict = "☀️ No rain expected tomorrow"
    summary = (f"## {verdict}\n"
               f"P(rain tomorrow) = **{p:.1%}** vs decision threshold **{THRESHOLD:.1%}** "
               f"(the threshold was chosen to maximise F1 on out-of-fold data, which is why it is above 50%).")
    return {"P(rain tomorrow)": p}, summary


# ------------------------------------------------------------------ actions
def fetch_and_predict(station: str, date: str):
    try:
        record = fetch_day(station, date)
    except WeatherSourceError as exc:
        raise gr.Error(str(exc)) from exc
    label, summary = _result(record)
    today = station_today(station)
    age = (today - dt.date.fromisoformat(record["Date"])).days
    source = ("Open-Meteo archive (ERA5 reanalysis)" if age >= 7
              else "Open-Meteo forecast endpoint (observations + short-range forecast for hours not yet elapsed)")
    note = (f"Inputs for **{station}** on **{record['Date']}** from {source}. "
            "Edit any value below and press *Re-predict with edited inputs*. "
            "Evaporation is left empty on purpose (no equivalent in the API) and is imputed by the model.")
    return label, summary, _record_to_table(record), note


def repredict(station: str, date: str, table: pd.DataFrame):
    try:
        label, summary = _result(_table_to_record(table, station, date))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return label, summary


def manual_predict(station, date, *values):
    record = {"Date": date, "Location": station}
    for column, v in zip(EDITABLE, values):
        record[column] = None if v is None or (isinstance(v, str) and not v.strip()) else v
    try:
        return _result(record)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def default_date(station: str) -> str:
    return station_today(station).isoformat()


# ------------------------------------------------------------------ layout
ABOUT = """
## What this is
A demo of the classifier developed in the academic project
[australia-rain-prediction](https://github.com/ZahirJacob/australia-rain-prediction)
(Dimenna, Jacob, Taborda): a small neural network (`nn_config_5`, 3,909 parameters)
predicting whether **more than 1 mm of rain** falls in the 24 h from 9 am tomorrow
at one of 49 Australian weather stations, from today's observations.
This app, the TensorFlow-free inference and the weather-API integration are
personal follow-up work by Zahir Jacob — [source](https://github.com/ZahirJacob/australia-rain-app).

## How good is it?
| Evaluation | F1 (rain) | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Final held-out test, BoM observations (28,431 days) | 0.653 | 0.640 | 0.667 | 0.885 |
| Expanding-window temporal check (train on past, predict future) | 0.640 | 0.612 | 0.670 | 0.871 |
| Inputs from Open-Meteo instead of BoM instruments (12 stations × 2016) | 0.624 | 0.599 | 0.651 | — |

Roughly: when it says rain, it is right about 6 times in 10; it catches about
2 in 3 rainy days. Feeding it from the free weather API (what this app does)
costs about 0.04 F1, mostly as extra rain warnings.

## Honest caveats
* This is **not** a better forecast than the weather service — Open-Meteo itself
  gives you a rain forecast. It demonstrates a trained classifier on real inputs.
* The model was trained on 2007–2017 Bureau of Meteorology station readings.
  Open-Meteo values are gridded model/reanalysis estimates; rainfall and wind
  direction transfer worst, pressure and temperature best.
* For "today", some hours have not happened yet and come from Open-Meteo's
  short-range forecast.
* Predictions are exactly those of the frozen academic model (verified on all
  28,431 test rows); nothing was retrained.
"""

with gr.Blocks(title="Australia rain tomorrow") as demo:
    gr.Markdown("# Will it rain tomorrow?\nPick an Australian weather station and a date; today's observations are fetched automatically.")

    with gr.Tab("Station & date"):
        with gr.Row():
            station = gr.Dropdown(choices=list(STATION_NAMES), value="Sydney", label="Station")
            date = gr.Textbox(value=lambda: default_date("Sydney"), label="Date (YYYY-MM-DD, station local)")
            go = gr.Button("Fetch weather & predict", variant="primary")
        with gr.Row():
            label = gr.Label(label="Probability of rain tomorrow")
            summary = gr.Markdown()
        note = gr.Markdown()
        table = gr.Dataframe(headers=["Field", "Value", "Unit / meaning"], datatype=["str", "str", "str"],
                             interactive=True, label="Model inputs (editable)", row_count=(len(EDITABLE), "fixed"))
        redo = gr.Button("Re-predict with edited inputs")

        station.change(default_date, station, date)
        go.click(fetch_and_predict, [station, date], [label, summary, table, note])
        redo.click(repredict, [station, date, table], [label, summary])

    with gr.Tab("Manual entry"):
        gr.Markdown("Enter observations yourself (BoM conventions). Leave anything unknown blank — it will be imputed.")
        with gr.Row():
            m_station = gr.Dropdown(choices=list(STATION_NAMES), value="Sydney", label="Station")
            m_date = gr.Textbox(value=lambda: default_date("Sydney"), label="Date (YYYY-MM-DD)")
        inputs = []
        with gr.Row():
            for chunk in (EDITABLE[:7], EDITABLE[7:14], EDITABLE[14:]):
                with gr.Column():
                    for column in chunk:
                        if column in DIRECTION_FIELDS:
                            inputs.append(gr.Dropdown(choices=[""] + list(COMPASS), value="", label=f"{column} ({FIELD_HELP[column]})"))
                        elif column == "RainToday":
                            inputs.append(gr.Dropdown(choices=["", "Yes", "No"], value="", label=f"{column} ({FIELD_HELP[column]})"))
                        else:
                            inputs.append(gr.Textbox(value="", label=f"{column} ({FIELD_HELP[column]})"))
        m_go = gr.Button("Predict", variant="primary")
        with gr.Row():
            m_label = gr.Label(label="Probability of rain tomorrow")
            m_summary = gr.Markdown()
        m_go.click(manual_predict, [m_station, m_date, *inputs], [m_label, m_summary])

    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
