"""UI texts in English and Spanish, and the rule that picks the language.

One dictionary per language with identical keys (tested). The language is
chosen once per page load from `?lang=` (explicit override) or the browser's
Accept-Language header; everything the page shows - labels, buttons, the
verdict, the source note, error messages - comes from here.
"""

from __future__ import annotations

import re

DEFAULT_LANGUAGE = "en"
LANGUAGES = ("en", "es")


def pick_language(accept_language: str | None, query_lang: str | None = None) -> str:
    """`?lang=` wins; otherwise the first supported language in Accept-Language."""
    if query_lang and query_lang.lower()[:2] in LANGUAGES:
        return query_lang.lower()[:2]
    for part in (accept_language or "").split(","):
        code = part.split(";")[0].strip().lower()[:2]
        if code in LANGUAGES:
            return code
    return DEFAULT_LANGUAGE


FIELD_HELP = {
    "en": {
        "MinTemp": "°C, 24 h to 9 am", "MaxTemp": "°C, calendar day", "Rainfall": "mm, 24 h to 9 am",
        "Evaporation": "mm (pan), 24 h to 9 am", "Sunshine": "hours", "WindGustDir": "16-point compass",
        "WindGustSpeed": "km/h", "WindDir9am": "compass", "WindDir3pm": "compass",
        "WindSpeed9am": "km/h", "WindSpeed3pm": "km/h", "Humidity9am": "%", "Humidity3pm": "%",
        "Pressure9am": "hPa (MSL)", "Pressure3pm": "hPa (MSL)", "Cloud9am": "oktas 0-8",
        "Cloud3pm": "oktas 0-8", "Temp9am": "°C", "Temp3pm": "°C", "RainToday": "Yes/No (> 1 mm)",
    },
    "es": {
        "MinTemp": "°C, 24 h hasta las 9 am", "MaxTemp": "°C, día calendario", "Rainfall": "mm, 24 h hasta las 9 am",
        "Evaporation": "mm (tanque), 24 h hasta las 9 am", "Sunshine": "horas", "WindGustDir": "rosa de 16 puntos",
        "WindGustSpeed": "km/h", "WindDir9am": "rosa de vientos", "WindDir3pm": "rosa de vientos",
        "WindSpeed9am": "km/h", "WindSpeed3pm": "km/h", "Humidity9am": "%", "Humidity3pm": "%",
        "Pressure9am": "hPa (nivel del mar)", "Pressure3pm": "hPa (nivel del mar)", "Cloud9am": "octas 0-8",
        "Cloud3pm": "octas 0-8", "Temp9am": "°C", "Temp3pm": "°C", "RainToday": "Yes/No (> 1 mm)",
    },
}

TEXTS = {
    "en": {
        "title": "# Will it rain tomorrow?\nPick an Australian weather station and a date; today's observations are fetched automatically.",
        "tab_station": "Station & date", "tab_manual": "Manual entry", "tab_about": "About",
        "station": "Station", "date_local": "Date (YYYY-MM-DD, station local)", "date": "Date (YYYY-MM-DD)",
        "fetch": "Fetch weather & predict", "prob_label": "Probability of rain tomorrow",
        "accordion": "Model inputs (auto-filled from the weather API — expand to inspect or edit)",
        "table_label": "Model inputs (editable)", "col_field": "Field", "col_value": "Value", "col_unit": "Unit / meaning",
        "repredict": "Re-predict with edited inputs",
        "manual_intro": "Enter observations yourself (BoM conventions). Leave anything unknown blank — it will be imputed.",
        "predict": "Predict",
        "verdict_rain": "🌧️ Rain expected tomorrow", "verdict_dry": "☀️ No rain expected tomorrow",
        "prob_line": "P(rain tomorrow) = **{p}** vs decision threshold **{thr}**{why}.",
        "why": " (chosen to maximise F1 on out-of-fold data, which is why it is not 50%)",
        "bar": "P(rain tomorrow)",
        "src_archive": "Open-Meteo archive (ERA5 reanalysis)",
        "src_forecast": "Open-Meteo forecast endpoint (observations + short-range forecast for hours not yet elapsed)",
        "src_browser": "{src} — fetched by your browser",
        "note": "Inputs for **{station}** on **{date}** from {src}. Expand *Model inputs* below to see or edit them and re-predict. Evaporation is left empty on purpose (no equivalent in the API) and is imputed by the model.",
        "err_no_table": "No inputs to re-predict from yet — press *Fetch weather & predict* first.",
        "err_date": "Date must be YYYY-MM-DD.",
        "err_future": "That date is in the future; the model needs the day's observations.",
        "err_nodata": "No observations available for {station} on {date} yet (the archive lags a few days).",
        "err_weather": "Weather service unavailable: {msg}",
        "lang_note": "🌐 Language follows your browser; force it with `?lang=en` or `?lang=es` in the address.",
    },
    "es": {
        "title": "# ¿Llueve mañana?\nElegí una estación meteorológica de Australia y una fecha; las observaciones del día se descargan solas.",
        "tab_station": "Estación y fecha", "tab_manual": "Carga manual", "tab_about": "Acerca de",
        "station": "Estación", "date_local": "Fecha (AAAA-MM-DD, hora local de la estación)", "date": "Fecha (AAAA-MM-DD)",
        "fetch": "Buscar el clima y predecir", "prob_label": "Probabilidad de lluvia mañana",
        "accordion": "Entradas del modelo (completadas desde la API del clima — desplegá para ver o editar)",
        "table_label": "Entradas del modelo (editables)", "col_field": "Campo", "col_value": "Valor", "col_unit": "Unidad / significado",
        "repredict": "Volver a predecir con las entradas editadas",
        "manual_intro": "Ingresá las observaciones vos mismo (convenciones del BoM). Dejá en blanco lo que no sepas: se imputa.",
        "predict": "Predecir",
        "verdict_rain": "🌧️ Se espera lluvia mañana", "verdict_dry": "☀️ No se espera lluvia mañana",
        "prob_line": "P(lluvia mañana) = **{p}** frente al umbral de decisión **{thr}**{why}.",
        "why": " (elegido para maximizar el F1 fuera de muestra; por eso no es 50 %)",
        "bar": "P(lluvia mañana)",
        "src_archive": "archivo de Open-Meteo (reanálisis ERA5)",
        "src_forecast": "endpoint de pronóstico de Open-Meteo (observaciones + pronóstico a corto plazo para las horas que aún no pasaron)",
        "src_browser": "{src} — descargado por tu navegador",
        "note": "Entradas para **{station}** el **{date}** desde {src}. Desplegá *Entradas del modelo* abajo para verlas o editarlas y volver a predecir. La evaporación se deja vacía a propósito (no existe en la API) y el modelo la imputa.",
        "err_no_table": "Todavía no hay entradas para volver a predecir: primero presioná *Buscar el clima y predecir*.",
        "err_date": "La fecha debe tener el formato AAAA-MM-DD.",
        "err_future": "Esa fecha está en el futuro; el modelo necesita las observaciones del día.",
        "err_nodata": "Todavía no hay observaciones para {station} el {date} (el archivo se actualiza con unos días de retraso).",
        "err_weather": "Servicio meteorológico no disponible: {msg}",
        "lang_note": "🌐 El idioma sigue al navegador; forzalo con `?lang=es` o `?lang=en` en la dirección.",
    },
}

ABOUT = {
    "en": """
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
""",
    "es": """
## Qué es esto
Una demo del clasificador desarrollado en el proyecto académico
[australia-rain-prediction](https://github.com/ZahirJacob/australia-rain-prediction)
(Dimenna, Jacob, Taborda): una red neuronal pequeña (`nn_config_5`, 3.909 parámetros)
que predice si caerán **más de 1 mm de lluvia** en las 24 h desde las 9 am de mañana
en una de 49 estaciones meteorológicas de Australia, a partir de las observaciones de hoy.
Esta aplicación, la inferencia sin TensorFlow y la integración con la API del clima
son trabajo personal posterior de Zahir Jacob — [código](https://github.com/ZahirJacob/australia-rain-app).

## ¿Qué tan bueno es?
| Evaluación | F1 (lluvia) | Precisión | Recall | ROC-AUC |
|---|---|---|---|---|
| Test final reservado, observaciones del BoM (28.431 días) | 0,653 | 0,640 | 0,667 | 0,885 |
| Validación temporal con ventana creciente (entrenar con el pasado, predecir el futuro) | 0,640 | 0,612 | 0,670 | 0,871 |
| Entradas desde Open-Meteo en lugar de instrumentos del BoM (12 estaciones × 2016) | 0,624 | 0,599 | 0,651 | — |

En criollo: cuando dice que llueve, acierta unas 6 de cada 10 veces; detecta
unos 2 de cada 3 días de lluvia. Alimentarlo desde la API gratuita (lo que hace
esta app) cuesta unos 0,04 de F1, sobre todo como avisos de lluvia de más.

## Advertencias honestas
* Esto **no** es un pronóstico mejor que el del servicio meteorológico: la propia
  Open-Meteo te da un pronóstico de lluvia. Es una demostración de un clasificador
  entrenado sobre entradas reales.
* El modelo se entrenó con lecturas de estaciones del Bureau of Meteorology de 2007 a 2017.
  Los valores de Open-Meteo son estimaciones de modelos/reanálisis en grilla; la lluvia y la
  dirección del viento son lo que peor se traslada; la presión y la temperatura, lo mejor.
* Para "hoy", algunas horas todavía no ocurrieron y salen del pronóstico a corto plazo de Open-Meteo.
* Las predicciones son exactamente las del modelo académico congelado (verificado en las
  28.431 filas de test); no se reentrenó nada.
""",
}


def t(lang: str, key: str, **kw: object) -> str:
    text = TEXTS.get(lang, TEXTS[DEFAULT_LANGUAGE])[key]
    return text.format(**kw) if kw else text


def is_supported(lang: str | None) -> bool:
    return bool(lang) and lang in LANGUAGES
