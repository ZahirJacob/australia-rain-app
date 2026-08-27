# australia-rain-app

**Live demo:** https://australia-rain-app.onrender.com (free tier: the first request after 15 idle minutes takes about a minute to wake the server)

Web application for next-day rain prediction in Australia, serving the model
trained in the academic project
[australia-rain-prediction](https://github.com/ZahirJacob/australia-rain-prediction)
**without TensorFlow**.

**Author:** Zahir Jacob. This application is personal follow-up work and is
not part of the academic project.

## Provenance

The classifier (`nn_config_5`, a 3,909-parameter MLP, decision threshold
0.5958) and its preprocessing pipeline were developed, selected and evaluated
in the academic project *AA1 – TUIA* by **Dimenna, Jacob and Taborda**
(final test F1 = 0.653, ROC-AUC = 0.885; expanding-window temporal F1 = 0.640).
That project is the authority on methodology and metrics.

What this repository adds is entirely downstream of those frozen artifacts:

| Component | Origin |
|---|---|
| `model/preprocessor.joblib` | byte-identical copy of `artifacts/selected_nn_preprocessor.joblib` (SHA-256 `4bca0f30…`) |
| `model/nn_weights.npz` | the 8 layer tensors extracted from `artifacts/selected_nn_model.keras` (SHA-256 `041251e4…`) by `scripts/export_model.py` |
| `rainapp/weather_preprocessing.py` | verbatim copy of `src/weather_preprocessing.py`; registered in `sys.modules` under the original top-level name so the pickled preprocessor resolves its class |
| `rainapp/` | new: numpy forward pass, input coercion, single-load predictor |
| `scripts/verify_parity.py` | new: proves the numpy path reproduces the Keras model exactly |
| `rainapp/weather_source.py`, `rainapp/stations.py` | new: Open-Meteo adapter and the 49 station coordinates |
| `scripts/evaluate_api_shift.py` | new: measures the BoM → Open-Meteo input shift |

`model/manifest.json` records the SHA-256 of every source artifact.

## Parity with the original model

`scripts/verify_parity.py` (needs a checkout of the academic repo) checks:

* the shipped preprocessor hash matches the academic repo's frozen hash chain;
* on the 512-row parity sample and on the full 28,431-row final test set, the
  numpy network reproduces the Keras probabilities within `rtol=1e-6, atol=1e-7`
  and **every label is identical**; the published confusion matrix
  `[[19661, 2396], [2121, 4253]]` is reproduced exactly.

Result of the last run: max |Δp| = 4.8e-07 over the test set, all labels equal.

## Usage

```python
from rainapp import load_default_predictor

predictor = load_default_predictor()          # loads artifacts once
predictor.predict_one({"Date": "2015-06-10", "Location": "Sydney",
                       "Humidity3pm": 90, "RainToday": "Yes"})
# {'rain_tomorrow': 'Yes', 'probability': 0.7702456116676331, 'threshold': 0.5957959890365601}
```

Any of the 22 input columns may be missing or `None`; the preprocessor
imputes them with the statistics it learned in training. `Date` (→ season) and
`Location` (→ climatic region) are the only fields that materially need a value.

Input caveats (inherited from the original pipeline, which does the same):

* `Date` and `Location` are required in every row (ValueError otherwise).
* `Location` is matched case-insensitively against the 49 station names;
  unknown names are accepted and silently fall back to the default region.
* `RainToday` (`Yes`/`No`) and the wind directions (16 compass points) are
  normalised for case/whitespace; any other value is treated as missing and
  imputed, never fed to the model as an unseen category.
* Numeric fields are coerced; unparseable text (e.g. `"13.6C"`) becomes
  missing. Lists or other non-scalar cells raise ValueError.
* **Imputation of `Evaporation`, `Cloud9am` and `Cloud3pm` is KNN-based and
  can depend on batch composition**: for ~4–5% of rows the probability differs
  (by up to ~0.09) between predicting the row alone and inside a larger batch,
  because equidistant training neighbours are tie-broken differently. A single
  row is always deterministic. The parity check above uses the same batching
  as the academic evaluation.

## Weather data source (Open-Meteo)

Nobody will type 20 Bureau-of-Meteorology-style observations into a form, so
`rainapp.weather_source.fetch_day(station, date)` builds the model's input
record from [Open-Meteo](https://open-meteo.com) (free, no API key): archive
endpoint for past dates, forecast endpoint for the last week and today.

```python
from rainapp import load_default_predictor
from rainapp.weather_source import fetch_day

record = fetch_day("Sydney", "2016-06-10")
load_default_predictor().predict_one(record)
# {'rain_tomorrow': 'No', 'probability': 0.260..., 'threshold': 0.5957959890365601}
```

The adapter reproduces BoM daily-observation semantics, which the training
data uses: `Rainfall`, `MinTemp` and `Evaporation` are the 24 h **ending 09:00
local**, `MaxTemp`/`Sunshine`/gusts the local calendar day, `*9am`/`*3pm` the
local 09:00/15:00 hours. This matters because `RainTomorrow(D)` equals
`RainToday(D+1)` in the dataset, i.e. the label window starts at 09:00 today —
a midnight-to-midnight rainfall sum would leak part of it into the features.
All aggregates are computed from hourly data requested in UTC and converted
to true local time (Open-Meteo applies one fixed offset per response, with no
daylight-saving changes, so its "local" hours are off by one for half the
year in NSW/VIC/TAS/SA/ACT).

Open-Meteo values are gridded reanalysis/model estimates, not the instrument
readings the model was trained on. `scripts/evaluate_api_shift.py` measures
the cost on 12 stations × every day of 2016 (4,214 days), predicting the same
days from BoM observations and from Open-Meteo:

| Input source | F1 (positive) | Precision | Recall |
|---|---|---|---|
| BoM observations (native) | 0.664 | 0.673 | 0.656 |
| Open-Meteo, `Evaporation` missing (default) | 0.624 | 0.599 | 0.651 |
| Open-Meteo, `Evaporation` = ET0 | 0.628 | 0.606 | 0.652 |

Labels agree between the two sources on 88% of days; the API costs about
0.04 F1, almost entirely as extra false positives. Per station (F1 BoM → API):
Perth 0.79 → 0.75, Adelaide 0.77 → 0.75, Melbourne 0.67 → 0.67, Albury
0.69 → 0.65, Sydney 0.64 → 0.56, Brisbane 0.69 → 0.57, Darwin 0.68 → 0.53,
Canberra 0.55 → 0.62.

Where the two sources disagree most: `Rainfall` (correlation 0.59 — gridded
precipitation vs a rain gauge), the three wind directions (exact 16-point
match only ~30%), and wind speeds (corr ≈ 0.5). Pressure (corr 0.996) and
temperatures (0.97–0.98) transfer almost perfectly. Pan `Evaporation` has no
Open-Meteo equivalent and is left missing by default (ET0 is a different
quantity; using it anyway gains +0.004 F1, within noise).

Caveat: 2016 overlaps the academic project's development/test split, so these
are not held-out accuracies — only the *gap* between input sources is the
point. Full numbers: `artifacts/api_shift_2016_evap_missing.json` and
`..._et0.json`.

## Deployment

`render.yaml` is a [Render Blueprint](https://render.com/docs/blueprint-spec):
a free web service built from `requirements.txt`, started with
`GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=$PORT python app.py`, and
redeployed automatically on every push to `main`. Measured footprint: ~210 MB
RSS (free instances have 512 MB). Hugging Face Spaces was the original target,
but Gradio Spaces now require a paid plan (free accounts older than 30 days may
host two on ZeroGPU hardware — a possible later move).

## Development

```
pip install -r requirements-dev.txt
pytest
python scripts/verify_parity.py --source ../australia-rain-prediction
```

## Roadmap

1. ~~TensorFlow-free inference with verified parity~~
2. ~~Weather-data adapter (Open-Meteo) so users only pick a station and a date~~
3. ~~Gradio UI~~ — deployed on Render's free tier, auto-deployed from `main`
