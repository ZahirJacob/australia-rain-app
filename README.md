# australia-rain-app

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
| `weather_preprocessing.py` | verbatim copy of `src/weather_preprocessing.py`; required under that module name to unpickle the preprocessor |
| `rainapp/` | new: numpy forward pass, input coercion, single-load predictor |
| `scripts/verify_parity.py` | new: proves the numpy path reproduces the Keras model exactly |

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
# {'rain_tomorrow': 'Yes', 'probability': 0.8..., 'threshold': 0.5957959890365601}
```

Any of the 22 input columns may be missing or `None`; the preprocessor
imputes them exactly as it did in training. `Date` (→ season) and `Location`
(→ climatic region) are the only fields that materially need a value.

## Development

```
pip install -r requirements-dev.txt
pytest
python scripts/verify_parity.py --source ../australia-rain-prediction
```

## Roadmap

1. ~~TensorFlow-free inference with verified parity~~
2. Weather-data adapter (Open-Meteo) so users only pick a station and a date
3. Gradio UI, deployed on Hugging Face Spaces
