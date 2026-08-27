"""Fast unit tests that only need the shipped model/ directory (no source repo)."""

import numpy as np
import pandas as pd
import pytest

from rainapp import RainPredictor, load_default_predictor
from rainapp.model import NumpyMLP, _sigmoid


@pytest.fixture(scope="module")
def predictor() -> RainPredictor:
    return load_default_predictor()


def test_sigmoid_is_stable_and_correct():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = _sigmoid(x)
    assert np.all(np.isfinite(out))
    assert out[2] == 0.5 and out[0] == 0.0 and out[4] == 1.0
    assert np.isclose(out[3], 1 / (1 + np.exp(-1)))


def test_network_shape_and_size(predictor):
    net = predictor.network
    assert isinstance(net, NumpyMLP)
    assert net.input_dim == 80
    assert net.parameter_count == 3909


def test_full_record_predicts_in_unit_interval(predictor):
    record = {
        "Date": "2008-12-01", "Location": "Albury", "MinTemp": 13.6, "MaxTemp": 22.5,
        "Rainfall": 0.6, "Evaporation": None, "Sunshine": None, "WindGustDir": "W",
        "WindGustSpeed": 45.0, "WindDir9am": "W", "WindDir3pm": "WNW", "WindSpeed9am": 19.0,
        "WindSpeed3pm": 25.0, "Humidity9am": 68.8, "Humidity3pm": 23.8, "Pressure9am": 1008.1,
        "Pressure3pm": 1007.2, "Cloud9am": 8.0, "Cloud3pm": None, "Temp9am": 16.8,
        "Temp3pm": 20.8, "RainToday": "No",
    }
    out = predictor.predict_one(record)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["rain_tomorrow"] in {"Yes", "No"}
    assert out["threshold"] == pytest.approx(0.5957959890365601)


def test_missing_columns_are_tolerated(predictor):
    """Only Date and Location are given; every other column is absent, not NaN."""
    out = predictor.predict_one({"Date": "2015-06-10", "Location": "Sydney"})
    assert 0.0 <= out["probability"] <= 1.0


def test_unknown_location_and_unparseable_date_do_not_crash(predictor):
    out = predictor.predict_one({"Date": "not a date", "Location": "Atlantis", "MinTemp": 10})
    assert 0.0 <= out["probability"] <= 1.0


def test_batch_matches_single(predictor):
    a = {"Date": "2015-06-10", "Location": "Sydney", "Humidity3pm": 90, "RainToday": "Yes"}
    b = {"Date": "2015-01-10", "Location": "AliceSprings", "Humidity3pm": 10, "RainToday": "No"}
    batch = predictor.predict_frame(pd.DataFrame([a, b]))
    assert batch["probability"].iloc[0] == pytest.approx(predictor.predict_one(a)["probability"], abs=1e-9)
    assert batch["probability"].iloc[1] == pytest.approx(predictor.predict_one(b)["probability"], abs=1e-9)
    assert batch["probability"].iloc[0] > batch["probability"].iloc[1]


def test_empty_input_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict_frame(pd.DataFrame())


# ---- behaviours pinned after code review -----------------------------------

SYDNEY = {"Date": "2015-06-10", "Location": "Sydney", "Humidity3pm": 90, "RainToday": "Yes"}


def test_readme_example_value(predictor):
    assert predictor.predict_one(SYDNEY)["probability"] == pytest.approx(0.7702456116676331, abs=1e-6)


def test_duplicate_index_does_not_contaminate_rows(predictor):
    b = {"Date": "2015-01-10", "Location": "AliceSprings", "Humidity3pm": 10, "RainToday": "No"}
    dup = pd.concat([pd.DataFrame([SYDNEY]), pd.DataFrame([b])])  # index [0, 0]
    out = predictor.predict_frame(dup)
    assert list(out.index) == [0, 0]
    assert out["probability"].iloc[0] == pytest.approx(predictor.predict_one(SYDNEY)["probability"], abs=1e-9)


def test_categoricals_are_normalised_or_imputed(predictor):
    p_yes = predictor.predict_one(SYDNEY)["probability"]
    assert predictor.predict_one({**SYDNEY, "RainToday": " yes "})["probability"] == pytest.approx(p_yes, abs=1e-9)
    p_missing = predictor.predict_one({**SYDNEY, "RainToday": None})["probability"]
    assert predictor.predict_one({**SYDNEY, "RainToday": "maybe"})["probability"] == pytest.approx(p_missing, abs=1e-9)
    assert predictor.predict_one({**SYDNEY, "WindDir3pm": "wnw"})["probability"] == pytest.approx(
        predictor.predict_one({**SYDNEY, "WindDir3pm": "WNW"})["probability"], abs=1e-9)


def test_location_case_insensitive(predictor):
    assert predictor.predict_one({**SYDNEY, "Location": "sydney"})["probability"] == pytest.approx(
        predictor.predict_one(SYDNEY)["probability"], abs=1e-9)


def test_date_and_location_required(predictor):
    with pytest.raises(ValueError, match="Date and Location"):
        predictor.predict_one({"Location": None, "Date": None})
    with pytest.raises(ValueError):
        predictor.predict_one({})


def test_non_scalar_values_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict_one({**SYDNEY, "MinTemp": [1, 2]})
    with pytest.raises(ValueError):
        predictor.predict_proba({"Date": ["2015-06-10"], "Location": ["Sydney"]})


def test_unparseable_numeric_is_treated_as_missing(predictor):
    assert predictor.predict_one({**SYDNEY, "MinTemp": "13.6C"})["probability"] == pytest.approx(
        predictor.predict_one({**SYDNEY, "MinTemp": None})["probability"], abs=1e-9)


def test_sigmoid_accepts_int_input():
    out = _sigmoid(np.array([1, -1]))
    assert out.dtype.kind == "f" and np.isclose(out[0], 0.7310586)


def test_rainapp_importable_without_sys_path_hack():
    import sys
    assert "weather_preprocessing" in sys.modules
    assert sys.modules["weather_preprocessing"].__name__ == "rainapp.weather_preprocessing"
