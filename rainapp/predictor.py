"""High-level prediction API: raw weather observations -> rain probability."""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from . import weather_preprocessing as _wp
from .model import NumpyMLP

# The pickled preprocessor references its class as
# `weather_preprocessing.WeatherFeatureTransformer` (a top-level module name).
# Registering our vendored copy under that name lets pickle resolve it without
# touching sys.path, so `rainapp` stays a self-contained, installable package.
sys.modules.setdefault("weather_preprocessing", _wp)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "model"

NUMERIC_COLUMNS: tuple[str, ...] = tuple(_wp.NUMERIC_COLUMNS)
MODE_COLUMNS: tuple[str, ...] = tuple(_wp.MODE_COLUMNS)
INPUT_COLUMNS: tuple[str, ...] = ("Date", "Location", *NUMERIC_COLUMNS, *MODE_COLUMNS)
DROPPED_COLUMNS = ("Unnamed: 0", "RainTomorrow", "RainfallTomorrow", "Region")

COMPASS_POINTS = frozenset(
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
)
YES_NO = frozenset({"Yes", "No"})
# Vocabulary the OneHotEncoder was fitted on. Anything else would become an
# all-zero one-hot block never seen in training, so it is mapped to NaN and
# takes the imputation path instead.
CATEGORICAL_VOCAB: dict[str, frozenset[str]] = {
    "WindGustDir": COMPASS_POINTS,
    "WindDir9am": COMPASS_POINTS,
    "WindDir3pm": COMPASS_POINTS,
    "RainToday": YES_NO,
}


@dataclass(frozen=True)
class PredictionMetadata:
    candidate_id: str
    threshold: float
    probability_semantics: str = "P(RainTomorrow=Yes)"


def _normalise_category(value: Any, vocab: frozenset[str]) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    for candidate in vocab:
        if candidate.lower() == text.lower():
            return candidate
    return np.nan


class RainPredictor:
    """Loads the frozen bundle once; `predict_*` methods are then cheap."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.manifest = manifest
        self.threshold = float(manifest["threshold"])
        self.metadata = PredictionMetadata(manifest["candidate_id"], self.threshold)
        self.network = NumpyMLP.from_npz(model_dir / manifest["files"]["weights"])
        self.preprocessor = joblib.load(model_dir / manifest["files"]["preprocessor"])
        features = self.preprocessor.named_steps["weather_features"]
        self.known_locations: dict[str, str] = {
            name.lower(): name for name in features.location_coordinates
        }

    # ------------------------------------------------------------------ input
    def as_frame(self, records: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
        """Coerce a dict / list of dicts / DataFrame into the 22-column layout.

        * Absent columns are created as NaN: the preprocessor imputes missing
          values, so a missing column behaves like a column of missing values.
        * `Date` and `Location` must carry a value in every row (they drive
          Season and Region); otherwise ValueError.
        * `Location` is matched case-insensitively against the 49 stations.
          Unknown stations are accepted and fall back to the default region.
        * Categoricals are normalised (case/whitespace); values outside the
          training vocabulary become NaN and are imputed.
        * Numeric columns are coerced with `pd.to_numeric(errors="coerce")`;
          non-scalar cells (lists, dicts) raise ValueError.
        The caller's index is preserved on the output of predict_frame.
        """
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, Mapping):
            if any(isinstance(v, (list, tuple, dict, np.ndarray, pd.Series)) for v in records.values()):
                raise ValueError("a mapping must hold one record of scalar values; use a list of mappings or a DataFrame for several rows")
            frame = pd.DataFrame([records])
        else:
            frame = pd.DataFrame(list(records))
        frame = frame.drop(columns=list(DROPPED_COLUMNS), errors="ignore")
        for column in INPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[list(INPUT_COLUMNS)]
        if frame.empty:
            raise ValueError("no rows to predict")

        bad = frame[list(INPUT_COLUMNS)].map(lambda v: isinstance(v, (list, tuple, dict, np.ndarray)))
        if bad.to_numpy().any():
            raise ValueError(f"non-scalar values in columns: {sorted(bad.columns[bad.any()])}")

        missing_key = frame["Date"].isna() | frame["Location"].isna()
        if missing_key.any():
            raise ValueError(f"Date and Location are required; missing in rows {list(frame.index[missing_key])}")

        frame["Location"] = frame["Location"].map(
            lambda v: self.known_locations.get(str(v).strip().lower(), str(v).strip())
        )
        for column, vocab in CATEGORICAL_VOCAB.items():
            frame[column] = frame[column].map(lambda v, vocab=vocab: _normalise_category(v, vocab))
        for column in NUMERIC_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    # -------------------------------------------------------------- inference
    def predict_proba(self, records) -> np.ndarray:
        """P(RainTomorrow=Yes) per row, in input order.

        Note: the preprocessor's KNN imputation (Evaporation, Cloud9am,
        Cloud3pm) breaks distance ties in a batch-shape-dependent way, so a row
        with those fields missing can receive a slightly different probability
        alone vs. inside a batch (~4-5% of rows, max ~0.09 seen). A single row
        is deterministic. Inherited from the original pipeline.
        """
        frame = self.as_frame(records)
        # The preprocessor aligns by index label internally; a duplicate index
        # would silently cross-contaminate rows, so transform on a clean one.
        features = self.preprocessor.transform(frame.reset_index(drop=True))
        probabilities = self.network.predict_proba(features)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("model produced non-finite probabilities")
        return probabilities

    def predict_frame(self, records) -> pd.DataFrame:
        frame = self.as_frame(records)
        probabilities = self.predict_proba(frame)
        positive = probabilities >= self.threshold
        return pd.DataFrame(
            {
                "rain_tomorrow": np.where(positive, "Yes", "No"),
                "probability": probabilities.astype(float),
            },
            index=frame.index,
        )

    def predict_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        row = self.predict_frame(record).iloc[0]
        return {
            "rain_tomorrow": str(row["rain_tomorrow"]),
            "probability": float(row["probability"]),
            "threshold": self.threshold,
        }


_DEFAULT: RainPredictor | None = None
_LOCK = threading.Lock()


def load_default_predictor() -> RainPredictor:
    """Process-wide singleton so web apps load the artifacts exactly once."""
    global _DEFAULT
    if _DEFAULT is None:
        with _LOCK:
            if _DEFAULT is None:
                _DEFAULT = RainPredictor()
    return _DEFAULT
