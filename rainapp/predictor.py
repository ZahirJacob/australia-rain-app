"""High-level prediction API: raw weather observations -> rain probability."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from .model import NumpyMLP

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PACKAGE_ROOT / "model"

# The pickled preprocessor references the class as
# `weather_preprocessing.WeatherFeatureTransformer`, i.e. a top-level module of
# that exact name must be importable. It lives at the repo root; make sure it
# is on sys.path even when rainapp is imported from elsewhere.
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from weather_preprocessing import MODE_COLUMNS, NUMERIC_COLUMNS  # noqa: E402

INPUT_COLUMNS: tuple[str, ...] = ("Date", "Location", *NUMERIC_COLUMNS, *MODE_COLUMNS)
DROPPED_COLUMNS = ("Unnamed: 0", "RainTomorrow", "RainfallTomorrow", "Region")


@dataclass(frozen=True)
class PredictionMetadata:
    candidate_id: str
    threshold: float
    probability_semantics: str = "P(RainTomorrow=Yes)"


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

    # ------------------------------------------------------------------ input
    @staticmethod
    def as_frame(records: pd.DataFrame | Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> pd.DataFrame:
        """Coerce a dict / list of dicts / DataFrame into the 22-column layout.

        Columns that are absent are created as NaN: the preprocessor was
        trained to impute missing *values*, so a missing column is treated the
        same as a column full of missing values. `Date` and `Location` are the
        only fields that must actually carry a value to get a meaningful
        prediction (they drive Season and Region).
        """
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, Mapping):
            frame = pd.DataFrame([records])
        else:
            frame = pd.DataFrame(list(records))
        if frame.empty:
            raise ValueError("no rows to predict")
        frame = frame.drop(columns=list(DROPPED_COLUMNS), errors="ignore")
        for column in INPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[list(INPUT_COLUMNS)]

    # -------------------------------------------------------------- inference
    def predict_proba(self, records) -> np.ndarray:
        """P(RainTomorrow=Yes) per row.

        Note: the preprocessor's KNN imputation (Evaporation, Cloud9am,
        Cloud3pm) breaks distance ties in a batch-shape-dependent way, so a row
        with those fields missing can receive a slightly different probability
        alone vs. inside a batch (~4-5% of rows, max ~0.09 seen). A single row
        is deterministic. Inherited from the original pipeline.
        """
        frame = self.as_frame(records)
        features = self.preprocessor.transform(frame)
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


def load_default_predictor() -> RainPredictor:
    """Process-wide singleton so web apps load the artifacts exactly once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = RainPredictor()
    return _DEFAULT
