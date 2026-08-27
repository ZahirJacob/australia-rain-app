"""Pure-numpy forward pass of the frozen Keras MLP (nn_config_5).

Architecture (from config.json inside selected_nn_model.keras):
    Input(80) -> Dense(30, relu) -> Dropout -> Dense(26, relu) -> Dropout
             -> Dense(24, relu) -> Dense(1, sigmoid)
Dropout is the identity at inference time, so it is simply omitted here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LAYER_NAMES = ("dense", "dense_1", "dense_2", "dense_3")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable: never exponentiates a large positive number.
    positive = x >= 0
    out = np.empty_like(x)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


class NumpyMLP:
    """Holds the 4 (weight, bias) pairs and evaluates the network."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]]):
        if len(layers) != 4:
            raise ValueError(f"expected 4 dense layers, got {len(layers)}")
        self.layers = [(np.asarray(w, dtype=np.float32), np.asarray(b, dtype=np.float32)) for w, b in layers]
        self.input_dim = self.layers[0][0].shape[0]

    @classmethod
    def from_npz(cls, path: str | Path) -> "NumpyMLP":
        with np.load(path) as z:
            layers = [(z[f"{name}/kernel"], z[f"{name}/bias"]) for name in LAYER_NAMES]
        return cls(layers)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return P(RainTomorrow=Yes) for each row of x (shape [n, 80])."""
        h = np.asarray(x, dtype=np.float32)
        if h.ndim != 2 or h.shape[1] != self.input_dim:
            raise ValueError(f"expected shape [n, {self.input_dim}], got {h.shape}")
        *hidden, last = self.layers
        for w, b in hidden:
            h = _relu(h @ w + b)
        w, b = last
        return _sigmoid(h @ w + b).reshape(-1)

    @property
    def parameter_count(self) -> int:
        return int(sum(w.size + b.size for w, b in self.layers))
