"""One-time export of the frozen model from the academic repo into model/.

What it does
------------
1. Reads `artifacts/selected_nn_model.keras` (a zip) from the source repo and
   pulls the 8 layer tensors out of `model.weights.h5` with h5py. TensorFlow /
   Keras are NOT needed. The 18 Adam optimizer tensors are ignored.
2. Copies the preprocessor .joblib byte-for-byte (so its SHA-256 still matches
   the academic repo's hash chain) and reads the decision threshold from
   oof_selection.json.
3. Writes model/manifest.json with the SHA-256 of every source artifact, so the
   provenance of what the app serves is verifiable, not just stated.

Usage: python scripts/export_model.py --source ../australia-rain-prediction
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np

LAYER_NAMES = ("dense", "dense_1", "dense_2", "dense_3")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_layers(keras_path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(keras_path) as z:
        config = json.loads(z.read("config.json"))
        weights_bytes = z.read("model.weights.h5")
    tensors: dict[str, np.ndarray] = {}
    with h5py.File(io.BytesIO(weights_bytes), "r") as f:
        for name in LAYER_NAMES:
            group = f["layers"][name]["vars"]
            tensors[f"{name}/kernel"] = group["0"][()]
            tensors[f"{name}/bias"] = group["1"][()]
    # Sanity-check the architecture we hard-code in rainapp/model.py.
    layers = config["config"]["layers"]
    dense = [l for l in layers if l["class_name"] == "Dense"]
    activations = [l["config"]["activation"] for l in dense]
    if activations != ["relu", "relu", "relu", "sigmoid"]:
        raise RuntimeError(f"unexpected activations: {activations}")
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="path to the australia-rain-prediction repo")
    parser.add_argument("--dest", default=Path(__file__).resolve().parent.parent / "model", type=Path)
    args = parser.parse_args()

    artifacts = args.source / "artifacts"
    keras_path = artifacts / "selected_nn_model.keras"
    preprocessor_path = artifacts / "selected_nn_preprocessor.joblib"
    selection_path = artifacts / "oof_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    winner = selection["winner"]

    args.dest.mkdir(parents=True, exist_ok=True)
    tensors = extract_layers(keras_path)
    np.savez(args.dest / "nn_weights.npz", **tensors)
    shutil.copyfile(preprocessor_path, args.dest / "preprocessor.joblib")

    manifest = {
        "candidate_id": winner["candidate_id"],
        "family": winner["family"],
        "parameters": winner["parameters"],
        "threshold": winner["threshold"],
        "probability_semantics": "P(RainTomorrow=Yes)",
        "files": {"weights": "nn_weights.npz", "preprocessor": "preprocessor.joblib"},
        "source_repository": "https://github.com/ZahirJacob/australia-rain-prediction",
        "source_artifact_sha256": {
            "model": sha256(keras_path),
            "preprocessor": sha256(preprocessor_path),
            "selection": sha256(selection_path),
        },
        "exported_sha256": {
            "weights": sha256(args.dest / "nn_weights.npz"),
            "preprocessor": sha256(args.dest / "preprocessor.joblib"),
        },
        "parameter_count": int(sum(t.size for t in tensors.values())),
        "layer_shapes": {k: list(v.shape) for k, v in tensors.items()},
    }
    (args.dest / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
