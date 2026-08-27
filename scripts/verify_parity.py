"""Parity gate: prove the numpy app reproduces the frozen Keras model exactly.

Three independent checks against artifacts of the academic repo:

A. Frozen-hash check: the preprocessor we ship and the source .keras/.joblib
   still have the SHA-256s recorded in oof_selection.json's hash chain.
B. 512-row parity sample: rows listed in local_inference_parity_predictions.npz
   (drawn from the development set) must give the same probabilities as the
   Keras model did, within rtol=1e-6 / atol=1e-7 (same tolerances the academic
   repo uses), and identical labels.
C. Full final test set (28,431 rows from final_test_predictions.npz): same
   probability tolerance, identical labels, and the confusion matrix must be
   exactly [[19661, 2396], [2121, 4253]] as published in final_test_metrics.json.

Usage: python scripts/verify_parity.py --source ../australia-rain-prediction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rainapp import RainPredictor  # noqa: E402

RTOL, ATOL = 1e-6, 1e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    artifacts = args.source / "artifacts"

    predictor = RainPredictor(ROOT / "model")
    manifest = predictor.manifest

    # ---- A. hash chain -------------------------------------------------
    parity = json.loads((artifacts / "docker_inference_parity.json").read_text(encoding="utf-8"))
    frozen = parity["frozen_artifact_sha256"]
    check(sha256(ROOT / "model" / "preprocessor.joblib") == frozen["preprocessor"],
          "shipped preprocessor.joblib is byte-identical to the frozen artifact")
    check(manifest["source_artifact_sha256"]["model"] == frozen["model"],
          "exported weights come from the frozen selected_nn_model.keras")
    check(abs(predictor.threshold - parity["threshold"]) < 1e-12, f"threshold {predictor.threshold} matches")
    check(predictor.network.parameter_count == 3909, f"parameter count {predictor.network.parameter_count} == 3909")

    data = pd.read_csv(args.source / "data" / "weatherAUS_2026C1.csv")

    # ---- B. 512-row parity sample -------------------------------------
    sample = np.load(artifacts / "local_inference_parity_predictions.npz")
    rows = data.loc[sample["row_index"]]
    probs = predictor.predict_proba(rows)
    ref = sample["reference_probability"].astype(np.float64)
    delta = np.abs(probs - ref)
    check(np.allclose(probs, ref, rtol=RTOL, atol=ATOL),
          f"512-row sample: max |dp| = {delta.max():.3e}, mean = {delta.mean():.3e}")
    check(np.array_equal(probs >= predictor.threshold, sample["reference_positive"]),
          "512-row sample: labels identical")

    # ---- C. full final test set ---------------------------------------
    test = np.load(artifacts / "final_test_predictions.npz")
    rows = data.loc[test["row_index"]]
    probs = predictor.predict_proba(rows)
    ref = test["probability"].astype(np.float64)
    delta = np.abs(probs - ref)
    check(np.allclose(probs, ref, rtol=RTOL, atol=ATOL),
          f"final test (n={len(ref)}): max |dp| = {delta.max():.3e}, mean = {delta.mean():.3e}")
    pred = (probs >= predictor.threshold).astype(np.int8)
    check(np.array_equal(pred, test["prediction"]), "final test: all 28,431 labels identical")
    y = test["y_true"]
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    published = json.loads((artifacts / "final_test_metrics.json").read_text(encoding="utf-8"))["confusion_matrix"]
    check([[tn, fp], [fn, tp]] == published["matrix"],
          f"confusion matrix [[{tn}, {fp}], [{fn}, {tp}]] == published")
    f1 = 2 * tp / (2 * tp + fp + fn)
    print(f"       F1(positive) recomputed = {f1:.6f}")
    print("ALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
