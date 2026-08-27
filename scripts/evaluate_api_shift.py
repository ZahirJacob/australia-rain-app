"""Measure what feeding the model from Open-Meteo instead of BoM observations costs.

For each station and every day of the period that exists in the training CSV:
  * fetch the Open-Meteo archive record and predict from it;
  * predict from the CSV's own BoM observations (the model's native input);
  * compare both against the true RainTomorrow label.
Also reports per-feature agreement between the two input sources.

The period overlaps the academic project's development+test split, so the
absolute F1 numbers are NOT a held-out estimate; the point is the *gap*
between the two input sources on the same days.

Usage: python scripts/evaluate_api_shift.py --source ../australia-rain-prediction \
           --year 2016 --stations Sydney Melbourne ... --out artifacts/api_shift.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rainapp import RainPredictor, weather_source as ws  # noqa: E402
from rainapp.predictor import NUMERIC_COLUMNS  # noqa: E402

DEFAULT_STATIONS = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Hobart",
                    "Darwin", "Cairns", "AliceSprings", "Canberra", "Albury", "Townsville"]


def f1_report(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--stations", nargs="*", default=DEFAULT_STATIONS)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "api_shift.json")
    parser.add_argument("--evaporation", choices=["missing", "et0"], default="missing")
    args = parser.parse_args()

    data = pd.read_csv(args.source / "data" / "weatherAUS_2026C1.csv")
    data = data[data["RainTomorrow"].notna()]
    predictor = RainPredictor(ROOT / "model")
    start, end = dt.date(args.year, 1, 1), dt.date(args.year, 12, 31)

    per_station, api_rows, bom_rows, labels = {}, [], [], []
    for station in args.stations:
        bom = data[(data["Location"] == station) & (data["Date"] >= start.isoformat()) & (data["Date"] <= end.isoformat())]
        if bom.empty:
            print(f"{station}: no BoM rows in {args.year}, skipped"); continue
        api = pd.DataFrame(ws.fetch_range(station, start, end, evaporation=args.evaporation)).set_index("Date")
        common = bom[bom["Date"].isin(api.index)]
        api = api.loc[common["Date"].to_numpy()].reset_index()
        y = common["RainTomorrow"].map({"No": 0, "Yes": 1}).to_numpy()
        p_api = predictor.predict_proba(api); p_bom = predictor.predict_proba(common)
        thr = predictor.threshold
        per_station[station] = {
            "days": int(len(y)), "positive_rate": round(float(y.mean()), 3),
            "api": f1_report(y, (p_api >= thr).astype(int)),
            "bom": f1_report(y, (p_bom >= thr).astype(int)),
            "label_agreement_api_vs_bom": round(float(((p_api >= thr) == (p_bom >= thr)).mean()), 4),
            "mean_abs_prob_diff": round(float(np.abs(p_api - p_bom).mean()), 4),
        }
        api_rows.append(api); bom_rows.append(common); labels.append(y)
        print(f"{station:14s} n={len(y):3d}  F1 api={per_station[station]['api']['f1']:.3f}  "
              f"F1 bom={per_station[station]['bom']['f1']:.3f}  label agreement={per_station[station]['label_agreement_api_vs_bom']:.3f}")

    api_all = pd.concat(api_rows, ignore_index=True)
    bom_all = pd.concat(bom_rows, ignore_index=True)
    y_all = np.concatenate(labels)
    thr = predictor.threshold
    p_api, p_bom = predictor.predict_proba(api_all), predictor.predict_proba(bom_all)
    # feature-level agreement
    features = {}
    for col in NUMERIC_COLUMNS:
        a = pd.to_numeric(api_all[col], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(bom_all[col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        features[col] = {"mean_abs_diff": round(float(np.abs(a[m] - b[m]).mean()), 3) if m.any() else None,
                         "corr": round(float(np.corrcoef(a[m], b[m])[0, 1]), 3) if m.sum() > 2 else None,
                         "api_missing_rate": round(float(1 - np.isfinite(a).mean()), 3),
                         "bom_missing_rate": round(float(1 - np.isfinite(b).mean()), 3)}
    for col in ("WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"):
        a, b = api_all[col].to_numpy(dtype=object), bom_all[col].to_numpy(dtype=object)
        a_ok, b_ok = pd.notna(a), pd.notna(b)
        m = a_ok & b_ok
        features[col] = {"exact_match_rate": round(float((a[m] == b[m]).mean()), 3) if m.any() else None,
                         "api_missing_rate": round(float(1 - a_ok.mean()), 3),
                         "bom_missing_rate": round(float(1 - b_ok.mean()), 3)}
    result = {
        "year": args.year, "stations": args.stations, "days": int(len(y_all)), "evaporation": args.evaporation,
        "pooled": {"api": f1_report(y_all, (p_api >= thr).astype(int)), "bom": f1_report(y_all, (p_bom >= thr).astype(int)),
                   "label_agreement_api_vs_bom": round(float(((p_api >= thr) == (p_bom >= thr)).mean()), 4),
                   "mean_abs_prob_diff": round(float(np.abs(p_api - p_bom).mean()), 4)},
        "per_station": per_station, "features": features,
        "note": "period overlaps the academic dev/test split; compare api vs bom, do not read as held-out accuracy",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nPOOLED:", json.dumps(result["pooled"], indent=1))
    print("\nFEATURES:"); [print(f"  {k:14s} {v}") for k, v in features.items()]
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
