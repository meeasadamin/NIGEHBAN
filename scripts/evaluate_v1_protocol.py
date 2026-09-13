"""Re-score the archived v1 dataset with the v2 nested-CV protocol.

Isolates the effect of the evaluation protocol from the effect of the new data:
v1 notebook numbers (selected and scored on the same folds, v1 data) are
compared with nested-CV numbers on the *same* v1 data and v1 feature list.
Supports the "are honest metrics lower, and why" statement in README/AUDIT
(audit F1.5).

Usage:
    python scripts/evaluate_v1_protocol.py [--trials 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_DATA = PROJECT_ROOT / "archive" / "v1" / "pakistan_districts_v1.csv"
OUTPUT = PROJECT_ROOT / "archive" / "v1" / "v1_nested_cv_metrics.json"

#: Feature lists exactly as defined in analysis.ipynb (v1), cell "3. Preprocessing Pipeline".
V1_FEATURES = train.FeatureSet(
    numeric=(
        "latitude",
        "longitude",
        "elevation_m",
        "avg_annual_rainfall_mm",
        "river_proximity_km",
        "population_density",
        "avg_summer_temp_celsius",
        "historical_flood_events",
        "historical_earthquake_events",
        "historical_disasters",
        "vegetation_index_ndvi",
        "infrastructure_quality_score",
    ),
    categorical=("province", "soil_type"),
)


def main() -> int:
    """Run nested CV on v1 data and print it beside the v1 notebook numbers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=train.FULL.n_trials)
    parser.add_argument("--seed", type=int, default=train.SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = pd.read_csv(V1_DATA)
    config = train.SearchConfig(args.trials, train.FULL.outer_folds, train.FULL.inner_folds)
    results = {}
    for target in ("flood_risk", "heatwave_risk", "seismic_risk"):
        y = df[target].to_numpy()
        evaluation = train.nested_cv(df[V1_FEATURES.columns], y, config, args.seed, V1_FEATURES)
        results[target] = evaluation["summary"]
        f1 = evaluation["summary"]["uncalibrated"]["macro_f1"]
        print(
            f"{target:14s} v1 notebook {train.V1_REPORTED_MACRO_F1[target]:.3f} | "
            f"nested on same data {f1['mean']:.3f} ± {f1['std']:.3f} | "
            f"difference {f1['mean'] - train.V1_REPORTED_MACRO_F1[target]:+.3f} | families {evaluation['selected_families']}"
        )
    OUTPUT.write_text(
        json.dumps({"search": asdict(config), "seed": args.seed, "summary": results}, indent=2), encoding="utf-8"
    )
    gap = np.mean([results[t]["uncalibrated"]["macro_f1"]["mean"] - train.V1_REPORTED_MACRO_F1[t] for t in results])
    print(f"Wrote {OUTPUT}; mean macro-F1 difference vs v1 notebook: {gap:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
