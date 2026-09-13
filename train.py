"""Training pipeline for the Pakistan Multi-Hazard Risk Analyzer.

Replaces the training sections (3-9) of ``analysis.ipynb``, which is kept only
as an archived record of v1. Each change is tied to an AUDIT.md finding:

* F1.5  Model family and Optuna tuning were selected on the same folds that
        reported the score -> nested cross-validation: an inner loop selects
        family + hyperparameters, an untouched outer fold scores them.
* F1.4  XGBoost was compared without class weighting -> every candidate family
        is class-weighted ("balanced"). XGBoost is not a candidate: its sklearn
        API needs per-fold sample weights routed through calibration, and two
        weighted families (RandomForest, CatBoost) keep the comparison fair.
* F1.4  Only macro-F1 was reported -> also balanced accuracy, ROC-AUC,
        High-class recall/precision, quadratic weighted kappa.
* F1.3  Probabilities were never calibrated or measured -> temperature scaling
        (``CalibratedClassifierCV(method="temperature")``) with Brier score,
        log loss, and ECE reported before and after, on outer folds. Temperature
        scaling is fixed in advance (not selected on outer folds): it keeps each
        prediction's top class, so macro-F1 is unaffected. Sigmoid calibration is
        reported as a sensitivity check only.
* F1.1/F1.2/F6.2 The shipped model did not match the data or the README ->
        the bundle records the data SHA-256, library versions, and the honest
        metrics, and a ``.sha256`` manifest is written next to it.
* F1.6  All three final pipelines shared one preprocessor object -> each
        pipeline gets its own cloned preprocessor.

Usage:
    python train.py                 # full run, writes models/best_hazard_pipeline.pkl
    python train.py --quick --out /tmp/bundle.pkl   # CI smoke run (tiny search)
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import logging
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

import risk_engine as engine

logger = logging.getLogger("train")

SEED: Final[int] = 42
CLASSES: Final[np.ndarray] = np.arange(len(engine.CLASS_LABELS))
FAMILIES: Final[tuple[str, ...]] = ("RandomForest", "CatBoost")
#: Libraries whose versions must match between training and serving.
TRACKED_LIBRARIES: Final[tuple[str, ...]] = ("numpy", "pandas", "scikit-learn", "catboost", "shap", "joblib")

#: v1 numbers as printed in analysis.ipynb / README (Optuna best_value on the selection folds).
V1_REPORTED_MACRO_F1: Final[dict[str, float]] = {"flood_risk": 0.716, "heatwave_risk": 0.665, "seismic_risk": 0.791}


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Size of the nested search.

    Attributes:
        n_trials: Optuna trials per inner search.
        outer_folds: Outer folds used only for scoring.
        inner_folds: Inner folds used for selection and for calibration.
    """

    n_trials: int
    outer_folds: int
    inner_folds: int


FULL: Final = SearchConfig(n_trials=20, outer_folds=5, inner_folds=3)
QUICK: Final = SearchConfig(n_trials=2, outer_folds=2, inner_folds=2)


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Model input columns.

    Attributes:
        numeric: Numeric columns (robust-scaled).
        categorical: Categorical columns (one-hot encoded).
    """

    numeric: tuple[str, ...] = engine.NUMERIC_FEATURES
    categorical: tuple[str, ...] = engine.CATEGORICAL_FEATURES

    @property
    def columns(self) -> list[str]:
        """All input columns in pipeline order."""
        return [*self.numeric, *self.categorical]


V2_FEATURES: Final = FeatureSet()  # all columns; used only by scripts/evaluate_v1_protocol.py-style comparisons

#: Input isolation (Nigehban fix, priority 1). Until this change every hazard model trained on the same
#: 16-column matrix, so the seismic model learned to react to temperature and rainfall. Each hazard now
#: receives only inputs with a physical mechanism for that hazard. Latitude, longitude and province are
#: excluded everywhere: they let a model memorise geography instead of drivers.
HAZARD_FEATURES: Final[dict[str, FeatureSet]] = {
    "flood_risk": FeatureSet(
        numeric=(
            "avg_annual_rainfall_mm",  # water input: the primary flood driver
            "river_proximity_km",  # riverine inundation reach
            "elevation_m",  # low-lying plains accumulate and hold water
            "simulated_flood_events",  # recurrence: places that flooded before tend to flood again
            "population_density",  # exposure: more people and assets in harm's way
            "infrastructure_quality_score",  # drainage and flood protection reduce impact
        ),
        categorical=(),
    ),
    "heatwave_risk": FeatureSet(
        numeric=(
            "summer_max_temp_c",  # hazard intensity itself
            "vegetation_index_ndvi",  # vegetation shades and cools; bare ground heats
            "elevation_m",  # temperature falls ~6.5 °C per km of altitude
            "coast_distance_km",  # sea breezes moderate coastal heat
            "population_density",  # urban heat island and exposed population
            "infrastructure_quality_score",  # access to power and cooling reduces harm
        ),
        categorical=(),
    ),
    "seismic_risk": FeatureSet(
        numeric=(
            "active_fault_distance_km",  # proximity to crustal fault sources (incl. the Chaman fault zone)
            "makran_subduction_distance_km",  # proximity to the Makran megathrust source
            "simulated_earthquake_events",  # recurrence of shaking
            "infrastructure_quality_score",  # building vulnerability decides damage
        ),
        # Soil type affects real site amplification, but in this dataset it is a random negative
        # control (see DATA_DICTIONARY.md), so including it would only add noise.
        categorical=(),
    ),
}

#: Recall-oriented decision rule (priority 2): predict High when calibrated P(High) >= t, with t
#: chosen on training folds only to maximise F-beta for the High class. beta = 2 weights recall twice
#: as heavily as precision: for an early-warning screen, a missed High district costs more than a
#: false alarm on a safe one (judgment; stated so the trade-off is explicit).
HIGH_F_BETA: Final[float] = 2.0
THRESHOLD_GRID: Final[np.ndarray] = np.round(np.arange(0.05, 0.951, 0.01), 2)


def make_preprocessor(features: FeatureSet) -> ColumnTransformer:
    """Return a fresh, unfitted preprocessor.

    RobustScaler because the data contains bounded climate anomalies; tree models
    do not need scaling, but the scaler's stored medians double as a
    provenance check in ``risk_engine.audit_provenance``.
    """
    transformers: list[tuple[str, Any, list[str]]] = [("num", RobustScaler(), list(features.numeric))]
    if features.categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(features.categorical)))
    return ColumnTransformer(transformers)


def choose_high_threshold(y: np.ndarray, p_high: np.ndarray) -> float:
    """Pick the P(High) threshold that maximises F-beta (beta = HIGH_F_BETA) for the High class.

    Ties go to the higher threshold, which raises fewer false alarms for the same score.
    """
    is_high = y == len(engine.CLASS_LABELS) - 1
    best_threshold, best_score = 0.5, -1.0
    for threshold in THRESHOLD_GRID:
        score = fbeta_score(is_high, p_high >= threshold, beta=HIGH_F_BETA, zero_division=0)
        if score >= best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def suggest_classifier(trial: optuna.Trial, seed: int) -> BaseEstimator:
    """Sample a class-weighted classifier (family and hyperparameters) for one trial.

    Search ranges are conventional for ~150 rows: shallow trees and strong
    leaf/regularisation limits to avoid memorising a small dataset (judgment).
    """
    family = trial.suggest_categorical("family", FAMILIES)
    if family == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 200, 600, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", 0.5, 1.0]),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    return CatBoostClassifier(
        iterations=trial.suggest_int("cb_iterations", 150, 600, step=50),
        depth=trial.suggest_int("cb_depth", 3, 7),
        learning_rate=trial.suggest_float("cb_learning_rate", 0.02, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("cb_l2_leaf_reg", 1.0, 10.0),
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_pipeline(classifier: BaseEstimator, features: FeatureSet) -> Pipeline:
    """Wrap a classifier with its own preprocessor (never shared across pipelines)."""
    return Pipeline([("prep", make_preprocessor(features)), ("clf", classifier)])


def calibrate(pipeline: Pipeline, method: str, folds: int, seed: int) -> CalibratedClassifierCV:
    """Return an unfitted calibrated wrapper.

    ``ensemble=False`` refits one base pipeline on all given data and fits one
    calibrator on its cross-validated predictions, so the served model has a
    single base pipeline that SHAP can explain.
    """
    return CalibratedClassifierCV(
        pipeline, method=method, cv=StratifiedKFold(folds, shuffle=True, random_state=seed), ensemble=False
    )


def tune(
    X: pd.DataFrame, y: np.ndarray, config: SearchConfig, seed: int, features: FeatureSet
) -> optuna.trial.FrozenTrial:
    """Select family and hyperparameters by inner-CV macro-F1.

    Args:
        X: Training features (never includes the outer test fold).
        y: Training labels.
        config: Search size.
        seed: RNG seed for folds, sampler, and models.
        features: Input columns.

    Returns:
        Optuna's best trial.
    """
    inner = StratifiedKFold(config.inner_folds, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        scores = []
        for train_idx, valid_idx in inner.split(X, y):
            pipeline = make_pipeline(suggest_classifier(trial, seed), features).fit(X.iloc[train_idx], y[train_idx])
            predicted = np.asarray(pipeline.predict_proba(X.iloc[valid_idx])).argmax(axis=1)
            scores.append(f1_score(y[valid_idx], predicted, average="macro", labels=CLASSES, zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)
    return study.best_trial


def classifier_from_trial(trial: optuna.trial.FrozenTrial, seed: int) -> BaseEstimator:
    """Rebuild the classifier described by a finished trial."""
    return suggest_classifier(optuna.trial.FixedTrial(trial.params), seed)


def expected_calibration_error(y: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in itertools.pairwise(edges):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def score_probabilities(y: np.ndarray, proba: np.ndarray, predicted: np.ndarray | None = None) -> dict[str, float]:
    """Compute decision and probability-quality metrics; decisions default to argmax."""
    predicted = proba.argmax(axis=1) if predicted is None else predicted
    onehot = np.eye(len(CLASSES))[y]
    return {
        "macro_f1": float(f1_score(y, predicted, average="macro", labels=CLASSES, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "roc_auc_ovr": float(roc_auc_score(y, proba, multi_class="ovr", labels=CLASSES)),
        "high_recall": float(recall_score(y, predicted, labels=[2], average="macro", zero_division=0)),
        "high_precision": float(precision_score(y, predicted, labels=[2], average="macro", zero_division=0)),
        "quadratic_kappa": float(cohen_kappa_score(y, predicted, weights="quadratic")),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-15, 1.0), labels=CLASSES)),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
        "ece": expected_calibration_error(y, proba),
    }


#: Evaluation variants. "recall_tuned" is the deployed decision rule: temperature-calibrated
#: probabilities plus the High threshold chosen on the training folds.
VARIANTS: Final[tuple[str, ...]] = ("uncalibrated", "temperature", "recall_tuned")


def per_class_report(y: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Precision, recall, F1 and support per class, plus the High false-alarm rate."""
    report = classification_report(
        y, predicted, labels=CLASSES, target_names=list(engine.CLASS_LABELS), output_dict=True, zero_division=0
    )
    not_high = y != len(CLASSES) - 1
    report["high_false_alarm_rate"] = float((predicted[not_high] == len(CLASSES) - 1).mean())
    report["high_false_alarms"] = int((predicted[not_high] == len(CLASSES) - 1).sum())
    report["high_missed"] = int(((y == len(CLASSES) - 1) & (predicted != len(CLASSES) - 1)).sum())
    return report


def nested_cv(X: pd.DataFrame, y: np.ndarray, config: SearchConfig, seed: int, features: FeatureSet) -> dict[str, Any]:
    """Honest generalisation estimate: selection happens only inside each outer training fold.

    The High threshold is also chosen inside each outer training fold (on out-of-fold calibrated
    probabilities), so every outer-fold score is on districts that influenced neither the model,
    the calibration, nor the threshold. Pooling the outer folds gives one held-out prediction per
    district, which is the "same held-out set" for the before/after per-class comparison.

    Returns:
        ``{"folds", "summary": {variant: {metric: {"mean", "std"}}}, "per_class": {rule: report},
        "selected_families"}``
    """
    outer = StratifiedKFold(config.outer_folds, shuffle=True, random_state=seed)
    inner = StratifiedKFold(config.inner_folds, shuffle=True, random_state=seed)
    folds: list[dict[str, Any]] = []
    pooled_y, pooled_argmax, pooled_tuned = [], [], []
    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
        best = tune(X_train, y_train, config, seed + fold, features)

        def fresh(best: optuna.trial.FrozenTrial = best) -> Pipeline:
            return make_pipeline(classifier_from_trial(best, seed), features)

        raw = np.asarray(fresh().fit(X_train, y_train).predict_proba(X_test))
        calibrated_model = calibrate(fresh(), "temperature", config.inner_folds, seed).fit(X_train, y_train)
        calibrated = np.asarray(calibrated_model.predict_proba(X_test))
        oof_train = cross_val_predict(
            calibrate(fresh(), "temperature", config.inner_folds, seed),
            X_train,
            y_train,
            cv=inner,
            method="predict_proba",
        )
        threshold = choose_high_threshold(y_train, oof_train[:, -1])
        tuned = engine.decide_class(calibrated, threshold)

        record: dict[str, Any] = {
            "fold": fold,
            "family": best.params["family"],
            "inner_macro_f1": best.value,
            "high_threshold": threshold,
            "metrics": {
                "uncalibrated": score_probabilities(y_test, raw),
                "temperature": score_probabilities(y_test, calibrated),
                "recall_tuned": score_probabilities(y_test, calibrated, tuned),
            },
        }
        pooled_y.append(y_test)
        pooled_argmax.append(engine.decide_class(calibrated, None))
        pooled_tuned.append(tuned)
        logger.info(
            "  outer fold %d: %s | macro-F1 %.3f -> %.3f | High recall %.2f -> %.2f (threshold %.2f)",
            fold,
            record["family"],
            record["metrics"]["temperature"]["macro_f1"],
            record["metrics"]["recall_tuned"]["macro_f1"],
            record["metrics"]["temperature"]["high_recall"],
            record["metrics"]["recall_tuned"]["high_recall"],
            threshold,
        )
        folds.append(record)

    summary = {
        variant: {
            metric: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            for metric, values in pd.DataFrame([f["metrics"][variant] for f in folds]).items()
        }
        for variant in VARIANTS
    }
    y_all = np.concatenate(pooled_y)
    per_class = {
        "argmax": per_class_report(y_all, np.concatenate(pooled_argmax)),
        "recall_tuned": per_class_report(y_all, np.concatenate(pooled_tuned)),
    }
    return {
        "folds": folds,
        "summary": summary,
        "per_class": per_class,
        "selected_families": [f["family"] for f in folds],
    }


def library_versions() -> dict[str, str]:
    """Installed versions of the libraries that must match at serving time."""
    return {name: importlib.metadata.version(name) for name in TRACKED_LIBRARIES}


def train_all(data_path: Path, config: SearchConfig, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run nested CV and fit the deployable calibrated model for every target.

    Returns:
        ``(bundle, report)``: the bundle to serialise and a JSON-serialisable report.
    """
    data_sha256 = engine.file_sha256(data_path)  # hashed before reading, so it describes exactly the data trained on
    df = engine.load_districts(data_path)

    models: dict[str, Any] = {}
    bases: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    report: dict[str, Any] = {}
    for target in engine.TARGETS:
        features = HAZARD_FEATURES[target]
        if leaked := (engine.LABEL_DERIVED_COLUMNS | engine.IDENTIFIER_COLUMNS) & set(features.columns):
            raise ValueError(f"Leakage guard: label or identifier columns in {target} features: {sorted(leaked)}")
        X = df[features.columns]
        y = df[target].to_numpy()
        logger.info(
            "=== %s | %d inputs %s | class counts %s ===",
            target,
            len(features.columns),
            features.columns,
            np.bincount(y, minlength=3).tolist(),
        )
        started = time.perf_counter()
        evaluation = nested_cv(X, y, config, seed, features)

        best = tune(X, y, config, seed, features)  # final selection on all data; its score is NOT reported
        model = calibrate(
            make_pipeline(classifier_from_trial(best, seed), features), "temperature", config.inner_folds, seed
        ).fit(X, y)
        oof = cross_val_predict(
            calibrate(
                make_pipeline(classifier_from_trial(best, seed), features), "temperature", config.inner_folds, seed
            ),
            X,
            y,
            cv=StratifiedKFold(config.outer_folds, shuffle=True, random_state=seed),
            method="predict_proba",
        )
        thresholds[target] = choose_high_threshold(y, oof[:, -1])
        models[target] = model
        bases[target] = model.calibrated_classifiers_[0].estimator
        report[target] = {
            "features": features.columns,
            "class_counts": np.bincount(y, minlength=3).tolist(),
            "final_family": best.params["family"],
            "final_params": best.params,
            "high_threshold": thresholds[target],
            "nested_cv": evaluation,
            "seconds": round(time.perf_counter() - started, 1),
        }
        before, after = evaluation["per_class"]["argmax"]["High"], evaluation["per_class"]["recall_tuned"]["High"]
        logger.info(
            "%s: High recall %.2f -> %.2f, precision %.2f -> %.2f (held-out, pooled) | threshold %.2f | family %s",
            target,
            before["recall"],
            after["recall"],
            before["precision"],
            after["precision"],
            thresholds[target],
            best.params["family"],
        )

    metadata = {
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "library_versions": library_versions(),
        "data_file": data_path.name,
        "data_sha256": data_sha256,
        "n_rows": len(df),
        "seed": seed,
        "search": {"families": list(FAMILIES), "class_weighting": "balanced", **asdict(config)},
        "calibration_method": "temperature",
        "decision_rule": f"High if calibrated P(High) >= per-hazard threshold (max F{HIGH_F_BETA:g} for High); "
        "otherwise the more probable of Low and Medium",
        "model_families": {t: report[t]["final_family"] for t in engine.TARGETS},
        "nested_cv_summary": {t: report[t]["nested_cv"]["summary"] for t in engine.TARGETS},
        "per_class_held_out": {t: report[t]["nested_cv"]["per_class"] for t in engine.TARGETS},
        "class_prevalence": {t: (np.bincount(df[t], minlength=3) / len(df)).tolist() for t in engine.TARGETS},
    }
    used = {c for t in engine.TARGETS for c in HAZARD_FEATURES[t].columns}
    bundle = {
        "schema_version": engine.BUNDLE_SCHEMA_VERSION,
        "models": models,
        "base_pipelines": bases,
        "targets": list(engine.TARGETS),
        "numeric_features": [c for c in engine.NUMERIC_FEATURES if c in used],
        "categorical_features": [c for c in engine.CATEGORICAL_FEATURES if c in used],
        "feature_sets": {t: HAZARD_FEATURES[t].columns for t in engine.TARGETS},
        "high_thresholds": thresholds,
        "class_labels": list(engine.CLASS_LABELS),
        "metadata": metadata,
    }
    return bundle, {"metadata": metadata, "targets": report}


def write_outputs(bundle: dict[str, Any], report: dict[str, Any], out: Path) -> None:
    """Write the bundle, its SHA-256 manifest, and the metrics report next to it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    digest = engine.file_sha256(out)
    engine.manifest_path_for(out).write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    out.with_name("metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Saved %s (sha256 %s) + manifest + metrics.json", out, digest)


def print_summary(report: dict[str, Any]) -> None:
    """Print the honest metrics next to the v1 notebook's numbers."""
    rows = []
    for target, result in report["targets"].items():
        summary = result["nested_cv"]["summary"]
        tuned, cal, raw = summary["recall_tuned"], summary["temperature"], summary["uncalibrated"]
        rows.append(
            {
                "hazard": engine.TARGET_LABELS[target],
                "macro-F1 argmax -> tuned": f"{cal['macro_f1']['mean']:.3f} -> {tuned['macro_f1']['mean']:.3f}",
                "High recall argmax -> tuned": f"{cal['high_recall']['mean']:.2f} -> {tuned['high_recall']['mean']:.2f}",
                "ROC-AUC": f"{cal['roc_auc_ovr']['mean']:.3f}",
                "Brier raw -> cal": f"{raw['brier']['mean']:.3f} -> {cal['brier']['mean']:.3f}",
                "log loss raw -> cal": f"{raw['log_loss']['mean']:.3f} -> {cal['log_loss']['mean']:.3f}",
                "threshold": result["high_threshold"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    for target, result in report["targets"].items():
        print(f"\n{engine.TARGET_LABELS[target]} — per class, pooled held-out predictions (argmax -> recall-tuned)")
        for label in engine.CLASS_LABELS:
            a, t = (
                result["nested_cv"]["per_class"]["argmax"][label],
                result["nested_cv"]["per_class"]["recall_tuned"][label],
            )
            print(
                f"  {label:6s} precision {a['precision']:.2f} -> {t['precision']:.2f} | recall {a['recall']:.2f} -> "
                f"{t['recall']:.2f} | F1 {a['f1-score']:.2f} -> {t['f1-score']:.2f} | n={int(a['support'])}"
            )
        a, t = result["nested_cv"]["per_class"]["argmax"], result["nested_cv"]["per_class"]["recall_tuned"]
        print(
            f"  missed High {a['high_missed']} -> {t['high_missed']} | false High alarms {a['high_false_alarms']} -> "
            f"{t['high_false_alarms']} (rate {a['high_false_alarm_rate']:.1%} -> {t['high_false_alarm_rate']:.1%})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Train calibrated hazard models with nested CV.")
    parser.add_argument("--data", type=Path, default=engine.default_data_path())
    parser.add_argument("--out", type=Path, default=engine.default_model_path())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trials", type=int, help="Override Optuna trials per search.")
    parser.add_argument("--quick", action="store_true", help="Tiny search for CI smoke tests; never ship its output.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base = QUICK if args.quick else FULL
    config = SearchConfig(args.trials or base.n_trials, base.outer_folds, base.inner_folds)

    bundle, report = train_all(args.data, config, args.seed)
    write_outputs(bundle, report, args.out)
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
