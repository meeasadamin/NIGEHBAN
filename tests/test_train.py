"""Training pipeline tests (audit F1.3-F1.5). The end-to-end run is marked slow (~2 min)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import risk_engine as engine
import train


def test_perfect_predictions_score_perfectly() -> None:
    y = np.array([0, 1, 2, 2, 1, 0])
    scores = train.score_probabilities(y, np.eye(3)[y])
    assert scores["macro_f1"] == 1.0 and scores["high_recall"] == 1.0
    assert scores["brier"] == pytest.approx(0.0) and scores["ece"] == pytest.approx(0.0)
    assert scores["log_loss"] < 1e-6


def test_uniform_predictions_have_known_brier_and_log_loss() -> None:
    y = np.array([0, 1, 2] * 10)
    scores = train.score_probabilities(y, np.full((30, 3), 1 / 3))
    assert scores["brier"] == pytest.approx(2 / 3)  # (1 - 1/3)^2 + 2 * (1/3)^2
    assert scores["log_loss"] == pytest.approx(np.log(3))


def test_expected_calibration_error_detects_overconfidence() -> None:
    y = np.array([0, 1] * 50)
    overconfident = np.tile([0.95, 0.04, 0.01], (100, 1))  # always 95% sure of class 0, right half the time
    assert train.expected_calibration_error(y, overconfident) == pytest.approx(0.45)


def test_each_pipeline_gets_its_own_preprocessor() -> None:
    features = train.V2_FEATURES
    first = train.make_pipeline(train.RandomForestClassifier(), features)
    second = train.make_pipeline(train.RandomForestClassifier(), features)
    assert first.named_steps["prep"] is not second.named_steps["prep"]  # audit F1.6 shared-object hazard


def test_feature_sets_exclude_labels_identifiers_and_geography() -> None:
    for features in train.HAZARD_FEATURES.values():
        columns = set(features.columns)
        assert not columns & (engine.LABEL_DERIVED_COLUMNS | engine.IDENTIFIER_COLUMNS)
        assert not columns & {"latitude", "longitude", "province"}


def test_threshold_choice_prefers_recall() -> None:
    y = np.array([2, 2, 2, 0, 0, 0, 0, 0])
    p_high = np.array([0.9, 0.35, 0.3, 0.32, 0.1, 0.05, 0.05, 0.02])
    assert train.choose_high_threshold(y, p_high) <= 0.30  # catching all three High beats avoiding one false alarm


@pytest.mark.slow
def test_quick_training_run_produces_a_loadable_verified_bundle(tmp_path: Path) -> None:
    out = tmp_path / "bundle.pkl"
    assert train.main(["--quick", "--out", str(out)]) == 0
    bundle = engine.load_bundle(out)
    assert bundle.integrity_verified
    assert bundle.calibration_method == "temperature"
    assert bundle.metadata["data_sha256"] == engine.file_sha256(engine.default_data_path())
    summary = bundle.metadata["nested_cv_summary"]
    assert {"brier", "log_loss"} <= summary["flood_risk"]["temperature"].keys()
    assert (out.parent / "metrics.json").is_file()
