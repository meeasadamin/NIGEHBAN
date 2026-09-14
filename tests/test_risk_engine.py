"""Risk engine tests: provenance, prediction, calibration, SHAP, validation, exports."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import risk_engine as engine

ROOT = engine.PROJECT_ROOT

# ---------------------------------------------------------------- provenance


def test_committed_model_matches_committed_data(bundle: engine.ModelBundle, df: pd.DataFrame, data_path: Path) -> None:
    """Regression test for audit F1.1: the shipped model must belong to the shipped data."""
    issues = engine.audit_provenance(bundle, df, engine.file_sha256(data_path))
    assert [i.message for i in issues if i.severity == "critical"] == []


def test_committed_model_is_integrity_verified(bundle: engine.ModelBundle) -> None:
    assert bundle.integrity_verified  # manifest written by train.py matches the pickle


def test_data_mismatch_is_detected(bundle: engine.ModelBundle, df: pd.DataFrame, tmp_path: Path) -> None:
    changed = df.copy()
    changed["avg_annual_rainfall_mm"] = changed["avg_annual_rainfall_mm"] * 1.5
    path = tmp_path / "districts.csv"
    changed.to_csv(path, index=False)
    issues = engine.audit_provenance(bundle, engine.load_districts(path), engine.file_sha256(path))
    critical = [i.message for i in issues if i.severity == "critical"]
    assert any("different version" in m for m in critical)
    assert any("avg_annual_rainfall_mm" in m for m in critical)  # scaler-median check works independently of the hash


def test_tampered_model_is_refused_before_unpickling(tmp_path: Path) -> None:
    copy = tmp_path / "bundle.pkl"
    shutil.copyfile(engine.default_model_path(), copy)
    engine.manifest_path_for(copy).write_text("0" * 64 + "  bundle.pkl\n", encoding="utf-8")
    with pytest.raises(engine.ArtifactIntegrityError):
        engine.load_bundle(copy)


def test_unverified_model_is_a_critical_provenance_issue(
    df: pd.DataFrame, data_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpickled model with no trusted digest must be surfaced on the page, not only in the model card."""
    monkeypatch.delenv("NDMA_MODEL_SHA256", raising=False)
    copy = tmp_path / "bundle.pkl"
    shutil.copyfile(engine.default_model_path(), copy)  # no manifest copied
    unverified = engine.load_bundle(copy)
    assert not unverified.integrity_verified
    issues = engine.audit_provenance(unverified, df, engine.file_sha256(data_path))
    assert any(i.check == "integrity" and i.severity == "critical" for i in issues)


def test_legacy_v1_bundle_is_rejected() -> None:
    with pytest.raises(engine.SchemaError, match="schema version"):
        engine.load_bundle(ROOT / "archive" / "v1" / "best_hazard_pipeline_v1.pkl")


def test_bundle_without_provenance_is_rejected(tmp_path: Path) -> None:
    raw = joblib.load(engine.default_model_path())
    del raw["metadata"]["data_sha256"]
    path = tmp_path / "no_provenance.pkl"
    joblib.dump(raw, path)
    with pytest.raises(engine.SchemaError, match="data_sha256"):
        engine.load_bundle(path)


def test_leaky_bundle_is_rejected(tmp_path: Path) -> None:
    raw = joblib.load(engine.default_model_path())
    raw["numeric_features"] = [*raw["numeric_features"], "flood_risk"]
    path = tmp_path / "leaky.pkl"
    joblib.dump(raw, path)
    with pytest.raises(engine.SchemaError, match="leakage"):
        engine.load_bundle(path)


# ---------------------------------------------------------------- prediction and calibration


def test_probabilities_are_valid_for_every_district(bundle: engine.ModelBundle, df: pd.DataFrame) -> None:
    for target in bundle.targets:
        proba = bundle.models[target].predict_proba(df[list(bundle.features_for(target))])
        assert proba.shape == (len(df), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_temperature_calibration_never_changes_the_predicted_class(
    bundle: engine.ModelBundle, df: pd.DataFrame
) -> None:
    for target in bundle.targets:
        features = df[list(bundle.features_for(target))]
        calibrated = bundle.models[target].predict_proba(features).argmax(axis=1)
        base = bundle.base_pipelines[target].predict_proba(features).argmax(axis=1)
        np.testing.assert_array_equal(calibrated, base)


def test_bundle_records_brier_and_log_loss(bundle: engine.ModelBundle) -> None:
    summary = bundle.metadata["nested_cv_summary"]
    for target in bundle.targets:
        for variant in ("uncalibrated", "temperature"):
            assert {"brier", "log_loss", "macro_f1", "ece"} <= summary[target][variant].keys()


# ---------------------------------------------------------------- explanations


@pytest.mark.parametrize("district", ["Islamabad", "Gwadar", "Qila Abdullah"])
def test_shap_values_add_up_to_the_explained_output(
    bundle: engine.ModelBundle, df: pd.DataFrame, district: str
) -> None:
    scenario = engine.build_scenario(df, district)
    for target in bundle.targets:
        explanation = engine.explain_hazard(bundle, scenario, target, 2)
        pipeline = bundle.base_pipelines[target]
        clf = pipeline.named_steps["clf"]
        columns = list(bundle.features_for(target))
        transformed = pipeline.named_steps["prep"].transform(scenario[columns])
        if explanation.output_space == "log-odds":
            expected = float(np.asarray(clf.predict(transformed, prediction_type="RawFormulaVal"))[0, 2])
        else:
            expected = float(pipeline.predict_proba(scenario[columns])[0, 2])
        assert explanation.final_value == pytest.approx(expected, abs=1e-4)


def test_explanations_cover_exactly_the_hazard_inputs(bundle: engine.ModelBundle, df: pd.DataFrame) -> None:
    scenario = engine.build_scenario(df, "Karachi South")
    for target in bundle.targets:
        explanation = engine.explain_hazard(bundle, scenario, target, 2)
        assert {c.feature for c in explanation.contributions} == set(bundle.features_for(target))


# ---------------------------------------------------------------- Nigehban priority 1: input isolation


FORBIDDEN_INPUTS = {
    "seismic_risk": {"summer_max_temp_c", "avg_annual_rainfall_mm", "vegetation_index_ndvi", "river_proximity_km"},
    "flood_risk": {"summer_max_temp_c", "active_fault_distance_km", "makran_subduction_distance_km"},
    "heatwave_risk": {"avg_annual_rainfall_mm", "river_proximity_km", "active_fault_distance_km"},
}


def test_hazard_models_never_receive_unrelated_inputs(bundle: engine.ModelBundle) -> None:
    for target, forbidden in FORBIDDEN_INPUTS.items():
        assert not forbidden & set(bundle.features_for(target)), target
        prep = bundle.base_pipelines[target].named_steps["prep"]
        assert not forbidden & {name.split("__", 1)[-1] for name in prep.get_feature_names_out()}, target


@pytest.mark.parametrize("district", ["Islamabad", "Quetta", "Karachi South"])
def test_moving_unrelated_sliders_leaves_seismic_unchanged(
    bundle: engine.ModelBundle, df: pd.DataFrame, district: str
) -> None:
    """The live-demo check: temperature and rainfall must not move seismic risk at all."""
    base = engine.predict_hazards(bundle, engine.build_scenario(df, district))
    moved = engine.predict_hazards(
        bundle, engine.build_scenario(df, district, {"summer_max_temp_c": 47.9, "avg_annual_rainfall_mm": 1800.0})
    )
    assert moved["seismic_risk"].probabilities == base["seismic_risk"].probabilities
    assert moved["seismic_risk"].label == base["seismic_risk"].label


# ---------------------------------------------------------------- Nigehban priority 2: recall-oriented rule


def test_high_threshold_rule_flags_high_below_argmax() -> None:
    assert engine.HazardPrediction("flood_risk", (0.5, 0.2, 0.3), high_threshold=0.25).label == "High"
    assert engine.HazardPrediction("flood_risk", (0.5, 0.2, 0.3), high_threshold=0.35).label == "Low"
    assert engine.HazardPrediction("flood_risk", (0.3, 0.4, 0.3), high_threshold=None).label == "Medium"
    assert engine.decide_class(np.array([[0.3, 0.3, 0.4]]), 0.5).tolist() == [1]  # Low/Medium tie goes to Medium


def test_bundle_thresholds_and_held_out_reports_are_recorded(bundle: engine.ModelBundle) -> None:
    for target in bundle.targets:
        assert 0.0 < bundle.high_thresholds[target] < 1.0
        reports = bundle.metadata["per_class_held_out"][target]
        assert {"argmax", "recall_tuned"} <= reports.keys()


def test_explanations_do_not_crash_in_fresh_processes() -> None:
    """Guards the native SHAP crash found during the rebuild (see risk_engine._build_explainer)."""
    script = (
        f"import sys; sys.path.insert(0, r'{ROOT}'); import risk_engine as e\n"
        "b = e.load_bundle(e.default_model_path()); d = e.load_districts(e.default_data_path())\n"
        "dom = e.ApplicabilityDomain.fit(d, b.numeric_features)\n"
        "[e.build_assessment(b, d, dom, name) for name in ('Islamabad', 'Gwadar', 'Skardu')]\n"
    )
    for _ in range(4):
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
        )
        assert result.returncode == 0, result.stderr[-2000:]


def _explanation(values: dict[str, float], space: engine.OutputSpace = "probability") -> engine.HazardExplanation:
    contributions = tuple(
        sorted(
            (engine.Contribution(n, n.title(), "x", v) for n, v in values.items()),
            key=lambda c: abs(c.value),
            reverse=True,
        )
    )
    return engine.HazardExplanation("flood_risk", 2, 0.33, contributions, space)


def test_narrative_single_dominant_driver() -> None:
    text = engine.narrate_explanation(_explanation({"rain": 0.30, "river": 0.02, "soil": 0.01}))
    assert text.startswith("Rain = x raises the High flood score by 30.0 pts")
    assert "no other feature accounts for more than 6%" in text


def test_narrative_reinforcing_drivers_and_dominance() -> None:
    dominant = engine.narrate_explanation(_explanation({"rain": 0.30, "river": 0.10}))
    balanced = engine.narrate_explanation(_explanation({"rain": 0.12, "river": 0.10}))
    assert "pushed up mainly by Rain" in dominant and "together 100%" in dominant
    assert "pushed up by Rain" in balanced and "mainly" not in balanced


def test_narrative_offsetting_drivers_follow_the_signs() -> None:
    raising = engine.narrate_explanation(_explanation({"rain": 0.30, "river": -0.10}))
    lowering = engine.narrate_explanation(_explanation({"rain": -0.30, "river": 0.10}))
    assert "Rain = x raises" in raising and "partly offset by River = x, which lowers" in raising
    assert "Rain = x lowers" in lowering and "which raises" in lowering
    assert "log-odds" in engine.narrate_explanation(_explanation({"rain": 0.3, "river": -0.2}, "log-odds"))


# ---------------------------------------------------------------- scenarios and applicability


@pytest.mark.parametrize(
    ("district", "overrides", "message"),
    [
        ("Atlantis", {}, "Unknown district"),
        ("Islamabad", {"elevation_m": 100.0}, "cannot be changed"),
        ("Islamabad", {"avg_annual_rainfall_mm": float("nan")}, "must be between"),
        ("Islamabad", {"vegetation_index_ndvi": 1.5}, "must be between"),
    ],
)
def test_invalid_scenarios_are_rejected(
    df: pd.DataFrame, district: str, overrides: dict[str, float], message: str
) -> None:
    with pytest.raises(engine.ScenarioValidationError, match=message):
        engine.build_scenario(df, district, overrides)


def test_applicability_domain_flags_extrapolation(df: pd.DataFrame, domain: engine.ApplicabilityDomain) -> None:
    baseline = engine.build_scenario(df, "Lahore")
    assert domain.assess(baseline) == ()
    beyond = engine.build_scenario(
        df, "Lahore", {"avg_annual_rainfall_mm": domain.upper["avg_annual_rainfall_mm"] + 500}
    )
    assert domain.out_of_range(beyond) == ("avg_annual_rainfall_mm",)
    odd = engine.build_scenario(
        df, "Skardu", {"summer_max_temp_c": 47.0, "population_density": 7000.0, "vegetation_index_ndvi": 0.02}
    )
    assert domain.novelty(odd) > 1.0


def test_loaded_arrays_are_read_only(domain: engine.ApplicabilityDomain) -> None:
    with pytest.raises(ValueError, match="read-only"):
        domain.reference[0, 0] = 1.0  # shared across Streamlit sessions (audit F3.2)


# ---------------------------------------------------------------- exports


def test_csv_and_text_exports(bundle: engine.ModelBundle, df: pd.DataFrame, domain: engine.ApplicabilityDomain) -> None:
    assessment = engine.build_assessment(bundle, df, domain, "Gwadar", {"summer_max_temp_c": 45.0})
    rows = list(csv.DictReader(io.StringIO(engine.assessment_to_csv(assessment))))
    assert [r["hazard"] for r in rows] == ["Flood", "Heatwave", "Seismic"]
    assert all(r["simulated_inputs_changed"] == "summer_max_temp_c" for r in rows)
    assert all(abs(sum(float(r[k]) for k in ("p_low", "p_medium", "p_high")) - 1) < 1e-3 for r in rows)
    text = engine.assessment_to_text(assessment)
    assert text.startswith(engine.EXPORT_DISCLAIMER)
    assert "No historical event dataset exists" in text


def test_national_predictions_agree_with_single_district_predictions(
    bundle: engine.ModelBundle, df: pd.DataFrame
) -> None:
    national = engine.predict_dataset(bundle, df)
    assert len(national) == len(df) * len(bundle.targets)
    for district in ("Islamabad", "Gwadar", "Skardu"):
        single = engine.predict_hazards(bundle, engine.build_scenario(df, district))
        rows = national[national["district_name"] == district].set_index("target")
        for target, prediction in single.items():
            assert rows.loc[target, "label"] == prediction.label
            assert rows.loc[target, "p_high"] == pytest.approx(prediction.p_high)


def test_district_profile_marks_changes_and_sources(
    bundle: engine.ModelBundle, df: pd.DataFrame, domain: engine.ApplicabilityDomain
) -> None:
    assessment = engine.build_assessment(bundle, df, domain, "Quetta", {"summer_max_temp_c": 45.0})
    profile = engine.district_profile(df, assessment).set_index("Input")
    assert len(profile) == len(engine.PROFILE_FEATURES)
    assert profile.loc["Summer max temperature", "Changed"]
    assert profile.loc["Summer max temperature", "This scenario"] == "45.0 °C"
    assert not profile.loc["Annual rainfall", "Changed"]
    assert profile.loc["Distance to active fault", "Source"] == "Real"
    assert profile.loc["Annual rainfall", "Source"] == "Synthetic"


def test_provenance_checklist_covers_every_check(bundle: engine.ModelBundle, df: pd.DataFrame, data_path: Path) -> None:
    passing = engine.provenance_checklist(bundle, engine.audit_provenance(bundle, df, engine.file_sha256(data_path)))
    assert [item.label for item in passing] == list(engine.PROVENANCE_CHECKS.values())
    assert all(item.passed for item in passing)
    failing = engine.provenance_checklist(bundle, [engine.ProvenanceIssue("critical", "data", "mismatch")])
    assert [item.passed for item in failing] == [True, False, True, True]


def test_map_layers_are_real_reference_geometry() -> None:
    layers = engine.load_map_layers()
    roles = {feature["properties"]["role"] for feature in layers.faults["features"]}
    assert roles == {"crustal", "makran_subduction"}
    assert layers.coastline["features"]


def test_csv_cells_cannot_inject_formulas() -> None:
    assert engine._csv_safe('=HYPERLINK("http://x")') == '\'=HYPERLINK("http://x")'
    assert engine._csv_safe("Gwadar") == "Gwadar"
    assert engine._csv_safe(-1.5) == -1.5


# ---------------------------------------------------------------- planning tools: stress test and hotspots


@pytest.fixture(scope="module")
def stressed(bundle: engine.ModelBundle, df: pd.DataFrame, domain: engine.ApplicabilityDomain) -> pd.DataFrame:
    return engine.stress_test(bundle, df, domain, 2.0, 1.5)


def test_stress_test_leaves_seismic_untouched(stressed: pd.DataFrame) -> None:
    seismic = stressed[stressed["target"] == "seismic_risk"]
    assert (seismic["change_pts"].abs() < 1e-9).all()
    assert not seismic["becomes_high"].any() and not seismic["extrapolated"].any()


def test_becomes_high_means_high_only_under_the_scenario(stressed: pd.DataFrame, bundle: engine.ModelBundle) -> None:
    newly = stressed[stressed["becomes_high"]]
    assert not newly.empty
    assert (newly["stressed_label"] == "High").all() and (newly["current_label"] != "High").all()
    assert len(stressed) == 150 * len(bundle.targets)


def test_stress_test_flags_extrapolation_beyond_training_range(
    bundle: engine.ModelBundle, df: pd.DataFrame, domain: engine.ApplicabilityDomain
) -> None:
    hottest = df["summer_max_temp_c"].idxmax()
    results = engine.stress_test(bundle, df, domain, 2.0, 1.0)
    row = results[
        (results["district_name"] == df.loc[hottest, "district_name"]) & (results["target"] == "heatwave_risk")
    ]
    assert bool(row["extrapolated"].iloc[0])  # the hottest district is pushed past the training maximum
    none = engine.stress_test(bundle, df, domain, 0.0, 1.0)
    # Parallel tree summation leaves ~1e-14 pt float noise; no label may change.
    assert not none["extrapolated"].any() and (none["change_pts"].abs() < 1e-9).all()
    assert (none["current_label"] == none["stressed_label"]).all()


@pytest.mark.parametrize(("temp", "rain"), [(-1.0, 1.5), (5.0, 1.5), (2.0, 0.1), (2.0, float("nan"))])
def test_stress_settings_are_validated(
    bundle: engine.ModelBundle, df: pd.DataFrame, domain: engine.ApplicabilityDomain, temp: float, rain: float
) -> None:
    with pytest.raises(engine.ScenarioValidationError):
        engine.stress_test(bundle, df, domain, temp, rain)


def test_hotspots_are_districts_high_for_two_or_more_hazards(
    bundle: engine.ModelBundle, df: pd.DataFrame, stressed: pd.DataFrame
) -> None:
    national = engine.predict_dataset(bundle, df)
    hotspots = engine.multi_hazard_hotspots(national)
    counts = national[national["label"] == "High"].groupby("district_name").size()
    assert set(hotspots["district_name"]) == set(counts[counts >= 2].index)
    assert hotspots["high_count"].is_monotonic_decreasing
    under_stress = engine.multi_hazard_hotspots(stressed, "stressed_label", "stressed_p_high")
    assert (under_stress["high_count"] >= 2).all()
