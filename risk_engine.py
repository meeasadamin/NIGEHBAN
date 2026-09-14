"""Framework-agnostic risk engine for the Pakistan Multi-Hazard Risk Analyzer.

Everything that touches data, models, SHAP, or exports lives here, so it can be
unit tested without Streamlit and reused unchanged by a batch job or an API.
``app.py`` only renders what this module returns. (Audit F2.1: monolithic app.py.)

Design rules enforced here:

* No ``matplotlib.pyplot`` calls (audit F3.1). pyplot keeps process-global
  "current figure" state that concurrent Streamlit sessions race on. Note that
  ``import shap`` itself imports pyplot as a side effect; this module never
  creates a figure, and ``tests/test_no_pyplot.py`` enforces both facts.
* Loaded artifacts are immutable (frozen dataclasses, ``MappingProxyType``,
  read-only numpy arrays) because ``st.cache_resource`` shares one object
  across every user session (audit F3.2).
* Pickles are SHA-256 verified *before* ``joblib.load`` executes them (audit F4.3).
* Every failure a user can trigger raises a ``RiskEngineError`` whose message
  is safe to show in the UI (audit F4.2).
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import importlib.metadata
import io
import json
import logging
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
#: v3: per-hazard input isolation (``feature_sets``) and recall-oriented High thresholds (``high_thresholds``).
BUNDLE_SCHEMA_VERSION: Final[int] = 3

# --------------------------------------------------------------------------
# Schema (single source of truth, imported by train.py)
# --------------------------------------------------------------------------

TARGETS: Final[tuple[str, ...]] = ("flood_risk", "heatwave_risk", "seismic_risk")
CLASS_LABELS: Final[tuple[str, ...]] = ("Low", "Medium", "High")
TARGET_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {"flood_risk": "Flood", "heatwave_risk": "Heatwave", "seismic_risk": "Seismic"}
)

#: Label columns, plus the label-derived column v1 shipped. Any of them as a
#: model input is target leakage, so such bundles are rejected (audit F1.6-g).
LABEL_DERIVED_COLUMNS: Final[frozenset[str]] = frozenset({*TARGETS, "overall_risk_score"})
IDENTIFIER_COLUMNS: Final[frozenset[str]] = frozenset({"district_name", "geonames_id"})

PROVINCES: Final[tuple[str, ...]] = (
    "Azad Jammu & Kashmir",
    "Balochistan",
    "Gilgit-Baltistan",
    "Islamabad Capital Territory",
    "Khyber Pakhtunkhwa",
    "Punjab",
    "Sindh",
)
SOIL_TYPES: Final[tuple[str, ...]] = ("alluvial", "clay", "rocky", "sandy")

NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "latitude",
    "longitude",
    "elevation_m",
    "coast_distance_km",
    "active_fault_distance_km",
    "makran_subduction_distance_km",
    "avg_annual_rainfall_mm",
    "summer_max_temp_c",
    "vegetation_index_ndvi",
    "river_proximity_km",
    "population_density",
    "infrastructure_quality_score",
    "simulated_flood_events",
    "simulated_earthquake_events",
)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("province", "soil_type")

# Closed-vocabulary categoricals: an unknown province becomes NaN on read and is
# rejected by validation instead of silently one-hot encoding to all zeros.
CSV_DTYPES: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "district_name": "string",
        "geonames_id": "int64",
        "province": pd.CategoricalDtype(PROVINCES),
        **{name: "float64" for name in NUMERIC_FEATURES if not name.startswith("simulated_")},
        "simulated_flood_events": "int64",
        "simulated_earthquake_events": "int64",
        "soil_type": pd.CategoricalDtype(SOIL_TYPES),
        **{target: "int64" for target in TARGETS},
    }
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Display and validation metadata for one numeric feature.

    Attributes:
        name: Column name in the dataset and model input.
        label: Human-readable name without unit.
        unit: Unit suffix shown next to values ("" for unitless).
        lower: Physical lower bound; scenario values below are rejected.
        upper: Physical upper bound; scenario values above are rejected.
        step: Slider increment.
        decimals: Decimal places used for display.
        tunable: Whether the what-if simulator may change this feature.
    """

    name: str
    label: str
    unit: str
    lower: float
    upper: float
    step: float
    decimals: int
    tunable: bool = False

    def format(self, value: float) -> str:
        """Format ``value`` with this feature's precision and unit."""
        text = f"{value:,.{self.decimals}f}"
        return f"{text} {self.unit}" if self.unit else text


#: Geography and simulated event counts are fixed properties of a district,
#: so only climate and exposure variables are tunable in the simulator.
FEATURE_SPECS: Final[Mapping[str, FeatureSpec]] = MappingProxyType(
    {
        spec.name: spec
        for spec in (
            FeatureSpec("avg_annual_rainfall_mm", "Annual rainfall", "mm", 0.0, 12_000.0, 10.0, 0, tunable=True),
            FeatureSpec("summer_max_temp_c", "Summer max temperature", "°C", -20.0, 60.0, 0.1, 1, tunable=True),
            FeatureSpec("river_proximity_km", "Distance to nearest river", "km", 0.0, 500.0, 0.1, 1, tunable=True),
            FeatureSpec("vegetation_index_ndvi", "Vegetation index (NDVI)", "", -1.0, 1.0, 0.01, 2, tunable=True),
            FeatureSpec("population_density", "Population density", "/km²", 0.0, 100_000.0, 10.0, 0, tunable=True),
            FeatureSpec(
                "infrastructure_quality_score", "Infrastructure quality", "/10", 1.0, 10.0, 0.1, 1, tunable=True
            ),
            FeatureSpec("elevation_m", "Elevation", "m", -10.0, 8_611.0, 10.0, 0),
            FeatureSpec("latitude", "Latitude", "°N", 23.0, 37.5, 0.01, 2),
            FeatureSpec("longitude", "Longitude", "°E", 60.0, 78.0, 0.01, 2),
            FeatureSpec("coast_distance_km", "Distance to coast", "km", 0.0, 2_000.0, 1.0, 0),
            FeatureSpec("active_fault_distance_km", "Distance to active fault", "km", 0.0, 1_000.0, 1.0, 1),
            FeatureSpec(
                "makran_subduction_distance_km", "Distance to Makran subduction zone", "km", 0.0, 2_500.0, 1.0, 0
            ),
            FeatureSpec("simulated_flood_events", "Simulated past floods", "", 0.0, 1_000.0, 1.0, 0),
            FeatureSpec("simulated_earthquake_events", "Simulated past earthquakes", "", 0.0, 1_000.0, 1.0, 0),
        )
    }
)
TUNABLE_FEATURES: Final[tuple[FeatureSpec, ...]] = tuple(s for s in FEATURE_SPECS.values() if s.tunable)
TUNABLE_BY_NAME: Final[Mapping[str, FeatureSpec]] = MappingProxyType({s.name: s for s in TUNABLE_FEATURES})
FEATURE_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {**{name: spec.label for name, spec in FEATURE_SPECS.items()}, "province": "Province", "soil_type": "Soil type"}
)

REQUIRED_BUNDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "models",
        "base_pipelines",
        "feature_sets",
        "high_thresholds",
        "targets",
        "numeric_features",
        "categorical_features",
        "class_labels",
        "metadata",
    }
)
#: Provenance written by train.py and read by the app's model card and provenance audit.
REQUIRED_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {"trained_at_utc", "library_versions", "data_file", "data_sha256", "search", "model_families", "nested_cv_summary"}
)

OutputSpace = Literal["probability", "log-odds"]
Severity = Literal["critical", "info"]
ProvenanceCheck = Literal["integrity", "data", "libraries", "calibration"]
#: Every provenance check, in display order, with the label shown in the model card checklist.
PROVENANCE_CHECKS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "integrity": "Model file integrity (SHA-256)",
        "data": "Trained on the dataset shown",
        "libraries": "Library versions match training",
        "calibration": "Probabilities calibrated",
    }
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class RiskEngineError(Exception):
    """Base class. ``str(exc)`` is safe to display to end users."""


class ArtifactMissingError(RiskEngineError):
    """A required data or model file does not exist."""


class ArtifactIntegrityError(RiskEngineError):
    """A model file's SHA-256 does not match the trusted digest."""


class SchemaError(RiskEngineError):
    """A dataset or model bundle does not have the expected structure."""


class ScenarioValidationError(RiskEngineError):
    """A scenario request contains an unknown district, feature, or value."""


class ModelOutputError(RiskEngineError):
    """A model returned malformed output (wrong shape, NaN, not normalised)."""


# --------------------------------------------------------------------------
# Paths and hashing
# --------------------------------------------------------------------------


def default_data_path() -> Path:
    """Return the dataset path, overridable with ``NDMA_DATA_PATH``.

    Resolved from this file's location, not the working directory (audit F4.3),
    and at call time so tests and deployments can redirect it.
    """
    return Path(os.environ.get("NDMA_DATA_PATH", PROJECT_ROOT / "data" / "pakistan_districts.csv"))


def default_model_path() -> Path:
    """Return the model bundle path, overridable with ``NDMA_MODEL_PATH``."""
    return Path(os.environ.get("NDMA_MODEL_PATH", PROJECT_ROOT / "models" / "best_hazard_pipeline.pkl"))


def manifest_path_for(model_path: Path) -> Path:
    """Return the ``<model>.sha256`` manifest path that accompanies a bundle."""
    return model_path.with_name(model_path.name + ".sha256")


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute the hex SHA-256 digest of a file without loading it into memory.

    Args:
        path: File to hash.
        chunk_size: Bytes read per iteration.

    Returns:
        Lower-case hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_digest(model_path: Path) -> str | None:
    """Return the expected digest from ``NDMA_MODEL_SHA256`` or the manifest file.

    The environment variable wins: anyone able to replace the pickle on disk can
    usually replace a manifest next to it too, so production deployments should
    inject the digest through their secret store.
    """
    if env_digest := os.environ.get("NDMA_MODEL_SHA256"):
        return env_digest.strip().lower()
    manifest = manifest_path_for(model_path)
    if manifest.is_file():
        tokens = manifest.read_text(encoding="utf-8").split()
        return tokens[0].lower() if tokens else None
    return None


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


def load_districts(path: Path) -> pd.DataFrame:
    """Load and validate the district dataset.

    Args:
        path: CSV produced by ``generate_data.py``.

    Returns:
        DataFrame with enforced dtypes and a fresh RangeIndex.

    Raises:
        ArtifactMissingError: The file does not exist.
        SchemaError: Columns, dtypes, nulls, class codes, or uniqueness are violated.
    """
    if not path.is_file():
        raise ArtifactMissingError(f"District dataset not found at '{path.name}'. Run `python generate_data.py`.")
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(CSV_DTYPES) - set(header))
    if missing:
        raise SchemaError(f"District dataset is missing required columns: {missing}")
    try:
        df = pd.read_csv(path, dtype=dict(CSV_DTYPES), usecols=list(CSV_DTYPES))
    except (ValueError, TypeError) as exc:
        raise SchemaError(f"District dataset has invalid values: {exc}") from exc

    if df.empty:
        raise SchemaError("District dataset is empty.")
    null_counts = df.isna().sum()
    if null_counts.any():
        raise SchemaError(
            f"District dataset has missing or unrecognised values in: {null_counts[null_counts > 0].to_dict()}"
        )
    duplicates = df.loc[df["district_name"].duplicated(), "district_name"].tolist()
    if duplicates:
        raise SchemaError(f"District names must be unique; duplicated: {duplicates[:5]}")
    for target in TARGETS:
        if not df[target].isin(range(len(CLASS_LABELS))).all():
            raise SchemaError(f"'{target}' must only contain class codes 0, 1, 2.")
    return df.reset_index(drop=True)


def dataset_labels_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display copy of the dataset's synthetic labels (codes mapped to names)."""
    table = df[["district_name", "province", *TARGETS]].copy()
    for target in TARGETS:
        table[target] = table[target].map(dict(enumerate(CLASS_LABELS)))
    return table.rename(columns={t: f"{TARGET_LABELS[t]} (dataset label)" for t in TARGETS})


# --------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Immutable, validated view of a schema-v2 deployment bundle.

    Attributes:
        models: Calibrated classifier per target; used for every displayed score.
        base_pipelines: The uncalibrated ``Pipeline(prep, clf)`` inside each
            calibrated model; used for SHAP explanations.
        targets: Hazard targets, in display order.
        numeric_features: Union of numeric inputs across all hazards (for validation and sliders).
        categorical_features: Union of categorical inputs across all hazards.
        feature_sets: The only columns each hazard's model receives. Isolation is enforced here:
            a hazard model is never given another hazard's inputs.
        high_thresholds: Per hazard, the calibrated P(High) at or above which High is predicted,
            chosen in training to reduce missed High districts.
        class_labels: Ordered class names (index = class code).
        metadata: Provenance and honest nested-CV metrics written by ``train.py``.
        explainers: SHAP ``TreeExplainer`` per target for scikit-learn forests,
            ``None`` for CatBoost models (explained with CatBoost's native TreeSHAP;
            see ``_shap_for_class``).
        sha256: Digest of the bundle file.
        integrity_verified: True when the digest matched a trusted value.
    """

    models: Mapping[str, Any]
    base_pipelines: Mapping[str, Pipeline]
    targets: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    feature_sets: Mapping[str, tuple[str, ...]]
    high_thresholds: Mapping[str, float]
    class_labels: tuple[str, ...]
    metadata: Mapping[str, Any]
    explainers: Mapping[str, shap.TreeExplainer | None]
    sha256: str
    integrity_verified: bool

    @property
    def features(self) -> tuple[str, ...]:
        """Union of all hazards' input columns."""
        return self.numeric_features + self.categorical_features

    def features_for(self, target: str) -> tuple[str, ...]:
        """The columns the ``target`` hazard model receives, in pipeline order."""
        return self.feature_sets[target]

    @property
    def calibration_method(self) -> str | None:
        """Calibration applied to displayed scores, or None if uncalibrated."""
        return self.metadata.get("calibration_method")


# SHAP/CatBoost do not document thread safety for concurrent explain calls on a
# shared model. One row takes milliseconds, so serialising is cheap insurance.
_SHAP_LOCK: Final = threading.Lock()


def _is_catboost(estimator: Any) -> bool:
    """Whether ``estimator`` is a CatBoost model (checked by class name to keep catboost optional)."""
    return type(estimator).__module__.startswith("catboost")


def _build_explainer(classifier: Any) -> shap.TreeExplainer | None:
    """Build a SHAP explainer, or None for CatBoost.

    ``shap.TreeExplainer`` (shap 0.52) intermittently crashed the whole process
    with a native access violation while parsing the v2 CatBoost models
    (3 of 8 fresh processes; found during the rebuild). A segfault would take
    down the Streamlit server for every user, so CatBoost models are explained
    with CatBoost's own TreeSHAP implementation instead (8 of 8 clean).
    ``tests/test_risk_engine.py::test_explanations_do_not_crash_in_fresh_processes``
    guards this.
    """
    return None if _is_catboost(classifier) else shap.TreeExplainer(classifier)


def _shap_for_class(
    classifier: Any, explainer: shap.TreeExplainer | None, transformed: np.ndarray, class_idx: int
) -> tuple[np.ndarray, float]:
    """Return ``(per-encoded-feature SHAP values, base value)`` for one row and class."""
    with _SHAP_LOCK:
        if explainer is None:
            from catboost import Pool  # imported lazily: only needed when a CatBoost model is served

            # MultiClass layout: (n_rows, n_classes, n_features + 1); the last column is the expected value.
            raw = np.asarray(classifier.get_feature_importance(Pool(transformed), type="ShapValues"))
            return raw[0, class_idx, :-1], float(raw[0, class_idx, -1])
        values = explainer.shap_values(transformed)
        expected = explainer.expected_value
    # SHAP returns (n, features, classes) in current versions, a per-class list in older ones.
    selected = np.asarray(values[class_idx] if isinstance(values, list) else values[..., class_idx])[0]
    return selected, float(np.atleast_1d(expected)[class_idx])


def load_bundle(path: Path) -> ModelBundle:
    """Verify, unpickle, and validate the model bundle.

    Args:
        path: Bundle written by ``train.py``.

    Returns:
        A validated, immutable ``ModelBundle``.

    Raises:
        ArtifactMissingError: The file does not exist.
        ArtifactIntegrityError: The digest does not match the trusted digest.
        SchemaError: Wrong schema version, keys, targets, pipeline steps, or features.
    """
    if not path.is_file():
        raise ArtifactMissingError(f"Model bundle not found at '{path.name}'. Run `python train.py`.")
    digest = file_sha256(path)
    expected = _trusted_digest(path)
    if expected is not None and not hmac.compare_digest(digest, expected):
        raise ArtifactIntegrityError("Model file failed its SHA-256 integrity check and was not loaded.")
    if expected is None:
        logger.warning("No trusted SHA-256 for %s; loading an unverified pickle.", path)

    # joblib.load executes pickle bytecode: only reached after the hash check.
    raw = joblib.load(path)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise SchemaError(
            f"Model bundle is not schema version {BUNDLE_SCHEMA_VERSION}. Retrain with `python train.py`."
        )
    if missing := sorted(REQUIRED_BUNDLE_KEYS - raw.keys()):
        raise SchemaError(f"Model bundle is missing keys: {missing}")
    if missing_meta := sorted(REQUIRED_METADATA_KEYS - raw["metadata"].keys()):
        raise SchemaError(f"Model bundle metadata is missing provenance fields: {missing_meta}")

    targets = tuple(raw["targets"])
    numeric = tuple(raw["numeric_features"])
    categorical = tuple(raw["categorical_features"])
    if tuple(raw["class_labels"]) != CLASS_LABELS:
        raise SchemaError(f"Unexpected class labels {raw['class_labels']}; expected {CLASS_LABELS}.")
    if leaked := sorted((LABEL_DERIVED_COLUMNS | IDENTIFIER_COLUMNS) & {*numeric, *categorical}):
        raise SchemaError(f"Model uses label-derived or identifier columns as inputs (leakage): {leaked}")
    if unknown := sorted({*numeric, *categorical} - set(CSV_DTYPES)):
        raise SchemaError(f"Model expects columns the dataset does not provide: {unknown}")

    feature_sets = {t: tuple(cols) for t, cols in raw["feature_sets"].items()}
    thresholds = {t: float(v) for t, v in raw["high_thresholds"].items()}
    if set(feature_sets) != set(targets) or set(thresholds) != set(targets):
        raise SchemaError("Model bundle must define a feature set and a High threshold for every hazard.")
    for target, cols in feature_sets.items():
        if outside := sorted(set(cols) - {*numeric, *categorical}):
            raise SchemaError(f"Feature set for '{target}' uses undeclared columns: {outside}")
        if not 0.0 < thresholds[target] < 1.0:
            raise SchemaError(f"High threshold for '{target}' must be between 0 and 1.")

    models: dict[str, Any] = {}
    bases: dict[str, Pipeline] = {}
    for target in targets:
        model, base = raw["models"].get(target), raw["base_pipelines"].get(target)
        if not hasattr(model, "predict_proba"):
            raise SchemaError(f"Model for '{target}' does not support predict_proba.")
        if not isinstance(base, Pipeline) or {"prep", "clf"} - base.named_steps.keys():
            raise SchemaError(f"Base pipeline for '{target}' must be a Pipeline with 'prep' and 'clf' steps.")
        models[target], bases[target] = model, base

    explainers = {t: _build_explainer(p.named_steps["clf"]) for t, p in bases.items()}
    logger.info("Loaded model bundle %s (sha256=%s…, verified=%s)", path.name, digest[:12], expected is not None)
    return ModelBundle(
        models=MappingProxyType(models),
        base_pipelines=MappingProxyType(bases),
        targets=targets,
        numeric_features=numeric,
        categorical_features=categorical,
        feature_sets=MappingProxyType(feature_sets),
        high_thresholds=MappingProxyType(thresholds),
        class_labels=CLASS_LABELS,
        metadata=MappingProxyType(dict(raw["metadata"])),
        explainers=MappingProxyType(explainers),
        sha256=digest,
        integrity_verified=expected is not None,
    )


@dataclass(frozen=True, slots=True)
class ProvenanceIssue:
    """One finding from ``audit_provenance``.

    Attributes:
        severity: ``critical`` means displayed predictions may not correspond to the data.
        check: Which provenance check raised it (see ``PROVENANCE_CHECKS``).
        message: User-facing explanation.
    """

    severity: Severity
    check: ProvenanceCheck
    message: str


def _find_robust_scaler(pipeline: Pipeline) -> tuple[RobustScaler, list[str]] | None:
    """Return the fitted RobustScaler and its columns inside ``pipeline``, if any."""
    for _name, transformer, columns in getattr(pipeline.named_steps["prep"], "transformers_", []):
        if isinstance(transformer, RobustScaler) and hasattr(transformer, "center_"):
            return transformer, list(columns)
    return None


def _major_minor(version: str) -> tuple[str, ...]:
    """Return the ``(major, minor)`` part of a version string."""
    return tuple(version.split(".")[:2])


def audit_provenance(bundle: ModelBundle, df: pd.DataFrame, data_sha256: str) -> tuple[ProvenanceIssue, ...]:
    """Check that the model and the dataset actually belong together (audit F1.1, F6.2).

    Three independent checks: the data hash recorded at training time; the
    training medians stored in the fitted RobustScaler (works even if metadata
    were stripped); and the library versions used for training.

    Args:
        bundle: Loaded model bundle.
        df: Loaded district dataset.
        data_sha256: Digest of the dataset file.

    Returns:
        Issues found, critical first. Empty when everything is consistent.
    """
    issues: list[ProvenanceIssue] = []
    recorded = bundle.metadata.get("data_sha256")
    if recorded != data_sha256:
        issues.append(
            ProvenanceIssue(
                "critical",
                "data",
                "The model was trained on a different version of the district dataset. Retrain with `python train.py`.",
            )
        )

    for pipeline in bundle.base_pipelines.values():
        found = _find_robust_scaler(pipeline)
        if found is None:
            continue
        scaler, columns = found
        medians = df[columns].median().to_numpy(dtype=float)
        if not np.allclose(scaler.center_, medians, rtol=1e-6, atol=1e-9):
            gaps = np.abs(scaler.center_ - medians) / np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
            worst = int(np.argmax(gaps))
            issues.append(
                ProvenanceIssue(
                    "critical",
                    "data",
                    f"Model training median of '{columns[worst]}' is {scaler.center_[worst]:,.2f} but the dataset "
                    f"median is {medians[worst]:,.2f}: the model was fitted on different data.",
                )
            )
            break  # one mismatch is enough to prove the data differs

    for package, trained in bundle.metadata.get("library_versions", {}).items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            installed = "not installed"
        if installed != trained:
            severity: Severity = "critical" if _major_minor(installed) != _major_minor(trained) else "info"
            issues.append(
                ProvenanceIssue(
                    severity, "libraries", f"{package} {installed} is installed; the model was trained with {trained}."
                )
            )

    if not bundle.integrity_verified:
        # Critical, not info: the pickle was executed without a digest check, so the served model is unproven.
        issues.append(
            ProvenanceIssue(
                "critical",
                "integrity",
                "No trusted SHA-256 digest is configured, so the model file's integrity is unverified.",
            )
        )
    if bundle.calibration_method is None:
        issues.append(
            ProvenanceIssue(
                "info", "calibration", "Scores are not calibrated probabilities; treat them as relative model scores."
            )
        )
    return tuple(sorted(issues, key=lambda issue: issue.severity != "critical"))


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    """One row of the model card's provenance checklist.

    Attributes:
        label: What is checked.
        passed: True when no issue was raised for this check.
        detail: Evidence when passed, or the issue messages when not.
    """

    label: str
    passed: bool
    detail: str


def provenance_checklist(bundle: ModelBundle, issues: Sequence[ProvenanceIssue]) -> tuple[ChecklistItem, ...]:
    """Turn provenance issues into a pass/fail checklist covering every check, including passes."""
    evidence = {
        "integrity": f"SHA-256 {bundle.sha256[:12]}… matches the trusted digest.",
        "data": f"Data SHA-256 {str(bundle.metadata.get('data_sha256', ''))[:12]}… matches; training medians agree.",
        "libraries": ", ".join(f"{k} {v}" for k, v in bundle.metadata.get("library_versions", {}).items()),
        "calibration": f"{str(bundle.calibration_method).capitalize()} scaling.",
    }
    items = []
    for check, label in PROVENANCE_CHECKS.items():
        raised = [i.message for i in issues if i.check == check]
        items.append(ChecklistItem(label, not raised, " ".join(raised) if raised else evidence[check]))
    return tuple(items)


# --------------------------------------------------------------------------
# Scenarios and applicability domain
# --------------------------------------------------------------------------


def build_scenario(df: pd.DataFrame, district: str, overrides: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Build a one-row model input for a district, optionally with overrides.

    Args:
        df: Validated district dataset.
        district: Exact district name.
        overrides: Values for features in ``TUNABLE_FEATURES``.

    Returns:
        A one-row DataFrame that preserves the dataset's dtypes.

    Raises:
        ScenarioValidationError: Unknown district or feature, a non-finite value,
            or a value outside the feature's physical bounds.
    """
    matches = df.index[df["district_name"] == district]
    if len(matches) != 1:
        raise ScenarioValidationError(f"Unknown district: {district!r}")
    scenario = df.loc[[matches[0]]].copy()
    for name, value in (overrides or {}).items():
        spec = TUNABLE_BY_NAME.get(name)
        if spec is None:
            raise ScenarioValidationError(f"'{name}' cannot be changed in a scenario.")
        number = float(value)
        if not math.isfinite(number) or not spec.lower <= number <= spec.upper:
            raise ScenarioValidationError(
                f"{spec.label} must be between {spec.format(spec.lower)} and {spec.format(spec.upper)}."
            )
        scenario[name] = number
    return scenario.reset_index(drop=True)


def changed_features(baseline: pd.DataFrame, scenario: pd.DataFrame) -> tuple[str, ...]:
    """Return tunable features whose scenario value differs from the baseline."""
    return tuple(
        spec.name
        for spec in TUNABLE_FEATURES
        if not math.isclose(
            float(baseline[spec.name].iloc[0]), float(scenario[spec.name].iloc[0]), rel_tol=1e-9, abs_tol=1e-9
        )
    )


@dataclass(frozen=True, slots=True)
class ApplicabilityDomain:
    """Where the model has seen data, and how unusual a scenario is (audit F4.1).

    Tree ensembles cannot extrapolate: beyond the largest training value a
    feature's effect is frozen. Features can also each be in range while their
    combination is unlike any training row.

    Attributes:
        features: Numeric features considered.
        lower: Smallest training value per feature.
        upper: Largest training value per feature.
        center: Per-feature median used for robust scaling.
        scale: Per-feature IQR used for robust scaling (0 replaced by 1).
        reference: Robust-scaled training matrix (read-only).
        novelty_threshold: 95th percentile of leave-one-out nearest-neighbour
            distances among training rows.
    """

    features: tuple[str, ...]
    lower: Mapping[str, float]
    upper: Mapping[str, float]
    center: np.ndarray
    scale: np.ndarray
    reference: np.ndarray
    novelty_threshold: float

    @classmethod
    def fit(cls, df: pd.DataFrame, features: Sequence[str], quantile: float = 0.95) -> ApplicabilityDomain:
        """Fit the domain from the training dataset.

        Args:
            df: Dataset the model was trained on.
            features: Numeric features to include.
            quantile: Nearest-neighbour distance quantile above which a scenario
                counts as novel. 0.95 flags combinations more isolated than 95%
                of real districts are from each other (judgment).

        Returns:
            A fitted, immutable ``ApplicabilityDomain``.
        """
        matrix = df[list(features)].to_numpy(dtype=float)
        center = np.median(matrix, axis=0)
        iqr = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
        scale = np.where(iqr > 0, iqr, 1.0)
        reference = (matrix - center) / scale
        distances = np.linalg.norm(reference[:, None, :] - reference[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        threshold = float(np.quantile(distances.min(axis=1), quantile))
        for array in (center, scale, reference):
            array.flags.writeable = False
        return cls(
            features=tuple(features),
            lower=MappingProxyType(dict(zip(features, matrix.min(axis=0).tolist(), strict=True))),
            upper=MappingProxyType(dict(zip(features, matrix.max(axis=0).tolist(), strict=True))),
            center=center,
            scale=scale,
            reference=reference,
            novelty_threshold=threshold,
        )

    def novelty(self, scenario: pd.DataFrame) -> float:
        """Distance to the nearest training district divided by the threshold (>1 is novel)."""
        row = (scenario[list(self.features)].to_numpy(dtype=float)[0] - self.center) / self.scale
        nearest = float(np.linalg.norm(self.reference - row, axis=1).min())
        return nearest / self.novelty_threshold if self.novelty_threshold > 0 else 0.0

    def out_of_range(self, scenario: pd.DataFrame) -> tuple[str, ...]:
        """Return features whose scenario value lies outside the training range."""
        return tuple(f for f in self.features if not self.lower[f] <= float(scenario[f].iloc[0]) <= self.upper[f])

    def assess(self, scenario: pd.DataFrame) -> tuple[str, ...]:
        """Return user-facing extrapolation warnings for one scenario row."""
        warnings = [
            f"{FEATURE_LABELS.get(f, f)} = {float(scenario[f].iloc[0]):,.2f} is outside the training range "
            f"({self.lower[f]:,.2f}–{self.upper[f]:,.2f}); the model treats it as the nearest edge value."
            for f in self.out_of_range(scenario)
        ]
        if self.novelty(scenario) > 1.0:
            warnings.append(
                "This combination of conditions is unlike any district in the training data; treat the scores as low-confidence."
            )
        return tuple(warnings)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HazardPrediction:
    """Calibrated model output for one hazard.

    Attributes:
        target: Hazard target name.
        probabilities: Score per class, index-aligned with ``CLASS_LABELS``.
        high_threshold: If set, High is predicted whenever P(High) reaches it, even when
            Low or Medium is more probable (recall-oriented decision rule).
    """

    target: str
    probabilities: tuple[float, ...]
    high_threshold: float | None = None

    @property
    def class_idx(self) -> int:
        """Predicted class under the decision rule; exact ties resolve to the *more severe* class."""
        return decide_class(np.asarray([self.probabilities]), self.high_threshold)[0].item()

    @property
    def label(self) -> str:
        """Predicted class name."""
        return CLASS_LABELS[self.class_idx]

    @property
    def p_high(self) -> float:
        """Probability of the most severe class."""
        return self.probabilities[-1]


def decide_class(proba: np.ndarray, high_threshold: float | None) -> np.ndarray:
    """Apply the decision rule to rows of class probabilities.

    Without a threshold: argmax, exact ties to the more severe class. With a threshold:
    High when P(High) >= threshold, otherwise the more probable of Low and Medium.
    Shared by training, single-district prediction, and the national map so they cannot disagree.
    """
    severe_first = proba[:, ::-1]
    argmax = len(CLASS_LABELS) - 1 - severe_first.argmax(axis=1)
    if high_threshold is None:
        return argmax
    low_or_medium = np.where(proba[:, 1] >= proba[:, 0], 1, 0)
    return np.where(proba[:, -1] >= high_threshold, len(CLASS_LABELS) - 1, low_or_medium)


def _validated_input(bundle: ModelBundle, scenario: pd.DataFrame, target: str) -> pd.DataFrame:
    """Select exactly the ``target`` hazard's input columns from a one-row scenario."""
    if len(scenario) != 1:
        raise ScenarioValidationError("A scenario must contain exactly one district row.")
    columns = list(bundle.features_for(target))
    if missing := [f for f in columns if f not in scenario.columns]:
        raise SchemaError(f"Scenario is missing model features: {missing}")
    return scenario[columns]


def predict_hazards(bundle: ModelBundle, scenario: pd.DataFrame) -> dict[str, HazardPrediction]:
    """Score every hazard for a one-row scenario with the calibrated models.

    Uses ``predict_proba`` only: ``predict`` has inconsistent output shapes across
    estimator families (CatBoost returns ``(n, 1)``; audit F2.1).

    Raises:
        ScenarioValidationError: The scenario is not exactly one row.
        SchemaError: Model features are missing from the scenario.
        ModelOutputError: A model returned malformed probabilities.
    """
    predictions: dict[str, HazardPrediction] = {}
    for target in bundle.targets:
        features = _validated_input(bundle, scenario, target)
        proba = np.asarray(bundle.models[target].predict_proba(features), dtype=float)
        if (
            proba.shape != (1, len(CLASS_LABELS))
            or not np.isfinite(proba).all()
            or not math.isclose(float(proba.sum()), 1.0, abs_tol=1e-6)
        ):
            raise ModelOutputError(f"The {TARGET_LABELS.get(target, target)} model returned invalid scores.")
        predictions[target] = HazardPrediction(target, tuple(proba[0].tolist()), bundle.high_thresholds[target])
    return predictions


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contribution:
    """SHAP contribution of one original (pre-encoding) feature.

    Attributes:
        feature: Source column name.
        label: Human-readable feature name.
        display_value: Value in original units.
        value: SHAP value; one-hot columns are summed back into their source.
    """

    feature: str
    label: str
    display_value: str
    value: float


@dataclass(frozen=True, slots=True)
class HazardExplanation:
    """Additive explanation of one class score of the *uncalibrated* base model.

    Attributes:
        target: Hazard target name.
        class_idx: Explained class.
        base_value: Average base-model output over the background data.
        contributions: Per-feature contributions, largest magnitude first.
        output_space: ``probability`` (sums to the base model's probability) or
            ``log-odds`` (sums to the raw margin before softmax, e.g. CatBoost).
    """

    target: str
    class_idx: int
    base_value: float
    contributions: tuple[Contribution, ...]
    output_space: OutputSpace

    @property
    def final_value(self) -> float:
        """Base value plus all contributions (the explained base-model output)."""
        return self.base_value + sum(c.value for c in self.contributions)


def _source_feature(encoded_name: str, numeric: Sequence[str], categorical: Sequence[str]) -> str:
    """Map a ColumnTransformer output name (``cat__province_Sindh``) to its source column."""
    name = encoded_name.split("__", 1)[-1]
    if name in numeric:
        return name
    for column in sorted(categorical, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    raise SchemaError(f"Cannot map encoded feature '{encoded_name}' to a source column.")


def _display_value(feature: str, raw: Any) -> str:
    """Format a raw feature value in original units for explanation labels."""
    if feature in FEATURE_SPECS:
        return FEATURE_SPECS[feature].format(float(raw))
    return str(raw)


def explain_hazard(bundle: ModelBundle, scenario: pd.DataFrame, target: str, class_idx: int) -> HazardExplanation:
    """Compute a SHAP explanation for one hazard class on a one-row scenario.

    One-hot columns are folded back into their source feature (SHAP values are
    additive, so summing them is exact); this fixes the unreadable
    ``cat__province_...`` labels of v1 (audit F2.1).

    Args:
        bundle: Loaded model bundle.
        scenario: One-row scenario from ``build_scenario``.
        target: Hazard to explain.
        class_idx: Class whose score is explained.

    Returns:
        Explanation of the base model's output for that class.

    Raises:
        ScenarioValidationError: Unknown target or class, or not one row.
        ModelOutputError: SHAP output has an unexpected shape.
    """
    if target not in bundle.base_pipelines or not 0 <= class_idx < len(CLASS_LABELS):
        raise ScenarioValidationError(f"Cannot explain target={target!r}, class={class_idx}.")
    features = _validated_input(bundle, scenario, target)
    pipeline = bundle.base_pipelines[target]
    prep = pipeline.named_steps["prep"]
    transformed = prep.transform(features)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    encoded_names = prep.get_feature_names_out()

    values, base = _shap_for_class(pipeline.named_steps["clf"], bundle.explainers[target], transformed, class_idx)
    if values.shape != (len(encoded_names),):
        raise ModelOutputError("SHAP returned an unexpected shape.")

    totals: dict[str, float] = {}
    for encoded, value in zip(encoded_names, values, strict=True):
        source = _source_feature(str(encoded), features.columns.tolist(), bundle.categorical_features)
        totals[source] = totals.get(source, 0.0) + float(value)

    contributions = tuple(
        sorted(
            (
                Contribution(f, FEATURE_LABELS.get(f, f), _display_value(f, features[f].iloc[0]), v)
                for f, v in totals.items()
            ),
            key=lambda c: abs(c.value),
            reverse=True,
        )
    )
    base_score = float(pipeline.predict_proba(features)[0, class_idx])
    space: OutputSpace = (
        "probability" if math.isclose(base + float(values.sum()), base_score, abs_tol=1e-4) else "log-odds"
    )
    return HazardExplanation(target, class_idx, base, contributions, space)


def format_contribution(value: float, space: OutputSpace) -> str:
    """Format a SHAP value with its unit: percentage points or log-odds."""
    return f"{value * 100:+.1f} pts" if space == "probability" else f"{value:+.2f} log-odds"


#: A driver below this share of total attribution is not worth naming (judgment).
MINOR_SHARE: Final[float] = 0.10
#: The top driver "dominates" when at least this many times larger than the second (judgment).
DOMINANCE_RATIO: Final[float] = 2.0


def narrate_explanation(explanation: HazardExplanation) -> str:
    """Write a one-sentence insight whose structure is chosen by the top-2 SHAP values.

    No fixed sentence is filled in. The clauses are selected by three
    properties of the explanation at inference time:

    1. whether the second driver matters at all (share of total |SHAP|),
    2. whether the top two drivers push in the same or opposite directions,
    3. whether the first driver dominates the second (magnitude ratio).

    Args:
        explanation: Output of ``explain_hazard``.

    Returns:
        A plain-language sentence, or a statement that no feature stands out.
    """
    contributions = explanation.contributions
    total = sum(abs(c.value) for c in contributions)
    score_name = f"{CLASS_LABELS[explanation.class_idx]} {TARGET_LABELS[explanation.target].lower()} score"
    if not contributions or total == 0.0:
        return f"No feature moves the {score_name} away from the average district."

    def verb(c: Contribution) -> str:
        return "raises" if c.value > 0 else "lowers"

    def name(c: Contribution) -> str:
        return f"{c.label} = {c.display_value}"

    def amount(c: Contribution) -> str:
        return format_contribution(abs(c.value), explanation.output_space).lstrip("+")

    first = contributions[0]
    share_first = abs(first.value) / total
    second = contributions[1] if len(contributions) > 1 else None
    share_second = abs(second.value) / total if second else 0.0

    if second is None or share_second < MINOR_SHARE:
        return (
            f"{name(first)} {verb(first)} the {score_name} by {amount(first)}, "
            f"{share_first:.0%} of all feature influence; no other feature accounts for more than {share_second:.0%}."
        )
    if (first.value > 0) == (second.value > 0):
        lead = "mainly " if abs(first.value) >= DOMINANCE_RATIO * abs(second.value) else ""
        return (
            f"The {score_name} is {'pushed up' if first.value > 0 else 'held down'} {lead}by {name(first)} ({amount(first)}) "
            f"and {name(second)} ({amount(second)}), together {share_first + share_second:.0%} of all feature influence."
        )
    qualifier = "partly offset" if abs(first.value) >= DOMINANCE_RATIO * abs(second.value) else "largely offset"
    return (
        f"{name(first)} {verb(first)} the {score_name} by {amount(first)}, {qualifier} by "
        f"{name(second)}, which {verb(second)} it by {amount(second)}."
    )


# --------------------------------------------------------------------------
# Assessment export (CSV / TXT)
# --------------------------------------------------------------------------

_FORMULA_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection in text cells (OWASP CSV injection)."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


@dataclass(frozen=True, slots=True)
class Assessment:
    """Everything shown on screen for one district and scenario.

    Attributes:
        district: District name.
        province: Province name.
        scenario: One-row scenario that was scored.
        baseline: One-row unmodified district inputs.
        predictions: Calibrated predictions for the scenario.
        baseline_predictions: Calibrated predictions for the unmodified inputs.
        explanations: SHAP explanation of the High score per target.
        insights: One narrated sentence per target.
        warnings: Extrapolation and provenance warnings shown to the user.
        model_sha256: Digest of the model bundle used.
        generated_at: UTC timestamp.
    """

    district: str
    province: str
    scenario: pd.DataFrame
    baseline: pd.DataFrame
    predictions: Mapping[str, HazardPrediction]
    baseline_predictions: Mapping[str, HazardPrediction]
    explanations: Mapping[str, HazardExplanation]
    insights: Mapping[str, str]
    warnings: tuple[str, ...]
    model_sha256: str
    generated_at: datetime

    @property
    def modified_features(self) -> tuple[str, ...]:
        """Tunable inputs changed from the district's dataset values."""
        return changed_features(self.baseline, self.scenario)


def build_assessment(
    bundle: ModelBundle,
    df: pd.DataFrame,
    domain: ApplicabilityDomain,
    district: str,
    overrides: Mapping[str, float] | None = None,
    extra_warnings: Sequence[str] = (),
) -> Assessment:
    """Score, explain, and assess one district scenario end to end.

    Args:
        bundle: Loaded model bundle.
        df: Validated district dataset.
        domain: Applicability domain fitted on the training data.
        district: District name.
        overrides: Tunable feature overrides from the simulator.
        extra_warnings: Additional warnings to record (e.g. provenance issues).

    Returns:
        A complete ``Assessment``.
    """
    baseline = build_scenario(df, district)
    scenario = build_scenario(df, district, overrides)
    predictions = predict_hazards(bundle, scenario)
    explanations = {t: explain_hazard(bundle, scenario, t, len(CLASS_LABELS) - 1) for t in bundle.targets}
    return Assessment(
        district=district,
        province=str(scenario["province"].iloc[0]),
        scenario=scenario,
        baseline=baseline,
        predictions=predictions,
        baseline_predictions=predict_hazards(bundle, baseline),
        explanations=MappingProxyType(explanations),
        insights=MappingProxyType({t: narrate_explanation(e) for t, e in explanations.items()}),
        warnings=(*domain.assess(scenario), *extra_warnings),
        model_sha256=bundle.sha256,
        generated_at=datetime.now(UTC),
    )


EXPORT_DISCLAIMER: Final[str] = (
    "SIMULATED SCENARIO - not an operational forecast. Synthetic data; portfolio demonstration only."
)


def assessment_to_csv(assessment: Assessment) -> str:
    """Serialise an assessment as a CSV table with one row per hazard.

    Returns:
        CSV text (UTF-8) with formula-injection-safe text cells.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "generated_at_utc",
            "district",
            "province",
            "simulated_inputs_changed",
            "hazard",
            "predicted_level",
            "p_low",
            "p_medium",
            "p_high",
            "baseline_p_high",
            "p_high_change_pts",
            "insight",
            "model_sha256",
            "disclaimer",
        ]
    )
    changed = ";".join(assessment.modified_features) or "none"
    for target, prediction in assessment.predictions.items():
        base = assessment.baseline_predictions[target]
        writer.writerow(
            [
                _csv_safe(v)
                for v in (
                    assessment.generated_at.isoformat(timespec="seconds"),
                    assessment.district,
                    assessment.province,
                    changed,
                    TARGET_LABELS[target],
                    prediction.label,
                    *(round(p, 4) for p in prediction.probabilities),
                    round(base.p_high, 4),
                    round((prediction.p_high - base.p_high) * 100, 2),
                    assessment.insights[target],
                    assessment.model_sha256,
                    EXPORT_DISCLAIMER,
                )
            ]
        )
    return buffer.getvalue()


def assessment_to_text(assessment: Assessment) -> str:
    """Serialise an assessment as a human-readable plain-text report."""
    lines = [
        EXPORT_DISCLAIMER,
        "",
        f"District: {assessment.district} ({assessment.province})",
        f"Generated: {assessment.generated_at:%Y-%m-%d %H:%M} UTC",
        f"Model bundle SHA-256: {assessment.model_sha256}",
        "",
        "Inputs",
    ]
    for spec in TUNABLE_FEATURES:
        value = float(assessment.scenario[spec.name].iloc[0])
        base = float(assessment.baseline[spec.name].iloc[0])
        suffix = f"   (dataset value {spec.format(base)})" if spec.name in assessment.modified_features else ""
        lines.append(f"  {spec.label}: {spec.format(value)}{suffix}")
    lines += ["", "Hazard scores (calibrated probabilities)"]
    for target, prediction in assessment.predictions.items():
        base = assessment.baseline_predictions[target]
        lines.append(
            f"  {TARGET_LABELS[target]}: {prediction.label}  P(High) {prediction.p_high:.1%}"
            f"  (unmodified inputs {base.p_high:.1%}, change {(prediction.p_high - base.p_high) * 100:+.1f} pts)"
        )
        lines.append(f"    Why: {assessment.insights[target]}")
    lines += ["", "Warnings"] + ([f"  - {w}" for w in assessment.warnings] or ["  none"])
    lines += [
        "",
        "Note: 'change' compares against this district's unmodified dataset inputs, not against historical",
        "observations. No historical event dataset exists in this project.",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Dashboard views (UI enhancement: national map, district profile)
# --------------------------------------------------------------------------


def predict_dataset(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Calibrated predictions for every district's unmodified inputs, one row per district and hazard.

    Uses ``decide_class`` with each hazard's High threshold, exactly as ``HazardPrediction``
    does, so the map and the KPI cards always agree for an unmodified district.

    Returns:
        Columns ``district_name, province, latitude, longitude, target, label, p_high``.

    Raises:
        ModelOutputError: A model returned malformed probabilities.
    """
    frames = []
    for target in bundle.targets:
        proba = np.asarray(bundle.models[target].predict_proba(df[list(bundle.features_for(target))]), dtype=float)
        if proba.shape != (len(df), len(CLASS_LABELS)) or not np.isfinite(proba).all():
            raise ModelOutputError(f"The {TARGET_LABELS.get(target, target)} model returned invalid scores.")
        class_idx = decide_class(proba, bundle.high_thresholds[target])
        frames.append(
            pd.DataFrame(
                {
                    "district_name": df["district_name"].astype(str).to_numpy(),
                    "province": df["province"].astype(str).to_numpy(),
                    "latitude": df["latitude"].to_numpy(dtype=float),
                    "longitude": df["longitude"].to_numpy(dtype=float),
                    "target": target,
                    "label": np.asarray(CLASS_LABELS)[class_idx],
                    "p_high": proba[:, -1],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


#: Rows shown in the district profile: tunable inputs first, then fixed real geography.
PROFILE_FEATURES: Final[tuple[str, ...]] = (
    *(spec.name for spec in TUNABLE_FEATURES),
    "elevation_m",
    "active_fault_distance_km",
    "makran_subduction_distance_km",
    "coast_distance_km",
)


def district_profile(df: pd.DataFrame, assessment: Assessment) -> pd.DataFrame:
    """Compare the scored scenario with the district's data, its province, and all of Pakistan.

    Returns:
        One row per ``PROFILE_FEATURES`` entry with formatted, unit-labelled values and a
        ``Changed`` flag for simulated inputs.
    """
    province_rows = df[df["province"].astype(str) == assessment.province]
    rows = []
    for name in PROFILE_FEATURES:
        spec = FEATURE_SPECS[name]
        rows.append(
            {
                "Input": spec.label,
                "This scenario": spec.format(float(assessment.scenario[name].iloc[0])),
                "Changed": name in assessment.modified_features,
                "Dataset value": spec.format(float(assessment.baseline[name].iloc[0])),
                "Province median": spec.format(float(province_rows[name].median())),
                "Pakistan range": f"{spec.format(float(df[name].min()))} – {spec.format(float(df[name].max()))}",
                "Source": "Real" if name in _REAL_PROFILE_FEATURES else "Synthetic",
            }
        )
    return pd.DataFrame(rows)


_REAL_PROFILE_FEATURES: Final[frozenset[str]] = frozenset(
    {"elevation_m", "active_fault_distance_km", "makran_subduction_distance_km", "coast_distance_km"}
)


@dataclass(frozen=True, slots=True)
class MapLayers:
    """Real context geometry for the national map (GeoJSON dicts, coordinates rounded to ~100 m).

    Attributes:
        faults: Active crustal faults and the Makran subduction segments (GEM).
        coastline: Natural Earth coastline.
    """

    faults: Mapping[str, Any]
    coastline: Mapping[str, Any]


def _rounded_geojson(path: Path, decimals: int = 3) -> dict[str, Any]:
    """Load a line GeoJSON with coordinates rounded to shrink the browser payload."""
    collection = json.loads(path.read_text(encoding="utf-8"))
    for feature in collection["features"]:
        geometry = feature["geometry"]
        parts = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        rounded = [[[round(x, decimals), round(y, decimals)] for x, y, *_ in part] for part in parts]
        feature["geometry"] = {"type": "MultiLineString", "coordinates": rounded}
    return collection


def load_map_layers(reference_dir: Path | None = None) -> MapLayers:
    """Load the real fault and coastline layers from ``data/reference``.

    Raises:
        ArtifactMissingError: A reference layer is missing.
    """
    directory = reference_dir or PROJECT_ROOT / "data" / "reference"
    paths = {"faults": directory / "active_faults.geojson", "coastline": directory / "coastline.geojson"}
    for path in paths.values():
        if not path.is_file():
            raise ArtifactMissingError(
                f"Reference layer '{path.name}' not found. Run `python scripts/build_reference_data.py`."
            )
    return MapLayers(
        faults=MappingProxyType(_rounded_geojson(paths["faults"])),
        coastline=MappingProxyType(_rounded_geojson(paths["coastline"])),
    )


# --------------------------------------------------------------------------
# Planning tools: climate stress test and multi-hazard hotspots
# --------------------------------------------------------------------------

#: Allowed stress-test settings. Wider shifts would push most districts outside the training
#: range, where tree models freeze and results would be mostly extrapolated (judgment).
STRESS_TEMP_SHIFT_RANGE: Final[tuple[float, float]] = (0.0, 4.0)
STRESS_RAIN_FACTOR_RANGE: Final[tuple[float, float]] = (0.5, 2.0)
#: The inputs a stress test changes; every other input keeps its dataset value.
STRESS_FEATURES: Final[tuple[str, ...]] = ("summer_max_temp_c", "avg_annual_rainfall_mm")


def stressed_dataset(df: pd.DataFrame, temp_shift_c: float, rain_factor: float) -> pd.DataFrame:
    """Apply a uniform climate shift to every district.

    Only the two stated inputs change. Dependent synthetic inputs (e.g. NDVI, which the generator
    derives from rainfall) are deliberately left unchanged, so the scenario is exactly what the
    user set and nothing is modelled implicitly.

    Raises:
        ScenarioValidationError: A shift is outside the allowed range.
    """
    low_t, high_t = STRESS_TEMP_SHIFT_RANGE
    low_r, high_r = STRESS_RAIN_FACTOR_RANGE
    if not (math.isfinite(temp_shift_c) and low_t <= temp_shift_c <= high_t):
        raise ScenarioValidationError(f"Temperature shift must be between +{low_t:g} and +{high_t:g} °C.")
    if not (math.isfinite(rain_factor) and low_r <= rain_factor <= high_r):
        raise ScenarioValidationError(f"Rainfall factor must be between ×{low_r:g} and ×{high_r:g}.")
    stressed = df.copy()
    stressed["summer_max_temp_c"] = stressed["summer_max_temp_c"] + temp_shift_c
    stressed["avg_annual_rainfall_mm"] = stressed["avg_annual_rainfall_mm"] * rain_factor
    return stressed


def stress_test(
    bundle: ModelBundle, df: pd.DataFrame, domain: ApplicabilityDomain, temp_shift_c: float, rain_factor: float
) -> pd.DataFrame:
    """Re-score every district under a climate shift with the deployed, input-isolated models.

    Returns:
        One row per district and hazard: current and stressed label and P(High), the change in
        percentage points, ``becomes_high`` (not High now, High under stress), and ``extrapolated``
        (a stressed input *used by that hazard's model* falls outside the training range, so the
        tree model is holding its edge value and the result deserves less confidence).
    """
    current = predict_dataset(bundle, df)
    stressed_df = stressed_dataset(df, temp_shift_c, rain_factor)
    stressed = predict_dataset(bundle, stressed_df)
    result = current.rename(columns={"label": "current_label", "p_high": "current_p_high"})
    result["stressed_label"] = stressed["label"].to_numpy()
    result["stressed_p_high"] = stressed["p_high"].to_numpy()
    result["change_pts"] = (result["stressed_p_high"] - result["current_p_high"]) * 100
    result["becomes_high"] = (result["stressed_label"] == "High") & (result["current_label"] != "High")

    outside: dict[str, np.ndarray] = {
        feature: ~stressed_df[feature].between(domain.lower[feature], domain.upper[feature]).to_numpy()
        for feature in STRESS_FEATURES
    }
    flags = np.zeros(len(result), dtype=bool)
    for target in bundle.targets:
        rows = (result["target"] == target).to_numpy()
        used = [f for f in STRESS_FEATURES if f in bundle.features_for(target)]
        per_district = np.any([outside[f] for f in used], axis=0) if used else np.zeros(len(df), dtype=bool)
        flags[rows] = per_district
    result["extrapolated"] = flags
    return result


def multi_hazard_hotspots(
    predictions: pd.DataFrame, label_column: str = "label", p_column: str = "p_high"
) -> pd.DataFrame:
    """Districts predicted High for two or more hazards at once.

    Args:
        predictions: Long-format predictions (``predict_dataset`` output, or ``stress_test`` output
            with ``label_column="stressed_label"``).
        label_column: Column holding the predicted level.
        p_column: Column holding P(High).

    Returns:
        ``district_name, province, high_count, high_hazards, combined_p_high``, sorted with the most
        hazards first, then by combined P(High).
    """
    highs = predictions[predictions[label_column] == "High"]
    grouped = highs.groupby(["district_name", "province"], sort=False).agg(
        high_count=("target", "size"),
        high_hazards=("target", lambda t: ", ".join(TARGET_LABELS[x] for x in sorted(t, key=TARGETS.index))),
        combined_p_high=(p_column, "sum"),
    )
    hotspots = grouped[grouped["high_count"] >= 2].reset_index()
    return hotspots.sort_values(["high_count", "combined_p_high"], ascending=False, ignore_index=True)
