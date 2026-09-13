"""Shared fixtures. Tests run against the committed dataset and model bundle.

The project root is put on ``sys.path`` by ``[tool.pytest.ini_options] pythonpath`` in pyproject.toml.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import risk_engine as engine


@pytest.fixture(scope="session")
def data_path() -> Path:
    return engine.default_data_path()


@pytest.fixture(scope="session")
def df(data_path: Path) -> pd.DataFrame:
    return engine.load_districts(data_path)


@pytest.fixture(scope="session")
def bundle() -> engine.ModelBundle:
    return engine.load_bundle(engine.default_model_path())


@pytest.fixture(scope="session")
def domain(df: pd.DataFrame, bundle: engine.ModelBundle) -> engine.ApplicabilityDomain:
    return engine.ApplicabilityDomain.fit(df, bundle.numeric_features)
