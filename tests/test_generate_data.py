"""Dataset integrity tests (audit F1.6)."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

import generate_data as gen

ROOT = gen.PROJECT_ROOT
REFERENCE = ROOT / "data" / "reference"


@pytest.fixture(scope="module")
def committed() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "pakistan_districts.csv")


def test_committed_csv_is_reproducible_from_seed(committed: pd.DataFrame) -> None:
    fresh = pd.read_csv(io.StringIO(gen.generate_dataset().to_csv(index=False)))
    pd.testing.assert_frame_equal(committed, fresh)


def test_every_value_is_within_physical_bounds(committed: pd.DataFrame) -> None:
    gen.validate_physical_bounds(committed)  # F1.6-c: raises on any impossible value


@pytest.mark.parametrize(
    ("column", "value"), [("elevation_m", 9_000.0), ("population_density", 0.0), ("summer_max_temp_c", 55.0)]
)
def test_bounds_validation_rejects_impossible_values(committed: pd.DataFrame, column: str, value: float) -> None:
    broken = committed.copy()
    broken.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        gen.validate_physical_bounds(broken)


def test_districts_are_real_geonames_points(committed: pd.DataFrame) -> None:
    reference = pd.read_csv(REFERENCE / "districts.csv")
    merged = committed.merge(reference, on="geonames_id", suffixes=("", "_ref"), validate="one_to_one")
    assert len(merged) == len(committed) == 150
    assert (merged["district_name"] == merged["district_name_ref"]).all()
    np.testing.assert_allclose(merged["latitude"], merged["latitude_ref"], atol=1e-5)  # F1.6-a
    np.testing.assert_allclose(merged["longitude"], merged["longitude_ref"], atol=1e-5)
    assert (merged["elevation_m"] == merged["elevation_m_ref"]).all()


def test_no_invented_district_names(committed: pd.DataFrame) -> None:
    assert not committed["district_name"].str.contains(r"Rural-|Extension-").any()  # F1.6-b
    assert committed["district_name"].is_unique


def test_label_derived_and_collinear_columns_are_gone(committed: pd.DataFrame) -> None:
    assert "overall_risk_score" not in committed.columns  # F1.6-g
    assert "historical_disasters" not in committed.columns  # F1.6-e
    assert not any(c.startswith("historical_") for c in committed.columns)  # F1.6-j


def test_distance_function_matches_haversine_at_a_vertex() -> None:
    # Clamped projection onto a segment endpoint must equal the great-circle distance to that point.
    line = np.array([[67.0, 25.0], [67.0, 24.0]])  # runs south from (25N, 67E)
    lat, lon = np.array([26.0]), np.array([67.5])  # north-east of the northern end
    ours = gen.distance_to_polylines_km(lat, lon, [line])[0]
    reference = gen.haversine_km(lat, lon, 25.0, 67.0)[0]
    assert ours == pytest.approx(reference, rel=0.01)


def test_seismic_geometry_is_physically_ordered(committed: pd.DataFrame) -> None:
    by_name = committed.set_index("district_name")
    # Makran coast is closer to the subduction zone than Karachi, which is closer than Lahore.
    assert (
        by_name.loc["Gwadar", "makran_subduction_distance_km"]
        < by_name.loc["Karachi South", "makran_subduction_distance_km"]
        < by_name.loc["Lahore", "makran_subduction_distance_km"]
    )
    # The Chaman fault zone runs through Qila Abdullah district (F1.6-d).
    assert by_name.loc["Qila Abdullah", "active_fault_distance_km"] < 25.0


def test_chaman_segment_identification_holds() -> None:
    faults = json.loads((REFERENCE / "active_faults.geojson").read_text(encoding="utf-8"))
    chaman_lat, chaman_lon = 30.91769, 66.45259  # GeoNames populated place "Chaman"
    nearest = min(
        (f for f in faults["features"] if f["properties"]["role"] == "crustal"),
        key=lambda f: gen.distance_to_polylines_km(
            np.array([chaman_lat]), np.array([chaman_lon]), [np.asarray(p) for p in f["geometry"]["coordinates"]]
        )[0],
    )
    assert nearest["properties"]["catalog_id"] == "ME_PK209"
    assert "Sinistral" in nearest["properties"]["slip_type"]


def test_published_values_are_rounded_uniformly(committed: pd.DataFrame) -> None:
    # F1.6-f: anomalies must not be identifiable by unrounded values.
    for column, decimals in {"avg_annual_rainfall_mm": 0, "summer_max_temp_c": 1, "river_proximity_km": 2}.items():
        assert np.allclose(committed[column], committed[column].round(decimals))


def test_every_class_is_present_for_every_target(committed: pd.DataFrame) -> None:
    for target in gen.TARGETS:
        assert set(committed[target].unique()) == {0, 1, 2}
