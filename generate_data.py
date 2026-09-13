"""Dataset generator for the Pakistan Multi-Hazard Risk Analyzer (v2).

Builds ``data/pakistan_districts.csv`` from two kinds of inputs, and says
which is which in ``data/DATA_DICTIONARY.md``:

* REAL, from ``data/reference/`` (see ``data/reference/SOURCES.md``):
  district names, provinces, coordinates, point elevation, distance to the
  nearest active fault, distance to the Makran subduction zone, distance to
  the coast, and distance to cities with 1M+ people.
* SYNTHETIC, simulated here with physically motivated structure: rainfall,
  summer temperature, NDVI, river proximity, population density,
  infrastructure quality, simulated event counts, soil type, and all three
  risk labels.

Changes from v1, each tied to an AUDIT.md finding:

* F1.6-a  Coordinates were Gaussian draws in a rectangle (Rawalpindi at 29.6°N,
          ~6 points outside Pakistan) -> real GeoNames district points.
* F1.6-b  44 invented "X Rural-N" districts -> 150 real GeoNames districts only.
* F1.6-c  Impossible values (Sialkot at 7,673 m, Faisalabad at 3 people/km²)
          -> real DEM elevation, bounded synthesis, and ``validate_physical_bounds``.
* F1.6-d  Seismic label driven 40% by elevation, Balochistan excluded
          -> distance to real GEM active-fault traces (including trace ME_PK209,
          identified as the Chaman fault's Pakistani segment in SOURCES.md) and
          to the Makran subduction zone.
* F1.6-e  ``historical_disasters`` was an exact sum of two other columns and
          outlier injection broke that identity -> column removed; anomalies
          applied before any dependent feature is derived.
* F1.6-f  Injected outliers were left unrounded (revealing which rows were
          injected) -> one rounding pass after all synthesis.
* F1.6-g  ``overall_risk_score`` (computed from the labels) shipped in the CSV
          -> removed.
* F1.6-h  Labels normalised with dataset min-max, so one outlier compressed
          every other district -> fixed physical reference ranges.
* F1.6-i  ``_temp_proxy`` described as "used for heatwave label" but unused -> removed.
* F1.6-j  Columns named ``historical_*`` implied real records -> renamed
          ``simulated_*`` (there is no historical dataset in this repository).

Usage:
    python generate_data.py            # write data/pakistan_districts.csv
    python generate_data.py --check    # regenerate in memory; exit 1 if the committed CSV differs
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger("generate_data")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
REFERENCE_DIR: Final[Path] = PROJECT_ROOT / "data" / "reference"
OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "data" / "pakistan_districts.csv"
DICTIONARY_PATH: Final[Path] = PROJECT_ROOT / "data" / "DATA_DICTIONARY.md"

SEED: Final[int] = 42
EARTH_RADIUS_KM: Final[float] = 6371.0088
SOIL_TYPES: Final[tuple[str, ...]] = ("alluvial", "clay", "rocky", "sandy")
TARGETS: Final[tuple[str, ...]] = ("flood_risk", "heatwave_risk", "seismic_risk")

#: (lower, upper) physical bounds every published value must satisfy.
PHYSICAL_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "latitude": (23.5, 37.2),  # Pakistan's land extent
    "longitude": (60.8, 77.9),
    "elevation_m": (-10.0, 8_611.0),  # coastal DEM noise .. K2
    "coast_distance_km": (0.0, 2_000.0),
    "active_fault_distance_km": (0.0, 1_000.0),
    "makran_subduction_distance_km": (0.0, 2_500.0),
    "avg_annual_rainfall_mm": (30.0, 2_500.0),
    "summer_max_temp_c": (5.0, 49.0),
    "vegetation_index_ndvi": (0.02, 0.85),
    "river_proximity_km": (0.2, 150.0),
    "population_density": (1.0, 60_000.0),
    "infrastructure_quality_score": (1.0, 10.0),
    "simulated_flood_events": (0.0, 40.0),
    "simulated_earthquake_events": (0.0, 40.0),
}

#: Column -> (provenance, description). Written to DATA_DICTIONARY.md.
COLUMN_DOCS: Final[dict[str, tuple[str, str]]] = {
    "district_name": ("REAL", "GeoNames ADM2 name, suffixes such as 'District' removed."),
    "geonames_id": ("REAL", "GeoNames identifier, for traceability. Not a model feature."),
    "province": ("REAL", "Province/territory from GeoNames admin1 code."),
    "latitude": ("REAL", "GeoNames representative point (°N)."),
    "longitude": ("REAL", "GeoNames representative point (°E)."),
    "elevation_m": ("REAL", "GeoNames DEM (SRTM3/GTOPO30) at the representative point, not the district mean."),
    "coast_distance_km": ("REAL (derived)", "Distance from the district point to the Natural Earth 1:50m coastline."),
    "active_fault_distance_km": (
        "REAL (derived)",
        "Distance to the nearest GEM active crustal fault trace (folds excluded; includes the Chaman fault zone, see SOURCES.md).",
    ),
    "makran_subduction_distance_km": (
        "REAL (derived)",
        "Distance to the Makran subduction zone (Bird 2003 segments in GEM).",
    ),
    "avg_annual_rainfall_mm": (
        "SYNTHETIC",
        "Monsoon + westerly + orographic structure with noise; not observed climatology.",
    ),
    "summer_max_temp_c": (
        "SYNTHETIC",
        "June mean daily maximum from a lapse-rate, latitude and coastal model; not observed.",
    ),
    "vegetation_index_ndvi": ("SYNTHETIC", "Saturating function of synthetic rainfall, reduced above 3,500 m."),
    "river_proximity_km": ("SYNTHETIC", "Province-conditioned gamma draw; no river network data is used."),
    "population_density": ("SYNTHETIC", "Province-level log-normal with a boost near real 1M+ cities."),
    "infrastructure_quality_score": ("SYNTHETIC", "1-10 score correlated with population density."),
    "simulated_flood_events": (
        "SYNTHETIC",
        "Poisson count driven by rainfall and river proximity. NOT a historical record.",
    ),
    "simulated_earthquake_events": (
        "SYNTHETIC",
        "Poisson count driven by fault and subduction proximity. NOT a historical record.",
    ),
    "soil_type": ("SYNTHETIC", "Uniform random category; an intentional negative control with no effect on any label."),
    "flood_risk": ("SYNTHETIC LABEL", "0/1/2 = Low/Medium/High from a weighted score (see _derive_risk_labels)."),
    "heatwave_risk": ("SYNTHETIC LABEL", "0/1/2 = Low/Medium/High from a weighted score."),
    "seismic_risk": ("SYNTHETIC LABEL", "0/1/2 = Low/Medium/High from a weighted score."),
}
OUTPUT_COLUMNS: Final[tuple[str, ...]] = tuple(COLUMN_DOCS)


# --------------------------------------------------------------------------
# Geometry (real reference layers)
# --------------------------------------------------------------------------


def load_polylines(path: Path, role: str | None = None) -> list[np.ndarray]:
    """Load LineString/MultiLineString vertices from a GeoJSON file.

    Args:
        path: GeoJSON FeatureCollection.
        role: If given, keep only features whose ``properties.role`` equals it.

    Returns:
        One ``(n, 2)`` array of ``(lon, lat)`` vertices per line.
    """
    collection = json.loads(path.read_text(encoding="utf-8"))
    lines: list[np.ndarray] = []
    for feature in collection["features"]:
        if role is not None and feature["properties"].get("role") != role:
            continue
        geometry = feature["geometry"]
        parts = [geometry["coordinates"]] if geometry["type"] == "LineString" else geometry["coordinates"]
        lines.extend(np.asarray(part, dtype=float)[:, :2] for part in parts if len(part) >= 2)
    if not lines:
        raise ValueError(f"No line geometries found in {path} (role={role}).")
    return lines


def distance_to_polylines_km(lat: np.ndarray, lon: np.ndarray, lines: Sequence[np.ndarray]) -> np.ndarray:
    """Shortest distance from each point to any segment of ``lines``.

    Each point uses its own local equirectangular projection, which is within
    about 1% of the great-circle distance at the few-hundred-km scales that
    matter here. Beyond that the distance-decay terms are effectively zero.

    Args:
        lat: Point latitudes in degrees, shape ``(n,)``.
        lon: Point longitudes in degrees, shape ``(n,)``.
        lines: Polylines as ``(m, 2)`` arrays of ``(lon, lat)``.

    Returns:
        Distances in km, shape ``(n,)``.
    """
    starts = np.concatenate([line[:-1] for line in lines])
    ends = np.concatenate([line[1:] for line in lines])
    km_per_degree = EARTH_RADIUS_KM * np.pi / 180.0
    distances = np.empty(len(lat))
    for i, (plat, plon) in enumerate(zip(lat, lon, strict=True)):
        scale = np.array([np.cos(np.radians(plat)) * km_per_degree, km_per_degree])
        a = (starts - [plon, plat]) * scale
        d = (ends - starts) * scale
        length_sq = np.einsum("ij,ij->i", d, d)
        t = np.clip(-np.einsum("ij,ij->i", a, d) / np.where(length_sq > 0, length_sq, 1.0), 0.0, 1.0)
        closest = a + t[:, None] * d
        distances[i] = np.sqrt(np.einsum("ij,ij->i", closest, closest).min())
    return distances


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance in km from points to one location."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(h))


def load_real_layers(reference_dir: Path = REFERENCE_DIR) -> pd.DataFrame:
    """Join real district points with distances derived from real geometry.

    Returns:
        One row per district with the REAL columns of the data dictionary,
        plus ``nearest_city_km`` (used internally by the synthesis step).
    """
    districts = pd.read_csv(reference_dir / "districts.csv")
    cities = pd.read_csv(reference_dir / "major_cities.csv")
    lat = districts["latitude"].to_numpy(float)
    lon = districts["longitude"].to_numpy(float)

    districts["coast_distance_km"] = distance_to_polylines_km(
        lat, lon, load_polylines(reference_dir / "coastline.geojson")
    )
    districts["active_fault_distance_km"] = distance_to_polylines_km(
        lat, lon, load_polylines(reference_dir / "active_faults.geojson", "crustal")
    )
    districts["makran_subduction_distance_km"] = distance_to_polylines_km(
        lat, lon, load_polylines(reference_dir / "active_faults.geojson", "makran_subduction")
    )
    districts["nearest_city_km"] = np.min(
        [haversine_km(lat, lon, c.latitude, c.longitude) for c in cities.itertuples()], axis=0
    )
    return districts


# --------------------------------------------------------------------------
# Synthetic features
# --------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function."""
    return 1.0 / (1.0 + np.exp(-x))


def _unit(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Scale ``x`` to [0, 1] against FIXED physical references, clipping outside."""
    return np.clip((x - lower) / (upper - lower), 0.0, 1.0)


def _synthesise_climate(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add synthetic rainfall and summer temperature consistent with real geography."""
    df = df.copy()
    lat, lon = df["latitude"].to_numpy(), df["longitude"].to_numpy()
    elev = df["elevation_m"].to_numpy(float).clip(min=0)
    coast = df["coast_distance_km"].to_numpy()

    # Shape constants are judgment calls chosen to reproduce Pakistan's broad
    # pattern (wet north-east, arid south-west, dry high Karakoram); they are
    # not fitted to observations.
    monsoon = _sigmoid((lon - 72.0) / 0.7) * np.exp(
        -(((lat - 33.8) / 2.4) ** 2)
    )  # SW monsoon peaks over the Potohar/Kashmir foothills
    south_monsoon = _sigmoid((lon - 68.0) / 1.0) * np.exp(
        -(((lat - 25.0) / 2.0) ** 2)
    )  # weak monsoon reach into lower Sindh
    westerlies = np.exp(-(((lat - 31.0) / 3.0) ** 2)) * _sigmoid(
        (72.0 - lon) / 1.5
    )  # winter western disturbances over Balochistan/KP
    north_shadow = np.exp(-np.clip(lat - 35.0, 0, None) * 1.5)  # Karakoram rain shadow north of ~35°N
    orographic = _unit(elev, 0, 2_000) * (
        1 - _unit(elev, 3_000, 5_000)
    )  # uplift enhancement that fades at glacial altitudes
    rainfall = (
        60.0
        + 1_100.0 * monsoon * north_shadow
        + 150.0 * south_monsoon
        + 120.0 * westerlies
        + 500.0 * orographic * (monsoon + 0.3 * westerlies) * north_shadow
    ) * rng.lognormal(0.0, 0.18, size=len(df))  # ±~20% district-to-district variability
    df["avg_annual_rainfall_mm"] = rainfall

    # 6.5 °C/km is the ICAO standard-atmosphere lapse rate; 46 °C intercept at
    # 27°N approximates the upper-Sindh plains; the 9 °C / 40 km coastal term
    # represents sea-breeze moderation (judgment).
    temperature = 46.0 - 6.5 * (elev / 1_000.0) - 0.45 * (lat - 27.0) - 9.0 * np.exp(-coast / 40.0)
    df["summer_max_temp_c"] = temperature + rng.normal(0.0, 1.0, size=len(df))
    return df


def _inject_climate_anomalies(df: pd.DataFrame, rng: np.random.Generator, frac: float = 0.04) -> pd.DataFrame:
    """Apply bounded extreme-season anomalies to a few districts.

    Kept so RobustScaler has genuine extremes to be robust against, but every
    anomaly stays physically possible, and it is applied BEFORE any dependent
    feature (NDVI, event counts, labels) is derived, so no identity breaks.
    """
    df = df.copy()
    n = max(2, round(len(df) * frac))
    rows = rng.choice(df.index.to_numpy(), size=n, replace=False)
    kinds = rng.choice(["extreme_monsoon", "extreme_heat"], size=n)
    for row, kind in zip(rows, kinds, strict=True):
        if kind == "extreme_monsoon":
            # 1.5-2.0x the district's synthetic normal: an exceptional but physically possible wet year (judgment range).
            df.loc[row, "avg_annual_rainfall_mm"] *= rng.uniform(1.5, 2.0)
        else:
            df.loc[row, "summer_max_temp_c"] += rng.uniform(2.0, 3.5)
    logger.info("Applied %d bounded climate anomalies (%.1f%% of districts).", n, 100 * n / len(df))
    return df


#: Province-level synthetic priors: (median population density /km², mean river distance km).
#: Order-of-magnitude judgment values, not census figures.
PROVINCE_PRIORS: Final[dict[str, tuple[float, float]]] = {
    "Punjab": (450.0, 15.0),
    "Sindh": (250.0, 15.0),
    "Khyber Pakhtunkhwa": (350.0, 12.0),
    "Balochistan": (30.0, 45.0),
    "Gilgit-Baltistan": (15.0, 8.0),
    "Azad Jammu & Kashmir": (300.0, 6.0),
    "Islamabad Capital Territory": (1_500.0, 10.0),
}


def _synthesise_land_and_exposure(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add NDVI, river proximity, population density and infrastructure quality."""
    df = df.copy()
    n = len(df)
    elev = df["elevation_m"].to_numpy(float).clip(min=0)
    median_density = df["province"].map({k: v[0] for k, v in PROVINCE_PRIORS.items()}).to_numpy(float)
    mean_river = df["province"].map({k: v[1] for k, v in PROVINCE_PRIORS.items()}).to_numpy(float)

    # Greenness saturates with rainfall (e-folding 900 mm) and falls off on bare rock/ice above 3,500 m (judgment).
    df["vegetation_index_ndvi"] = (
        0.04
        + 0.62 * (1 - np.exp(-df["avg_annual_rainfall_mm"].to_numpy() / 900.0))
        - 0.25 * _unit(elev, 3_500, 5_000)
        + rng.normal(0.0, 0.05, size=n)
    )
    df["river_proximity_km"] = rng.gamma(
        shape=2.0, scale=mean_river / 2.0
    )  # shape 2: few districts sit exactly on a river

    # Density rises up to ~21x within ~15 km of a real 1M+ city (e-folding 15 km, judgment).
    urban_boost = 1.0 + 20.0 * np.exp(-df["nearest_city_km"].to_numpy() / 15.0)
    df["population_density"] = median_density * urban_boost * rng.lognormal(0.0, 0.5, size=n)
    df["infrastructure_quality_score"] = (
        4.5 + 1.2 * np.log10(df["population_density"].clip(lower=1) / 100.0) + rng.normal(0.0, 1.2, size=n)
    )
    df["soil_type"] = rng.choice(SOIL_TYPES, size=n)
    return df


def _simulate_event_counts(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add Poisson event counts driven by the same physical drivers as the labels."""
    df = df.copy()
    flood_rate = (
        0.5
        + 6.0 * np.exp(-df["river_proximity_km"].to_numpy() / 20.0)
        + 4.0 * _unit(df["avg_annual_rainfall_mm"].to_numpy(), 0, 1_500)
    )
    quake_rate = (
        0.2
        + 4.0 * np.exp(-df["active_fault_distance_km"].to_numpy() / 30.0)
        + 2.0 * np.exp(-df["makran_subduction_distance_km"].to_numpy() / 120.0)
    )
    df["simulated_flood_events"] = rng.poisson(flood_rate)
    df["simulated_earthquake_events"] = rng.poisson(quake_rate)
    return df


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def _threshold_minority_high(score: np.ndarray, medium_q: float = 0.55, high_q: float = 0.85) -> np.ndarray:
    """Cut a continuous score into 0/1/2 at quantiles.

    0.55/0.85 keeps the v1 class balance (~55% Low, ~30% Medium, ~15% High) so
    v1 and v2 metrics are comparable. Labels remain RELATIVE ranks within
    Pakistan, not absolute hazard levels (documented limitation).
    """
    medium_cut, high_cut = np.quantile(score, [medium_q, high_q])
    return np.where(score >= high_cut, 2, np.where(score >= medium_cut, 1, 0))


def _derive_risk_labels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Derive the three risk labels from fixed-reference weighted scores plus noise."""
    df = df.copy()
    n = len(df)
    log_density = np.log10(df["population_density"].to_numpy().clip(min=1))
    vulnerability = 1 - _unit(df["infrastructure_quality_score"].to_numpy(), 1, 10)

    flood_score = (
        0.35 * _unit(df["avg_annual_rainfall_mm"].to_numpy(), 0, 2_000)
        + 0.25
        * np.exp(
            -df["river_proximity_km"].to_numpy() / 10.0
        )  # riverine inundation concentrates within ~10 km (judgment)
        + 0.15 * _unit(df["simulated_flood_events"].to_numpy(), 0, 12)
        + 0.15
        * (
            1 - _unit(df["elevation_m"].to_numpy(), 0, 1_500)
        )  # low plains flood; valleys above 1,500 m rarely inundate widely
        + 0.10 * _unit(log_density, 0, 4.5)
    )
    heatwave_score = (
        0.50 * _unit(df["summer_max_temp_c"].to_numpy(), 30, 48)
        + 0.20 * (1 - _unit(df["vegetation_index_ndvi"].to_numpy(), 0, 0.85))
        + 0.15 * _unit(log_density, 0, 4.5)  # urban heat island and exposed population
        + 0.15 * vulnerability  # access to cooling and power
    )
    seismic_score = (
        # 25 km e-folding: damage from crustal ruptures concentrates within a few tens of km (judgment);
        # 100 km for the megathrust, whose rupture area spans the whole Makran coast (judgment).
        0.40 * np.exp(-df["active_fault_distance_km"].to_numpy() / 25.0)
        + 0.15 * np.exp(-df["makran_subduction_distance_km"].to_numpy() / 100.0)
        + 0.25 * _unit(df["simulated_earthquake_events"].to_numpy(), 0, 8)
        + 0.20 * vulnerability
    )
    for target, score in zip(TARGETS, (flood_score, heatwave_score, seismic_score), strict=True):
        df[target] = _threshold_minority_high(score + rng.normal(0.0, 0.04, size=n))
    return df


# --------------------------------------------------------------------------
# Validation and orchestration
# --------------------------------------------------------------------------


def _round_for_publication(df: pd.DataFrame) -> pd.DataFrame:
    """Round every column once, after all synthesis, so no row type stands out."""
    df = df.copy()
    for column, decimals in {
        "latitude": 5,
        "longitude": 5,
        "coast_distance_km": 1,
        "active_fault_distance_km": 1,
        "makran_subduction_distance_km": 1,
        "avg_annual_rainfall_mm": 0,
        "summer_max_temp_c": 1,
        "vegetation_index_ndvi": 3,
        "river_proximity_km": 2,
        "population_density": 0,
        "infrastructure_quality_score": 1,
    }.items():
        df[column] = df[column].round(decimals)
    return df


def _clip_to_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clip SYNTHETIC columns present so far into their physical bounds (real columns are only validated)."""
    df = df.copy()
    for column, (provenance, _) in COLUMN_DOCS.items():
        if provenance == "SYNTHETIC" and column in PHYSICAL_BOUNDS and column in df.columns:
            df[column] = df[column].clip(*PHYSICAL_BOUNDS[column])
    return df


def validate_physical_bounds(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any value is outside its physical bounds.

    Args:
        df: Generated dataset.

    Raises:
        ValueError: Lists every violating column with its offending range.
    """
    problems = []
    for column, (lower, upper) in PHYSICAL_BOUNDS.items():
        values = df[column]
        if values.isna().any() or (values < lower).any() or (values > upper).any():
            problems.append(f"{column}: observed [{values.min()}, {values.max()}], allowed [{lower}, {upper}]")
    if df["district_name"].duplicated().any():
        problems.append("district_name: duplicates present")
    if problems:
        raise ValueError("Physical bounds violated:\n  " + "\n  ".join(problems))


def generate_dataset(seed: int = SEED, reference_dir: Path = REFERENCE_DIR) -> pd.DataFrame:
    """Generate the full v2 dataset.

    Args:
        seed: RNG seed; the same seed and reference files always give the same CSV.
        reference_dir: Directory holding the real reference layers.

    Returns:
        DataFrame with ``OUTPUT_COLUMNS`` in order, one row per real district.
    """
    rng = np.random.default_rng(seed)
    df = load_real_layers(reference_dir)
    df = _synthesise_climate(df, rng)
    df = _inject_climate_anomalies(df, rng)
    df = _clip_to_bounds(df)
    df = _synthesise_land_and_exposure(df, rng)
    df = _clip_to_bounds(df)
    df = _simulate_event_counts(df, rng)
    df = _derive_risk_labels(df, rng)
    df = _round_for_publication(df)[list(OUTPUT_COLUMNS)]
    validate_physical_bounds(df)
    for target in TARGETS:
        logger.info(
            "%s class counts (0=Low,1=Med,2=High): %s", target, df[target].value_counts().sort_index().to_dict()
        )
    return df


def write_data_dictionary(path: Path = DICTIONARY_PATH) -> None:
    """Write the column provenance table (REAL vs SYNTHETIC)."""
    rows = [f"| `{c}` | {p} | {d} |" for c, (p, d) in COLUMN_DOCS.items()]
    path.write_text(
        "# Data dictionary: `pakistan_districts.csv`\n\n"
        "Generated by `generate_data.py`. REAL columns come from `data/reference/` (see `SOURCES.md`).\n"
        "Everything marked SYNTHETIC, including all labels, is simulated. There is no historical event dataset in this repository.\n\n"
        "| Column | Provenance | Definition |\n|---|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Generate the district dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true", help="Exit 1 if --out differs from a fresh generation.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    df = generate_dataset(seed=args.seed)
    if args.check:
        committed = pd.read_csv(args.out)
        fresh = pd.read_csv(io.StringIO(df.to_csv(index=False)))
        if not committed.equals(fresh):
            logger.error("%s does not match a fresh generation with seed %d.", args.out, args.seed)
            return 1
        logger.info("%s is reproducible from seed %d.", args.out, args.seed)
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    write_data_dictionary()
    logger.info("Saved %d rows x %d columns -> %s", *df.shape, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
