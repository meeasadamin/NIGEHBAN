"""Build the real (non-synthetic) reference layers used by generate_data.py.

Downloads three public datasets, extracts the Pakistan-relevant subset, and
writes small committed files under ``data/reference/``:

* ``districts.csv``        GeoNames ADM2 district points + DEM elevation (CC BY 4.0)
* ``major_cities.csv``     GeoNames populated places with population >= 1,000,000 (CC BY 4.0)
* ``active_faults.geojson`` GEM Global Active Faults Database traces (CC BY-SA 4.0)
* ``coastline.geojson``    Natural Earth 1:50m coastline (public domain)
* ``SOURCES.md``           Attribution, licences, retrieval date, raw SHA-256, rules, exclusions

Resolves audit finding 1.6 (district coordinates were random draws inside a
rectangle; seismic risk used elevation instead of fault proximity).

The outputs are committed so the rest of the pipeline and CI never need the
network. Re-run this script only to refresh the reference layers:

    python scripts/build_reference_data.py            # downloads into a temp dir
    python scripts/build_reference_data.py --cache DIR # reuse previously downloaded files
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import tempfile
import unicodedata
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

logger = logging.getLogger("build_reference_data")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "reference"


@dataclass(frozen=True, slots=True)
class Source:
    """A pinned public download.

    Attributes:
        key: Local file name.
        url: Download URL.
        licence: Licence of the upstream data.
        citation: Attribution text required by the licence.
    """

    key: str
    url: str
    licence: str
    citation: str


SOURCES: Final[tuple[Source, ...]] = (
    Source(
        "PK.zip",
        "https://download.geonames.org/export/dump/PK.zip",
        "CC BY 4.0",
        "GeoNames geographical database, https://www.geonames.org/",
    ),
    Source(
        "gem_active_faults_harmonized.geojson",
        "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/geojson/gem_active_faults_harmonized.geojson",
        "CC BY-SA 4.0",
        "Styron, R., and Pagani, M. (2020). The GEM Global Active Faults Database. Earthquake Spectra 36(1_suppl), 160-180. doi:10.1177/8755293020944182",
    ),
    Source(
        "ne_50m_coastline.geojson",
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_coastline.geojson",
        "Public domain",
        "Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com",
    ),
)

GEONAMES_COLUMNS: Final[list[str]] = [
    "geonameid",
    "name",
    "asciiname",
    "alternatenames",
    "latitude",
    "longitude",
    "feature_class",
    "feature_code",
    "country_code",
    "cc2",
    "admin1",
    "admin2",
    "admin3",
    "admin4",
    "population",
    "elevation",
    "dem",
    "timezone",
    "modified",
]

#: GeoNames admin1 codes for Pakistan -> province names used throughout the project.
ADMIN1_TO_PROVINCE: Final[dict[str, str]] = {
    "02": "Balochistan",
    "03": "Khyber Pakhtunkhwa",
    "04": "Punjab",
    "05": "Sindh",
    "06": "Azad Jammu & Kashmir",
    "07": "Gilgit-Baltistan",
    "08": "Islamabad Capital Territory",
}

#: Region used to clip fault and coastline layers: Pakistan's extent padded so
#: offshore Makran structures and cross-border faults near districts are kept.
REGION_LON: Final[tuple[float, float]] = (57.0, 80.0)
REGION_LAT: Final[tuple[float, float]] = (20.0, 38.5)

#: Pakistan's land extent. Every district point must fall inside it.
PAKISTAN_LAT: Final[tuple[float, float]] = (23.5, 37.2)
PAKISTAN_LON: Final[tuple[float, float]] = (60.8, 77.9)

#: GEM slip types that describe folds, not faults that rupture.
FOLD_TYPES: Final[frozenset[str]] = frozenset({"Anticline", "Syncline"})

#: Exclusion rules for GeoNames ADM2 records, each with the reason written to SOURCES.md.
SUPERSEDED_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    (
        r"Tribal Area|Frontier Region",
        "Frontier Regions were merged into adjacent districts by the 25th Amendment (2018).",
    ),
    (r"^Hunza-Nagar", "Superseded by the 2015 split into Hunza and Nagar; Nagar is present as its own record."),
)


def _sha256(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fetch(source: Source, cache: Path) -> bytes:
    """Return the bytes of ``source``, downloading into ``cache`` if needed."""
    target = cache / source.key
    if not target.is_file():
        logger.info("Downloading %s", source.url)
        with urllib.request.urlopen(source.url, timeout=300) as response:  # noqa: S310 - pinned https URLs
            target.write_bytes(response.read())
    return target.read_bytes()


def _clean_name(ascii_name: str) -> str:
    """Normalise a GeoNames ADM2 name to a display district name."""
    name = unicodedata.normalize("NFKC", ascii_name).strip()
    name = re.sub(r"^District of (\w+) (\w+)$", r"\1 \2", name)
    name = re.sub(r"\s+(District|Agency|Protected Area)$", "", name)
    return name


def read_geonames(pk_zip: bytes) -> pd.DataFrame:
    """Parse the GeoNames ``PK.txt`` table from the downloaded archive."""
    with zipfile.ZipFile(io.BytesIO(pk_zip)) as archive:
        return pd.read_csv(
            archive.open("PK.txt"),
            sep="\t",
            names=GEONAMES_COLUMNS,
            quoting=3,
            dtype={"admin1": "string"},
            keep_default_na=False,
        )


def chaman_town(raw: pd.DataFrame) -> tuple[float, float]:
    """Return the GeoNames coordinates of Chaman town (most populous populated place of that name)."""
    towns = raw[(raw["asciiname"] == "Chaman") & (raw["feature_class"] == "P")].sort_values(
        "population", ascending=False
    )
    if towns.empty:
        raise ValueError("Chaman town not found in GeoNames.")
    return float(towns.iloc[0]["latitude"]), float(towns.iloc[0]["longitude"])


def build_districts(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract district points, major cities, and the exclusion log from GeoNames.

    Args:
        raw: Parsed GeoNames ``PK.txt`` table.

    Returns:
        ``(districts, major_cities, excluded)`` DataFrames.
    """

    adm2 = raw[raw["feature_code"] == "ADM2"].copy()
    adm2["district_name"] = adm2["asciiname"].map(_clean_name)
    adm2["province"] = adm2["admin1"].map(ADMIN1_TO_PROVINCE)
    excluded: list[dict[str, Any]] = []

    for pattern, reason in SUPERSEDED_PATTERNS:
        mask = adm2["asciiname"].str.contains(pattern, regex=True)
        excluded += [{"geonameid": r.geonameid, "name": r.asciiname, "reason": reason} for r in adm2[mask].itertuples()]
        adm2 = adm2[~mask]

    # Duplicate records for one district (e.g. "Gilgit" and "Gilgit District"): keep the most recently maintained.
    adm2 = adm2.sort_values("modified", ascending=False)
    dupes = adm2.duplicated(subset=["district_name", "province"], keep="first")
    excluded += [
        {
            "geonameid": r.geonameid,
            "name": r.asciiname,
            "reason": "Duplicate of a more recently modified GeoNames record for the same district.",
        }
        for r in adm2[dupes].itertuples()
    ]
    adm2 = adm2[~dupes]

    # Islamabad Capital Territory has no ADM2 record; its ADM1 point stands in for the single district.
    ict = raw[(raw["feature_code"] == "ADM1") & (raw["admin1"] == "08")].copy()
    ict["district_name"] = "Islamabad"
    ict["province"] = ADMIN1_TO_PROVINCE["08"]
    districts = pd.concat([adm2, ict], ignore_index=True)

    districts = districts.rename(
        columns={"geonameid": "geonames_id", "dem": "elevation_m", "modified": "geonames_modified"}
    )
    districts = districts[
        ["geonames_id", "district_name", "province", "latitude", "longitude", "elevation_m", "geonames_modified"]
    ]
    districts = districts.sort_values(["province", "district_name"], ignore_index=True)

    if districts["province"].isna().any():
        raise ValueError("A GeoNames district has an unmapped admin1 code.")
    if districts["district_name"].duplicated().any():
        raise ValueError(
            f"Duplicate district names: {districts.loc[districts['district_name'].duplicated(), 'district_name'].tolist()}"
        )
    inside = districts["latitude"].between(*PAKISTAN_LAT) & districts["longitude"].between(*PAKISTAN_LON)
    if not inside.all():
        raise ValueError(
            f"District points outside Pakistan's extent: {districts.loc[~inside, 'district_name'].tolist()}"
        )
    if (districts["elevation_m"] < -100).any():
        raise ValueError("GeoNames DEM value missing for a district.")

    cities = raw[(raw["feature_class"] == "P") & (raw["population"] >= 1_000_000)]
    cities = cities.rename(columns={"asciiname": "city", "geonameid": "geonames_id"})
    cities = cities[["geonames_id", "city", "latitude", "longitude", "population"]].sort_values(
        "population", ascending=False, ignore_index=True
    )
    return districts, cities, pd.DataFrame(excluded)


def _lines(geometry: dict[str, Any]) -> list[list[list[float]]]:
    """Return a geometry's coordinates as a list of LineStrings."""
    if geometry["type"] == "LineString":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiLineString":
        return list(geometry["coordinates"])
    return []


def _in_region(lines: list[list[list[float]]]) -> bool:
    """Whether any vertex lies in the clipping region."""
    return any(
        REGION_LON[0] <= p[0] <= REGION_LON[1] and REGION_LAT[0] <= p[1] <= REGION_LAT[1]
        for line in lines
        for p in line
    )


def fault_role(properties: dict[str, Any], lines: list[list[list[float]]]) -> str | None:
    """Classify a GEM trace as ``makran_subduction``, ``crustal``, or ``None`` (fold).

    The Makran megathrust is represented by the Bird (2003) plate-boundary
    segments typed ``Subduction_Thrust`` that lie off the Makran coast.
    """
    slip = properties.get("slip_type") or ""
    if slip in FOLD_TYPES:
        return None
    if slip == "Subduction_Thrust":
        lons = [p[0] for line in lines for p in line]
        lats = [p[1] for line in lines for p in line]
        return (
            "makran_subduction"
            if min(lons) >= 57.0 and max(lons) <= 67.0 and min(lats) >= 22.5 and max(lats) <= 26.0
            else None
        )
    return "crustal"


@dataclass(frozen=True, slots=True)
class ChamanIdentification:
    """Which GEM trace represents the Chaman fault inside Pakistan, and why.

    GEM's only trace *named* "Chaman Fault" (HimaTibetMap) lies in Afghanistan
    (~33.6-34.0°N), hundreds of km from Chaman town. The Pakistani segment is in
    the EMME catalogue without a name, so it is identified by geometry and
    kinematics: the crustal trace nearest the GeoNames point for Chaman town,
    required to be left-lateral (sinistral) and within 25 km (judgment tolerance).

    Attributes:
        catalog_id: GEM catalogue id of the identified trace.
        distance_km: Distance from Chaman town to that trace.
        slip_type: GEM slip type of that trace.
        net_slip_rate: GEM slip-rate triple (preferred, min, max) in mm/yr.
        named_trace_distance_km: Distance from Chaman town to GEM's trace named "Chaman Fault".
    """

    catalog_id: str
    distance_km: float
    slip_type: str
    net_slip_rate: str | None
    named_trace_distance_km: float


def _point_to_lines_km(lat: float, lon: float, lines: list[list[list[float]]]) -> float:
    """Local-equirectangular point-to-polyline distance (same method as generate_data.py)."""
    import math

    km_lat = 6371.0088 * math.pi / 180
    km_lon = km_lat * math.cos(math.radians(lat))
    best = math.inf
    for line in lines:
        for (x1, y1), (x2, y2) in zip((p[:2] for p in line[:-1]), (p[:2] for p in line[1:]), strict=True):
            ax, ay = (x1 - lon) * km_lon, (y1 - lat) * km_lat
            dx, dy = (x2 - x1) * km_lon, (y2 - y1) * km_lat
            t = max(0.0, min(1.0, -(ax * dx + ay * dy) / (dx * dx + dy * dy))) if dx or dy else 0.0
            best = min(best, math.hypot(ax + t * dx, ay + t * dy))
    return best


def identify_chaman_trace(faults: dict[str, Any], chaman_lat: float, chaman_lon: float) -> ChamanIdentification:
    """Identify the Pakistani Chaman fault segment in the GEM subset (see ``ChamanIdentification``)."""
    crustal = [f for f in faults["features"] if f["properties"]["role"] == "crustal"]
    distances = [(_point_to_lines_km(chaman_lat, chaman_lon, f["geometry"]["coordinates"]), f) for f in crustal]
    distance, nearest = min(distances, key=lambda pair: pair[0])
    props = nearest["properties"]
    if "Sinistral" not in (props["slip_type"] or "") or distance > 25.0:
        raise ValueError(
            f"Nearest trace to Chaman town ({props['catalog_id']}, {props['slip_type']}, {distance:.1f} km) is not a plausible Chaman segment."
        )
    named = [d for d, f in distances if f["properties"].get("fs_name") == "Chaman Fault"]
    return ChamanIdentification(
        catalog_id=str(props["catalog_id"]),
        distance_km=round(distance, 1),
        slip_type=str(props["slip_type"]),
        net_slip_rate=props.get("net_slip_rate"),
        named_trace_distance_km=round(min(named), 1) if named else float("nan"),
    )


def build_faults(gem_geojson: bytes) -> dict[str, Any]:
    """Extract active fault traces around Pakistan from the GEM database."""
    collection = json.loads(gem_geojson)
    features = []
    for feature in collection["features"]:
        geometry = feature.get("geometry")
        if not geometry:
            continue
        lines = _lines(geometry)
        if not lines or not _in_region(lines):
            continue
        props = feature["properties"]
        role = fault_role(props, lines)
        if role is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiLineString", "coordinates": [[p[:2] for p in line] for line in lines]},
                "properties": {
                    "role": role,
                    "catalog_id": props.get("catalog_id"),
                    "catalog_name": props.get("catalog_name"),
                    "fs_name": props.get("fs_name"),
                    "slip_type": props.get("slip_type"),
                    "net_slip_rate": props.get("net_slip_rate"),
                },
            }
        )
    roles = {f["properties"]["role"] for f in features}
    if {"crustal", "makran_subduction"} - roles:
        raise ValueError(f"Fault subset is missing required roles; found {roles}.")
    return {"type": "FeatureCollection", "features": features}


def _clip_to_region(line: list[list[float]]) -> list[list[list[float]]]:
    """Split a line into runs of consecutive vertices inside the clipping region.

    Natural Earth stores whole continental coastlines as single lines; only the
    runs near Pakistan matter for distance-to-coast.
    """
    runs: list[list[list[float]]] = []
    current: list[list[float]] = []
    for point in line:
        if REGION_LON[0] <= point[0] <= REGION_LON[1] and REGION_LAT[0] <= point[1] <= REGION_LAT[1]:
            current.append(point[:2])
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return [run for run in runs if len(run) >= 2]


def build_coastline(ne_geojson: bytes) -> dict[str, Any]:
    """Extract Natural Earth coastline runs in the region."""
    collection = json.loads(ne_geojson)
    runs = [
        run
        for f in collection["features"]
        if f.get("geometry")
        for line in _lines(f["geometry"])
        for run in _clip_to_region(line)
    ]
    if not runs:
        raise ValueError("No coastline found in the region.")
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": runs}, "properties": {}}
        ],
    }


def write_sources(
    digests: dict[str, str],
    districts: pd.DataFrame,
    cities: pd.DataFrame,
    faults: dict[str, Any],
    excluded: pd.DataFrame,
    chaman: ChamanIdentification,
) -> str:
    """Render SOURCES.md with attribution, provenance and every exclusion."""
    roles = pd.Series([f["properties"]["role"] for f in faults["features"]]).value_counts().to_dict()
    lines = [
        "# Reference data sources",
        "",
        f"Generated by `scripts/build_reference_data.py` on {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. Do not edit by hand.",
        "",
        "These layers are **real** data. Every other feature in `data/pakistan_districts.csv` is synthetic (see `generate_data.py`).",
        "",
        "| File | Upstream | Licence | Raw download SHA-256 |",
        "|---|---|---|---|",
    ]
    for source in SOURCES:
        lines.append(f"| derived from `{source.key}` | {source.url} | {source.licence} | `{digests[source.key]}` |")
    lines += ["", "## Attribution", ""] + [f"- {s.citation} ({s.licence})." for s in SOURCES]
    lines += [
        "",
        "`active_faults.geojson` is a derivative of the GEM Global Active Faults Database and is distributed under CC BY-SA 4.0.",
        "",
        "## Contents",
        "",
        f"- `districts.csv`: {len(districts)} district points ({districts['province'].value_counts().to_dict()}).",
        "  Coordinates are GeoNames representative points for each ADM2 unit; `elevation_m` is the GeoNames DEM value (SRTM3/GTOPO30) **at that point**, not the district mean.",
        f"- `major_cities.csv`: {len(cities)} populated places with population >= 1,000,000.",
        f"- `active_faults.geojson`: {len(faults['features'])} traces by role {roles}. Folds (Anticline/Syncline) are excluded. "
        "`makran_subduction` = Bird (2003) `Subduction_Thrust` segments within 57-67°E, 22.5-26°N (the database gives no name; identified by type and location).",
        "- `coastline.geojson`: Natural Earth 1:50m coastline clipped to 57-80°E, 20-38.5°N.",
        "",
        "## Chaman fault identification (inference, not a database label)",
        "",
        f'- GEM\'s only trace *named* "Chaman Fault" (HimaTibetMap) is the northern segment in Afghanistan, {chaman.named_trace_distance_km} km from Chaman town.',
        f"- The Pakistani segment is taken to be EMME trace `{chaman.catalog_id}`: the active crustal trace nearest the GeoNames point for Chaman town "
        f"({chaman.distance_km} km), slip type `{chaman.slip_type}`, net slip rate `{chaman.net_slip_rate}` mm/yr. "
        "The builder fails if the nearest trace is not sinistral or is more than 25 km away.",
        "- `active_fault_distance_km` uses every crustal trace, so this identification documents coverage; it does not change any computed value.",
        "",
        "## Excluded GeoNames records",
        "",
        "| GeoNames id | Name | Reason |",
        "|---|---|---|",
    ]
    lines += [f"| {r.geonameid} | {r.name} | {r.reason} |" for r in excluded.itertuples()]
    lines += [
        "",
        "## Known gaps (need an official input to close)",
        "",
        "- GeoNames ADM2 is not the official district list: e.g. Keamari (Karachi) and Hunza are absent, and several 2022-23 Balochistan districts may be missing. "
        "Replace `districts.csv` with the PBS 2023 census district list joined to OCHA COD-AB admin-2 boundaries (HDX: `cod-ab-pak`) to fix this.",
        "- District points are not polygons: distances to faults and coast are measured from one point per district.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Download sources, build the reference layers, and write them to ``data/reference``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, help="Directory holding (or receiving) raw downloads.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    with tempfile.TemporaryDirectory() as tmp:
        cache = args.cache or Path(tmp)
        cache.mkdir(parents=True, exist_ok=True)
        raw = {source.key: fetch(source, cache) for source in SOURCES}

    digests = {key: _sha256(data) for key, data in raw.items()}
    geonames = read_geonames(raw["PK.zip"])
    districts, cities, excluded = build_districts(geonames)
    faults = build_faults(raw["gem_active_faults_harmonized.geojson"])
    chaman = identify_chaman_trace(faults, *chaman_town(geonames))
    coastline = build_coastline(raw["ne_50m_coastline.geojson"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    districts.to_csv(OUTPUT_DIR / "districts.csv", index=False)
    cities.to_csv(OUTPUT_DIR / "major_cities.csv", index=False)
    (OUTPUT_DIR / "active_faults.geojson").write_text(json.dumps(faults, separators=(",", ":")), encoding="utf-8")
    (OUTPUT_DIR / "coastline.geojson").write_text(json.dumps(coastline, separators=(",", ":")), encoding="utf-8")
    (OUTPUT_DIR / "SOURCES.md").write_text(
        write_sources(digests, districts, cities, faults, excluded, chaman), encoding="utf-8"
    )
    logger.info(
        "Wrote %d districts, %d cities, %d fault traces, %d coastline runs -> %s",
        len(districts),
        len(cities),
        len(faults["features"]),
        len(coastline["features"][0]["geometry"]["coordinates"]),
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
