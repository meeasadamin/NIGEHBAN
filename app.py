"""Pakistan Multi-Hazard Risk Analyzer — Streamlit dashboard (UI layer only).

All data, model, SHAP, and export logic lives in ``risk_engine.py``; colours, CSS and
HTML fragments live in ``ui_theme.py``. This file only lays out what those modules return.

Traceability (see AUDIT.md):
* F3.1  SHAP waterfall drawn with Plotly; no matplotlib.pyplot calls anywhere.
* F3.2  Cached loaders raise instead of returning None, key on file mtime, and
        return immutable objects.
* F4.1  Sliders are bounded to the training range; novel combinations are flagged.
* F4.2  One top-level error boundary; no tracebacks reach the browser.
* F5    Colour-blind-safe palette with shapes, verified contrast, persistent
        simulated-scenario banner, per-hazard explanations, text alternatives.
* New feature requests (no audit finding): glass KPI cards, CSV/TXT export, and the
  professional layout pass (header, scenario strip, hazard selector scoping both the
  explanation and a national map, district profile, provenance checklist).

Run:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import risk_engine as engine
import ui_theme as theme

logger = logging.getLogger("ndma_dashboard")

DEFAULT_DISTRICT = "Islamabad"  # neutral default: the capital territory
# Browser tab icon: the Nigehban emblem, rendered to PNG by scripts/build_favicon.py.
FAVICON_PATH = Path(__file__).resolve().parent / "static" / "nigehban-favicon.png"
MAX_WATERFALL_FEATURES = 8  # keeps the chart readable on a phone; the rest are grouped
#: Sidebar grouping of the tunable inputs (order matches engine.TUNABLE_FEATURES).
SLIDER_GROUPS: dict[str, tuple[str, ...]] = {
    "Climate": ("avg_annual_rainfall_mm", "summer_max_temp_c"),
    "Land and exposure": (
        "river_proximity_km",
        "vegetation_index_ndvi",
        "population_density",
        "infrastructure_quality_score",
    ),
}
#: Sidebar card hint and icon per slider group.
SLIDER_GROUP_HEADS: dict[str, tuple[str, str]] = {
    "Climate": ("Rainfall and summer heat", "climate"),
    "Land and exposure": ("Terrain, people, infrastructure", "layers"),
}
LIMITATIONS: tuple[tuple[str, str], ...] = (
    (
        "Synthetic data",
        "Climate, exposure, event counts and all labels are synthetic. Real inputs: district names, coordinates, "
        "point elevation, active-fault and Makran subduction geometry, and coastline (data/reference/SOURCES.md).",
    ),
    ("Relative labels", "Labels are relative ranks within Pakistan (top ~15% = High), not absolute hazard levels."),
    (
        "No historical baseline",
        "There is no historical event dataset in this project, so changes are shown against the district's "
        "unmodified dataset inputs only.",
    ),
    ("Small sample", "About 150 rows: every metric has wide uncertainty; see the ± values."),
    ("Not operational", "Not for operational disaster response."),
)
MAP_CENTER = {"lat": 30.4, "lon": 70.0}  # centre of the district points' bounding box (rounded)
MAP_ZOOM = 4.1  # fits all districts in a ~340px-wide panel (judgment, checked in Edge at 390px and 1440px)


# --------------------------------------------------------------------------
# Cached resources (raise on failure so errors are never cached)
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading hazard models…", max_entries=2)
def get_bundle(path: str, mtime_ns: int, size: int) -> engine.ModelBundle:
    """Load the verified model bundle once per file version."""
    del mtime_ns, size  # cache key only: replacing the file loads the new model
    return engine.load_bundle(Path(path))


@st.cache_data(show_spinner="Loading district data…", max_entries=2)
def get_districts(path: str, mtime_ns: int) -> tuple[pd.DataFrame, str]:
    """Load the validated dataset and its SHA-256. Streamlit returns a copy per call."""
    del mtime_ns  # cache key only
    data_path = Path(path)
    return engine.load_districts(data_path), engine.file_sha256(data_path)


@st.cache_resource(max_entries=2)
def get_domain(data_sha256: str, _df: pd.DataFrame, features: tuple[str, ...]) -> engine.ApplicabilityDomain:
    """Fit the applicability domain once per dataset version (read-only arrays)."""
    del data_sha256  # cache key only; _df is excluded from hashing
    return engine.ApplicabilityDomain.fit(_df, features)


@st.cache_data(max_entries=2, show_spinner=False)
def get_national_predictions(
    bundle_sha256: str, data_sha256: str, _bundle: engine.ModelBundle, _df: pd.DataFrame
) -> pd.DataFrame:
    """Predict every district's unmodified inputs once per model and dataset version."""
    del bundle_sha256, data_sha256  # cache key only
    return engine.predict_dataset(_bundle, _df)


@st.cache_resource(max_entries=1)
def get_map_layers() -> engine.MapLayers:
    """Load the real fault and coastline layers once per process."""
    return engine.load_map_layers()


def _file_key(path: Path) -> tuple[str, int, int]:
    """Return ``(path, mtime_ns, size)`` or raise a user-facing missing-file error."""
    if not path.is_file():
        raise engine.ArtifactMissingError(
            f"Required file '{path.name}' was not found. Run `python generate_data.py` and `python train.py`."
        )
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


# --------------------------------------------------------------------------
# Sidebar: district + what-if simulator
# --------------------------------------------------------------------------


def _slider_key(district: str, feature: str) -> str:
    return f"whatif::{district}::{feature}"


def _reset_sliders(district: str) -> None:
    """Button callback: drop this district's slider state so sliders return to dataset values."""
    for spec in engine.TUNABLE_FEATURES:
        st.session_state.pop(_slider_key(district, spec.name), None)


def render_sidebar(df: pd.DataFrame, domain: engine.ApplicabilityDomain) -> tuple[str, dict[str, float]]:
    """Render the district selector and grouped, bounded what-if sliders.

    Returns:
        ``(district, overrides)`` where overrides hold every tunable slider value.
    """
    sidebar = st.sidebar
    # Native logo slot: level with the sidebar's collapse button, and still shown top-left when collapsed.
    st.logo(str(theme.LOGO_PATH), size="large", icon_image=str(FAVICON_PATH))
    sidebar.markdown(theme.sidebar_intro_html(), unsafe_allow_html=True)  # static markup only

    with sidebar.container(key="sb_district"):
        st.markdown(
            theme.sidebar_card_head_html("District", hint="Where to assess", icon="pin"), unsafe_allow_html=True
        )
        districts = sorted(df["district_name"].tolist())
        district = st.selectbox(
            "District",
            districts,
            index=districts.index(DEFAULT_DISTRICT) if DEFAULT_DISTRICT in districts else 0,
            key="district",
            label_visibility="collapsed",
        )
        baseline = engine.build_scenario(df, district)
        st.markdown(
            theme.sidebar_meta_html(
                [str(baseline["province"].iloc[0]), f"GeoNames {int(baseline['geonames_id'].iloc[0])}"]
            ),
            unsafe_allow_html=True,
        )

    overrides: dict[str, float] = {}
    for group_index, (group, names) in enumerate(SLIDER_GROUPS.items()):
        with sidebar.container(key=f"sb_group_{group_index}"):
            hint, icon = SLIDER_GROUP_HEADS[group]
            st.markdown(theme.sidebar_card_head_html(group, hint=hint, icon=icon), unsafe_allow_html=True)
            for name in names:
                spec = engine.TUNABLE_BY_NAME[name]
                lower, upper = domain.lower[name], domain.upper[name]
                dataset_value = float(baseline[name].iloc[0])
                overrides[name] = st.slider(
                    f"{spec.label} ({spec.unit})" if spec.unit else spec.label,
                    min_value=float(lower),
                    max_value=float(upper),
                    value=min(max(dataset_value, lower), upper),
                    step=spec.step,
                    format=f"%.{spec.decimals}f",
                    key=_slider_key(district, name),
                    help=(
                        f"Dataset value for {district}: {spec.format(dataset_value)}. "
                        f"Training range: {spec.format(lower)} – {spec.format(upper)}."
                    ),
                )

    changed = len(engine.changed_features(baseline, engine.build_scenario(df, district, overrides)))
    with sidebar.container(key="sb_scenario"):
        st.markdown(
            theme.sidebar_card_head_html(
                "Scenario", f"{changed} of {len(engine.TUNABLE_FEATURES)} changed", hint="Your what-if", icon="sliders"
            ),
            unsafe_allow_html=True,
        )
        st.button(
            "Reset to dataset values",
            on_click=_reset_sliders,
            args=(district,),
            width="stretch",
            icon=":material/restart_alt:",
            disabled=changed == 0,
        )
    # st.markdown rather than st.caption: caption text is a faded grey whose contrast is not verified.
    sidebar.markdown(
        '<div class="sb-foot">Sliders stay within the training range: beyond it, tree models silently freeze '
        "their scores.</div>",
        unsafe_allow_html=True,
    )
    return district, overrides


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------


def waterfall_figure(explanation: engine.HazardExplanation) -> go.Figure:
    """Build a horizontal SHAP waterfall: base value at the bottom, explained output at the top."""
    top = list(explanation.contributions[:MAX_WATERFALL_FEATURES])
    rest = explanation.contributions[MAX_WATERFALL_FEATURES:]
    if rest:
        top.append(engine.Contribution("_other", f"{len(rest)} other features", "", sum(c.value for c in rest)))
    ordered = top[::-1]  # smallest near the base, largest just below the total
    probability = explanation.output_space == "probability"

    # Two-line category labels (name, then value) keep the plot area usable at 360px width;
    # units live in the axis title so bar labels stay short (both found in a 390px Edge render).
    def short(value: float, signed: bool) -> str:
        if probability:
            return f"{value * 100:+.1f}" if signed else f"{value:.0%}"
        return f"{value:+.2f}" if signed else f"{value:.2f}"

    # Base and total values live in their row labels, not beside the bar: at 390px a value drawn
    # outside a short base bar collided with the axis labels (seen in an Edge render).
    labels = [f"Average district (base)<br><b>{short(explanation.base_value, False)}</b>"]
    labels += [f"{c.label}<br>{c.display_value}" if c.display_value else c.label for c in ordered]
    labels += [f"This scenario<br><b>{short(explanation.final_value, False)}</b>"]
    values = [explanation.base_value, *(c.value for c in ordered), explanation.final_value]
    text = ["", *(short(c.value, True) for c in ordered), ""]

    running = np.cumsum([explanation.base_value, *(c.value for c in ordered)])
    low, high = min(0.0, float(running.min())), max(0.0, float(running.max()))
    span = high - low or 1.0
    # Asymmetric room for outside labels; the right side needs more at phone width (judgment, checked at 390px).
    left_pad, right_pad = span * 0.18, span * 0.32

    figure = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["absolute", *["relative"] * len(ordered), "total"],
            y=labels,
            x=values,
            width=0.6,  # thin bars with air between them (dataviz mark spec)
            text=text,
            textposition="outside",
            textfont={"color": theme.TEXT_PRIMARY, "size": 13},
            cliponaxis=False,
            increasing={"marker": {"color": theme.RAISES_COLOR}},
            decreasing={"marker": {"color": theme.LOWERS_COLOR}},
            totals={"marker": {"color": theme.TOTAL_COLOR}},
            connector={"line": {"color": theme.HAIRLINE, "width": 1}},
            hovertemplate="<b>%{text}</b><br>%{y}<extra></extra>",
        )
    )
    # Short axis title: the longer form clipped at 390px width.
    unit = "Uncalibrated P(High)" if probability else "Log-odds of High"
    figure.update_layout(
        height=80 + 46 * len(labels),
        margin={"l": 4, "r": 12, "t": 4, "b": 4},
        bargap=0.35,
        xaxis={
            "title": {"text": unit, "font": {"color": theme.TEXT_SECONDARY}},
            "tickformat": ".0%" if probability else ".1f",
            "tickfont": {"color": theme.TEXT_SECONDARY},
            "range": [low - left_pad, high + right_pad],
            "gridcolor": theme.HAIRLINE,
            "gridwidth": 1,
            "zerolinecolor": theme.CHIP_BORDER,
            "automargin": True,
        },
        yaxis={"automargin": True, "tickfont": {"color": theme.TEXT_SECONDARY}},
        plot_bgcolor=theme.PAGE_BACKGROUND,
        paper_bgcolor=theme.PAGE_BACKGROUND,
        showlegend=False,
        font={"size": 13, "color": theme.TEXT_SECONDARY},
        hoverlabel={"bgcolor": theme.PAGE_BACKGROUND, "font": {"color": theme.TEXT_PRIMARY}},
    )
    return figure


def map_figure(
    national: pd.DataFrame, assessment: engine.Assessment, target: str, layers: engine.MapLayers
) -> go.Figure:
    """National map using emphasis encoding: High districts in the High hue, all others slate.

    No basemap is used: context comes from the real fault and coastline layers, which also
    keeps the map working offline and avoids drawing contested boundaries.
    """
    rows = national[national["target"] == target].copy()
    selected = rows["district_name"] == assessment.district
    scenario_prediction = assessment.predictions[target]
    rows.loc[selected, "label"] = scenario_prediction.label  # the selected district shows the simulated scenario
    rows.loc[selected, "p_high"] = scenario_prediction.p_high
    return emphasis_map_figure(rows, rows["label"] == "High", selected, layers)


def emphasis_map_figure(
    rows: pd.DataFrame,
    emphasized: pd.Series,
    selected: pd.Series | None,
    layers: engine.MapLayers,
    height: int = 430,
) -> go.Figure:
    """Shared national map: emphasised districts in the High hue, all others slate, optional selection ring.

    ``rows`` needs ``latitude, longitude, district_name, province, label, p_high``; ``label`` is the hover text.
    Used by the assessment panel, the climate stress test, and the hotspot view, so all three share one
    verified encoding (see ui_theme module docstring).
    """
    hover = (
        "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} · P(High) %{customdata[3]:.1%}<extra></extra>"
    )

    def trace(frame: pd.DataFrame, color: str, size: float, name: str, hoverable: bool = True) -> go.Scattermap:
        return go.Scattermap(
            lat=frame["latitude"],
            lon=frame["longitude"],
            mode="markers",
            marker={"size": size, "color": color},
            customdata=frame[["district_name", "province", "label", "p_high"]].to_numpy(),
            hovertemplate=hover,
            hoverinfo=None if hoverable else "skip",
            name=name,
        )

    others, highs = rows[~emphasized], rows[emphasized]
    traces = [
        trace(others, theme.PAGE_BACKGROUND, 11, "ring", hoverable=False),  # 2px surface ring under each mark
        trace(others, theme.MAP_OTHER, 7, "Other"),
        trace(highs, theme.PAGE_BACKGROUND, 13, "ring", hoverable=False),
        trace(highs, theme.MAP_HIGH, 9, "Emphasised"),
    ]
    if selected is not None and bool(selected.any()):
        chosen = rows[selected]
        chosen_color = theme.MAP_HIGH if bool(emphasized[selected].iloc[0]) else theme.MAP_OTHER
        traces += [
            trace(chosen, theme.MAP_SELECTED_RING, 20, "Selected district ring", hoverable=False),
            trace(chosen, theme.PAGE_BACKGROUND, 15, "ring", hoverable=False),
            trace(chosen, chosen_color, 10, "Selected district"),
        ]
    figure = go.Figure(traces)
    figure.update_layout(
        map={
            "style": "white-bg",
            "center": MAP_CENTER,
            "zoom": MAP_ZOOM,
            "layers": [
                {
                    "source": dict(layers.coastline),
                    "type": "line",
                    "color": theme.MAP_COAST,
                    "line": {"width": 1},
                    "below": "traces",
                },
                {
                    "source": dict(layers.faults),
                    "type": "line",
                    "color": theme.MAP_FAULT,
                    "opacity": 0.4,  # context layer: must stay behind the district marks (checked at 390px)
                    "line": {"width": 1},
                    "below": "traces",
                },
            ],
        },
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        showlegend=False,
        paper_bgcolor=theme.PAGE_BACKGROUND,
        hoverlabel={"bgcolor": theme.PAGE_BACKGROUND, "font": {"color": theme.TEXT_PRIMARY}},
    )
    return figure


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


def render_explanation(assessment: engine.Assessment, target: str) -> None:
    """Insight sentence, SHAP waterfall, calibrated probabilities, and a table view."""
    prediction = assessment.predictions[target]
    explanation = assessment.explanations[target]
    hazard = engine.TARGET_LABELS[target]
    st.markdown(
        theme.panel_head_html(
            "chart",
            "Score drivers",
            f"What drives the {hazard.lower()} score",
            "Each bar is one input. Red bars push the chance of High up, blue bars pull it down.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(theme.insight_html("In one sentence", assessment.insights[target]), unsafe_allow_html=True)
    st.markdown(
        theme.chart_legend_html(
            [
                (theme.RAISES_COLOR, "+ raises the High score"),
                (theme.LOWERS_COLOR, "− lowers it"),
                (theme.TOTAL_COLOR, "Base and final score"),
            ]
        ),
        unsafe_allow_html=True,
    )
    # Mode bar hidden: it overlapped the top bar at phone width. Hover and pinch-zoom still work.
    st.plotly_chart(
        waterfall_figure(explanation), width="stretch", config={"displayModeBar": False, "responsive": True}
    )
    st.markdown(
        theme.sub_head_html("Chance of each rating", "calibrated probabilities, adding up to 100%"),
        unsafe_allow_html=True,
    )
    # Each bar wears its own row's level colour (the word and shape are printed beside it).
    st.markdown(
        theme.data_table_html(
            [theme.Column("Level"), theme.Column("Probability")],
            [
                [theme.level_pill_html(level), theme.prob_bar_html(p, level, decimals=2)]
                for level, p in zip(engine.CLASS_LABELS, prediction.probabilities, strict=True)
            ],
            caption=f"Calibrated {hazard.lower()} probabilities",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        theme.note_html(
            "The chart explains the uncalibrated base model. Calibration rescales confidence but never changes "
            "which level is most likely."
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Table view of this chart"):
        st.markdown(
            theme.data_table_html(
                [theme.Column("Feature", "strong"), theme.Column("Value", "num"), theme.Column("Contribution", "num")],
                [
                    [c.label, c.display_value, engine.format_contribution(c.value, explanation.output_space)]
                    for c in explanation.contributions
                ],
                caption=f"SHAP contributions to the High {hazard.lower()} score",
            ),
            unsafe_allow_html=True,
        )


def render_map_panel(
    national: pd.DataFrame, assessment: engine.Assessment, target: str, layers: engine.MapLayers
) -> None:
    """National map for the selected hazard, its legend, and a table-view twin."""
    hazard = engine.TARGET_LABELS[target]
    rows = national[national["target"] == target]
    high_count = int((rows["label"] == "High").sum())
    st.markdown(
        theme.panel_head_html(
            "map",
            "National map",
            f"Where {hazard.lower()} risk is High",
            f"{high_count} of {len(rows)} districts are High today. {assessment.district} is ringed.",
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        map_figure(national, assessment, target, layers),
        width="stretch",
        config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
    )
    st.markdown(theme.map_legend_html(hazard), unsafe_allow_html=True)  # static labels, escaped
    with st.expander(f"Table view: districts predicted High for {hazard.lower()}"):
        high_table = rows[rows["label"] == "High"].sort_values("p_high", ascending=False)
        st.markdown(
            theme.data_table_html(
                [theme.Column("District"), theme.Column("P(High)")],
                [
                    [theme.two_line_html(r.district_name, r.province), theme.prob_bar_html(r.p_high, "High")]
                    for r in high_table.itertuples()
                ],
                caption=f"Districts predicted High for {hazard.lower()}",
                max_height_px=360,
            ),
            unsafe_allow_html=True,
        )


def render_profile_panel(assessment: engine.Assessment, df: pd.DataFrame) -> None:
    """District profile as grouped input cards (Climate, Land and exposure, Geography) plus a table view."""
    profile = engine.district_profile(df, assessment)
    standouts = int((profile[["share_below", "share_above"]].max(axis=1) >= theme.STANDOUT_SHARE).sum())
    st.markdown(
        theme.panel_head_html(
            "profile",
            "District data",
            f"{assessment.district} at a glance",
            f"{len(profile)} inputs in three groups, each placed on Pakistan's range from the lowest to the highest "
            f"district. {standouts} stand out from the national middle.",
        ),
        unsafe_allow_html=True,
    )

    def card(row: pd.Series) -> str:
        return theme.profile_card_html(
            label=str(row["Input"]),
            value=str(row["This scenario"]),
            dataset_value=str(row["Dataset value"]) if row["Changed"] else None,
            province_median=str(row["Province median"]),
            vs_province_pct=float(row["vs_province_pct"]),
            position=float(row["range_position"]),
            province_position=float(row["province_position"]),
            low=str(row["pakistan_min_label"]),
            high=str(row["pakistan_max_label"]),
            share_below=float(row["share_below"]),
            share_above=float(row["share_above"]),
        )

    # The same groups as the sidebar, then the fixed real geography.
    by_feature = profile.set_index("feature", drop=False)
    html_groups = []
    for name, features in SLIDER_GROUPS.items():
        hint, icon = SLIDER_GROUP_HEADS[name]
        html_groups.append(
            theme.profile_group_html(
                icon,
                name,
                f"{len(features)} inputs · {hint} · adjustable in the sidebar",
                "Synthetic",
                [card(by_feature.loc[f]) for f in features],
            )
        )
    real = profile[profile["Source"] == "Real"]
    html_groups.append(
        theme.profile_group_html(
            "mountain",
            "Geography",
            f"{len(real)} inputs · elevation and distances · fixed for each district",
            "Real",
            [card(row) for _, row in real.iterrows()],
        )
    )
    st.markdown(f'<div class="pgroups">{"".join(html_groups)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="profile-key"><span class="k"><span class="kdot"></span>This scenario</span>'
        '<span class="k"><span class="kmed"></span>Province median</span>'
        '<span class="k"><span class="kchg"></span>Changed in the sidebar</span>'
        '<span class="k">★ Stands out: more extreme than 90% of districts</span>'
        '<span class="k">Bars run from the lowest to the highest district in Pakistan</span></div>',
        unsafe_allow_html=True,
    )  # static markup only
    with st.expander("Table view of all inputs"):
        st.markdown(
            theme.data_table_html(
                [
                    theme.Column("Input", "strong nowrap"),
                    theme.Column("This scenario", "nowrap"),
                    theme.Column("Dataset value", "nowrap"),
                    theme.Column("Province median", "nowrap"),
                    theme.Column("Pakistan range", "nowrap"),
                    theme.Column("Rank"),
                    theme.Column("Source"),
                ],
                [
                    [
                        row["Input"],
                        row["This scenario"],
                        row["Dataset value"],
                        row["Province median"],
                        row["Pakistan range"],
                        theme.rank_sentence(row["share_below"], row["share_above"]),
                        row["Source"],
                    ]
                    for _, row in profile.iterrows()
                ],
                caption=f"{assessment.district} district profile",
            ),
            unsafe_allow_html=True,
        )


@st.cache_data(max_entries=16, show_spinner="Re-scoring every district under the scenario…")
def get_stress_test(
    bundle_sha256: str,
    data_sha256: str,
    temp_shift_c: float,
    rain_factor: float,
    _bundle: engine.ModelBundle,
    _df: pd.DataFrame,
    _domain: engine.ApplicabilityDomain,
) -> pd.DataFrame:
    """Stress-test results, cached per model, dataset, and scenario setting."""
    del bundle_sha256, data_sha256  # cache key only
    return engine.stress_test(_bundle, _df, _domain, temp_shift_c, rain_factor)


def _map_rows(results: pd.DataFrame, target: str, stressed: bool) -> pd.DataFrame:
    """Shape stress-test rows for the shared emphasis map (hover text marks extrapolated results)."""
    rows = results[results["target"] == target].copy()
    prefix = "stressed" if stressed else "current"
    rows["label"] = rows[f"{prefix}_label"]
    rows["p_high"] = rows[f"{prefix}_p_high"]
    if stressed:
        rows.loc[rows["extrapolated"], "label"] = rows["label"] + " (extrapolated)"
    return rows


def render_stress_test(
    bundle: engine.ModelBundle,
    df: pd.DataFrame,
    domain: engine.ApplicabilityDomain,
    data_sha256: str,
    layers: engine.MapLayers,
) -> None:
    """Feature 4: apply a stated climate shift to every district and show which ones become High."""
    st.markdown(
        theme.panel_head_html(
            "flame",
            "Climate stress test",
            "Which districts would turn High in a hotter, wetter climate?",
            "Shift summer temperature and rainfall for every district at once and re-score all 150.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        theme.planning_label_html(
            "what-if analysis, not a projection or forecast",
            "Every district's summer temperature and rainfall are shifted by the amounts below and re-scored with the "
            "same deployed models. Nothing else changes. Results flagged extrapolated go beyond the data the models "
            "were trained on and deserve less confidence.",
        ),
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="large")
    temp_shift = left.slider(
        "Summer temperature shift (°C)", *engine.STRESS_TEMP_SHIFT_RANGE, value=2.0, step=0.5, key="stress_temp"
    )
    rain_factor = right.slider(
        "Annual rainfall multiplier (×)", *engine.STRESS_RAIN_FACTOR_RANGE, value=1.5, step=0.1, key="stress_rain"
    )
    results = get_stress_test(bundle.sha256, data_sha256, float(temp_shift), float(rain_factor), bundle, df, domain)

    newly = results[results["becomes_high"]].sort_values("change_pts", ascending=False)
    extrapolated_share = results.loc[results["target"].isin(["flood_risk", "heatwave_risk"]), "extrapolated"].mean()
    tiles = [(str(len(newly)), "district-hazard results would become High")]
    tiles += [
        (str(int((newly["target"] == t).sum())), f"{engine.TARGET_LABELS[t]} districts newly High")
        for t in bundle.targets
    ]
    tiles.append((f"{extrapolated_share:.0%}", "of flood and heatwave results extrapolated"))
    st.markdown(theme.stat_tiles_html(tiles), unsafe_allow_html=True)

    st.markdown(
        theme.sub_head_html(
            "Newly High districts, biggest jump first", f"at +{temp_shift:g} °C and rainfall ×{rain_factor:g}"
        ),
        unsafe_allow_html=True,
    )
    if newly.empty:
        st.markdown(theme.note_html("No district moves to High under this scenario."), unsafe_allow_html=True)
    else:
        st.markdown(
            theme.data_table_html(
                [
                    theme.Column("District"),
                    theme.Column("Hazard"),
                    theme.Column("Now"),
                    theme.Column("Under scenario"),
                    theme.Column("Rise", "num"),
                    theme.Column("Confidence"),
                ],
                [
                    [
                        theme.two_line_html(r.district_name, r.province),
                        engine.TARGET_LABELS[r.target],
                        theme.level_pill_html(r.current_label, f"· {r.current_p_high:.0%}"),
                        theme.level_pill_html("High", f"· {r.stressed_p_high:.0%}"),
                        f"+{r.change_pts:.1f} pts",
                        theme.badge_html("⚠ Extrapolated", "changed")
                        if r.extrapolated
                        else theme.muted_text_html("Within training range"),
                    ]
                    for r in newly.itertuples()
                ],
                caption="Districts that would become High under the scenario",
                max_height_px=430,
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        theme.note_html(
            "Seismic results never change here: the seismic model does not use temperature or rainfall. "
            "Extrapolated means at least one shifted input is beyond the range the model was trained on."
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        theme.sub_head_html("Today versus the scenario", "pick a hazard to compare the two maps"),
        unsafe_allow_html=True,
    )
    hazard = st.segmented_control(
        "Compare hazard",
        options=[t for t in bundle.targets if t != "seismic_risk"],
        format_func=lambda t: engine.TARGET_LABELS[t],
        default="heatwave_risk",
        required=True,
        key="stress_hazard",
        label_visibility="collapsed",
    )
    now_col, stress_col = st.columns(2, gap="large")
    scenario_title = f"With +{temp_shift:g} °C and rain ×{rain_factor:g}"
    for column, stressed, title in ((now_col, False, "Today"), (stress_col, True, scenario_title)):
        with column:
            rows = _map_rows(results, hazard, stressed)
            emphasized = rows[f"{'stressed' if stressed else 'current'}_label"] == "High"
            st.markdown(
                theme.sub_head_html(
                    title, f"{int(emphasized.sum())} districts High for {engine.TARGET_LABELS[hazard].lower()}"
                ),
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                emphasis_map_figure(rows, emphasized, None, layers, height=380),
                width="stretch",
                config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
                key=f"stress_map_{stressed}",
            )
    st.markdown(
        theme.map_legend_html(engine.TARGET_LABELS[hazard], other_label="Medium or Low", selected=False),
        unsafe_allow_html=True,
    )


def render_hotspots(
    national: pd.DataFrame,
    bundle: engine.ModelBundle,
    df: pd.DataFrame,
    domain: engine.ApplicabilityDomain,
    data_sha256: str,
    layers: engine.MapLayers,
) -> None:
    """Feature 5: districts predicted High for two or more hazards at once."""
    st.markdown(
        theme.panel_head_html(
            "layers",
            "Multi-hazard hotspots",
            "Where do hazards pile up?",
            "Districts rated High for two or more hazards at once need joined-up preparedness.",
        ),
        unsafe_allow_html=True,
    )
    conditions = st.radio(
        "Conditions",
        ["Current conditions", "Climate stress-test scenario"],
        horizontal=True,
        key="hotspot_conditions",
        help="The scenario uses the temperature shift and rainfall multiplier set in the stress-test tab.",
    )
    if conditions == "Current conditions":
        predictions, label_col, p_col = national, "label", "p_high"
    else:
        temp_shift = float(st.session_state.get("stress_temp", 2.0))
        rain_factor = float(st.session_state.get("stress_rain", 1.5))
        st.markdown(
            theme.planning_label_html(
                "what-if analysis, not a projection or forecast",
                f"Hotspots recomputed with summer temperature +{temp_shift:g} °C and rainfall ×{rain_factor:g}.",
            ),
            unsafe_allow_html=True,
        )
        predictions = get_stress_test(bundle.sha256, data_sha256, temp_shift, rain_factor, bundle, df, domain)
        label_col, p_col = "stressed_label", "stressed_p_high"

    hotspots = engine.multi_hazard_hotspots(predictions, label_col, p_col)
    all_three = int((hotspots["high_count"] >= len(bundle.targets)).sum()) if not hotspots.empty else 0
    st.markdown(
        theme.stat_tiles_html(
            [
                (str(len(hotspots)), "districts are High for two or more hazards"),
                (str(all_three), "High for all three hazards"),
                (str(hotspots["province"].nunique()) if not hotspots.empty else "0", "provinces affected"),
            ]
        ),
        unsafe_allow_html=True,
    )
    table_col, map_col = st.columns([7, 5], gap="large")
    with table_col:
        if hotspots.empty:
            st.markdown(theme.note_html("No district is High for two or more hazards."), unsafe_allow_html=True)
        else:
            st.markdown(
                theme.data_table_html(
                    [
                        theme.Column("District"),
                        theme.Column("High for"),
                        theme.Column("Sum P(High)", "num"),
                    ],
                    [
                        [
                            theme.two_line_html(r.district_name, r.province),
                            theme.pill_row_html(
                                [theme.level_pill_html("High", label=h.strip()) for h in r.high_hazards.split(",")]
                            ),
                            f"{r.combined_p_high:.2f}",
                        ]
                        for r in hotspots.itertuples()
                    ],
                    caption="Districts High for two or more hazards",
                    max_height_px=440,
                ),
                unsafe_allow_html=True,
            )
    with map_col:
        rows = predictions[predictions["target"] == bundle.targets[0]].copy()
        by_name = hotspots.set_index("district_name")
        is_hotspot = rows["district_name"].isin(by_name.index)
        rows["label"] = "Fewer than two High hazards"
        rows.loc[is_hotspot, "label"] = (
            "High: " + rows.loc[is_hotspot, "district_name"].map(by_name["high_hazards"]) + " · mean"
        )
        # Hover shows the mean P(High) across the district's High hazards (not a combined probability).
        rows["p_high"] = 0.0
        rows.loc[is_hotspot, "p_high"] = rows.loc[is_hotspot, "district_name"].map(
            by_name["combined_p_high"] / by_name["high_count"]
        )
        emphasized = rows["district_name"].isin(hotspots["district_name"])
        st.plotly_chart(
            emphasis_map_figure(rows, emphasized, None, layers, height=400),
            width="stretch",
            config={"displayModeBar": False, "scrollZoom": False, "responsive": True},
            key="hotspot_map",
        )
        st.markdown(
            theme.map_legend_html(
                "", emphasized_label="High for 2+ hazards", other_label="Other districts", selected=False
            ),
            unsafe_allow_html=True,
        )


def render_model_card(bundle: engine.ModelBundle, issues: tuple[engine.ProvenanceIssue, ...]) -> None:
    """Provenance checklist, honest metrics, and limitations."""
    meta = bundle.metadata
    checklist = engine.provenance_checklist(bundle, issues)
    passed = sum(item.passed for item in checklist)
    st.markdown(
        theme.panel_head_html(
            "profile",
            "Model card",
            "Three questions to ask before trusting a score",
            "Is the right model running? How accurate is it? What can it not tell you?",
        ),
        unsafe_allow_html=True,
    )
    with st.container(key="card_provenance"):
        st.markdown(
            theme.card_head_html(
                "shield",
                "Is the right model running on the right data?",
                "Four provenance checks run at every start-up against the files being served.",
                badge=(f"{passed} of {len(checklist)} passed", "real" if passed == len(checklist) else "changed"),
                eyebrow="1 · Provenance checks",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            theme.checklist_html([(item.label, item.passed, item.detail) for item in checklist]),
            unsafe_allow_html=True,
        )

    with st.container(key="card_performance"):
        render_performance(bundle, meta)

    with st.container(key="card_limitations"):
        st.markdown(
            theme.card_head_html(
                "alert",
                "What can it not tell you?",
                "Read these five limits before relying on any number on this page.",
                badge=("Simulated data", "changed"),
                tone="warn",
                eyebrow="3 · Limitations",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(theme.limitations_html(LIMITATIONS), unsafe_allow_html=True)


def render_performance(bundle: engine.ModelBundle, meta: dict) -> None:
    """Nested cross-validation table, how to read it, the decision rule, and per-hazard inputs."""
    search = meta["search"]
    st.markdown(
        theme.card_head_html(
            "chart",
            "How accurate is it on districts it never saw?",
            f"Nested cross-validation: {search['outer_folds']} outer × {search['inner_folds']} inner folds, "
            f"{search['n_trials']} Optuna trials, class-weighted {', '.join(search['families'])}. "
            "Mean ± standard deviation over outer folds.",
            badge=("Held-out estimates", "synthetic"),
            eyebrow="2 · Performance",
        ),
        unsafe_allow_html=True,
    )
    summary = meta["nested_cv_summary"]
    rows = []
    for target in bundle.targets:
        tuned, cal, raw = (
            summary[target]["recall_tuned"],
            summary[target]["temperature"],
            summary[target]["uncalibrated"],
        )
        held_out = meta["per_class_held_out"][target]
        rows.append(
            {
                "Hazard": engine.TARGET_LABELS[target],
                "Model": f"{meta['model_families'].get(target, '?')} · {len(bundle.features_for(target))} inputs",
                "High from": f"{bundle.high_thresholds[target]:.0%}",
                "High recall": f"{cal['high_recall']['mean']:.2f} → {tuned['high_recall']['mean']:.2f}",
                "Missed / false High": (
                    f"{held_out['argmax']['high_missed']}/{held_out['argmax']['high_false_alarms']} → "
                    f"{held_out['recall_tuned']['high_missed']}/{held_out['recall_tuned']['high_false_alarms']}"
                ),
                "Macro-F1": f"{tuned['macro_f1']['mean']:.3f} ± {tuned['macro_f1']['std']:.3f}",
                "Brier": f"{raw['brier']['mean']:.3f} → {cal['brier']['mean']:.3f}",
            }
        )
    table = pd.DataFrame(rows)
    st.markdown(
        theme.data_table_html(
            [theme.Column(name, "strong" if name == "Hazard" else "nowrap") for name in table.columns],
            table.to_numpy().tolist(),
            caption="Nested cross-validation performance per hazard",
        ),
        unsafe_allow_html=True,
    )
    n_high = int(meta["per_class_held_out"][bundle.targets[0]]["argmax"]["High"]["support"])
    st.markdown(
        theme.note_html(
            "Arrows read argmax rule → recall-tuned rule; Brier reads raw → calibrated. Missed / false High counts "
            f"are out of {n_high} High and {int(meta['n_rows']) - n_high} other districts (pooled held-out predictions)."
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        theme.insight_html(
            "Decision rule",
            "A district is flagged High when its calibrated P(High) reaches the hazard's threshold, chosen on training "
            "folds to favour recall (F2). A missed High district is treated as costlier than a false alarm; the "
            "missed / false-alarm counts above show that trade-off on held-out predictions.",
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Inputs each hazard model uses (isolated by design)"):
        st.markdown(
            theme.data_table_html(
                [theme.Column("Hazard", "strong"), theme.Column("Inputs")],
                [
                    [engine.TARGET_LABELS[t], ", ".join(engine.FEATURE_LABELS[f] for f in bundle.features_for(t))]
                    for t in bundle.targets
                ],
                caption="Inputs used by each hazard model",
            ),
            unsafe_allow_html=True,
        )


def render_dashboard() -> None:
    """Load resources and lay out the page."""
    st.markdown(theme.page_css(), unsafe_allow_html=True)  # static CSS from constants only

    data_path, data_mtime, _ = _file_key(engine.default_data_path())
    model_path, model_mtime, model_size = _file_key(engine.default_model_path())
    df, data_sha256 = get_districts(data_path, data_mtime)
    bundle = get_bundle(model_path, model_mtime, model_size)
    domain = get_domain(data_sha256, df, bundle.numeric_features)
    issues = engine.audit_provenance(bundle, df, data_sha256)
    national = get_national_predictions(bundle.sha256, data_sha256, bundle, df)
    layers = get_map_layers()

    district, overrides = render_sidebar(df, domain)
    critical = [issue.message for issue in issues if issue.severity == "critical"]
    assessment = engine.build_assessment(bundle, df, domain, district, overrides, extra_warnings=critical)

    trained = str(bundle.metadata.get("trained_at_utc", ""))[:10]
    model_label = f"Model {bundle.sha256[:8]} · trained {trained} · {bundle.calibration_method} calibration"
    # Layout order: centered Nigehban brand header -> persistent SIMULATED SCENARIO banner -> district bar.
    st.markdown(theme.brand_header_html(), unsafe_allow_html=True)  # static markup only
    st.markdown(theme.banner_html(), unsafe_allow_html=True)  # static text only
    for message in critical:
        st.error(message)

    # 01 · Risk overview: district bar, scenario chips, the three risk cards, exports.
    with st.container(key="section_overview"):
        st.markdown(
            theme.page_section_head_html(
                1,
                "Risk overview",
                f"How exposed is {district}?",
                "The chance of a High rating for flood, heatwave and seismic risk, using the inputs in the sidebar.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(theme.phone_hint_html(), unsafe_allow_html=True)  # static markup only
        st.markdown(theme.header_html(district, assessment.province, model_label), unsafe_allow_html=True)
        if assessment.modified_features:
            chips = []
            for name in assessment.modified_features:
                spec = engine.TUNABLE_BY_NAME[name]
                before = spec.format(float(assessment.baseline[name].iloc[0]))
                after = spec.format(float(assessment.scenario[name].iloc[0]))
                chips.append(f"{spec.label}: {before} → {after}")
            st.markdown(theme.chips_html("Simulated inputs", chips), unsafe_allow_html=True)
        else:
            st.markdown(theme.chips_html("Scenario", ["Unmodified dataset inputs"]), unsafe_allow_html=True)
        for warning in assessment.warnings:
            if warning not in critical:
                st.warning(warning)

        cards = "".join(
            theme.kpi_card_html(
                engine.TARGET_LABELS[target],
                prediction.label,
                prediction.p_high,
                assessment.baseline_predictions[target].p_high if assessment.modified_features else None,
                bundle.high_thresholds[target],
            )
            for target, prediction in assessment.predictions.items()
        )
        st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)  # every dynamic value is escaped

        # Exports after the risk numbers: on a phone the columns stack, and risk must come before downloads.
        with st.container(key="export_bar"):
            note_col, csv_col, txt_col = st.columns([6, 2, 2], vertical_alignment="center")
            note_col.markdown(theme.export_head_html(), unsafe_allow_html=True)  # static markup only
            slug = re.sub(r"[^a-z0-9]+", "-", district.lower()).strip("-")
            csv_col.download_button(
                "Download CSV",
                engine.assessment_to_csv(assessment),
                f"{slug}-assessment.csv",
                "text/csv",
                on_click="ignore",
                width="stretch",
                icon=":material/table_view:",
                key="dl_csv",
                help="Download this assessment as CSV (one row per hazard)",
            )
            txt_col.download_button(
                "Download report",
                engine.assessment_to_text(assessment),
                f"{slug}-assessment.txt",
                "text/plain",
                on_click="ignore",
                width="stretch",
                icon=":material/description:",
                key="dl_report",
                help="Download this assessment as a plain-text report",
            )

    # 02 · Why this assessment: hazard picker drives the explanation and the national map.
    with st.container(key="section_detail"):
        st.markdown(
            theme.page_section_head_html(
                2,
                "Assessment detail",
                "Why did it get this rating?",
                "Pick a hazard below. The score drivers and the national map both follow your choice.",
            ),
            unsafe_allow_html=True,
        )
        target = st.segmented_control(
            "Hazard",
            options=list(bundle.targets),
            format_func=lambda t: f"{engine.TARGET_LABELS[t]} · {assessment.predictions[t].label}",
            default=bundle.targets[0],
            required=True,
            key="hazard",
            label_visibility="collapsed",
        )
        left, right = st.columns([7, 5], gap="large")
        with left, st.container(key="panel_explanation"):
            render_explanation(assessment, target)
        with right, st.container(key="panel_map"):
            render_map_panel(national, assessment, target, layers)
        with st.container(key="panel_profile"):
            render_profile_panel(assessment, df)

    # 03 · Planning ahead.
    with st.container(key="section_planning"):
        st.markdown(
            theme.page_section_head_html(
                3,
                "Planning ahead",
                "What if the climate shifts?",
                "Two national what-if tools: districts that would turn High, and places where hazards overlap.",
            ),
            unsafe_allow_html=True,
        )
        stress_tab, hotspot_tab = st.tabs(["Climate stress test", "Multi-hazard hotspots"])
        with stress_tab, st.container(key="panel_stress"):
            render_stress_test(bundle, df, domain, data_sha256, layers)
        with hotspot_tab, st.container(key="panel_hotspots"):
            render_hotspots(national, bundle, df, domain, data_sha256, layers)

    # 04 · Model and data.
    with st.container(key="section_model"):
        st.markdown(
            theme.page_section_head_html(
                4,
                "Model and data",
                "Can you trust these scores?",
                "How the models were built and checked, how accurate they are, and where their limits lie.",
            ),
            unsafe_allow_html=True,
        )
        model_tab, data_tab = st.tabs(["Model card", "Dataset labels"])
        with model_tab, st.container(key="panel_model_card"):
            render_model_card(bundle, issues)
        with data_tab, st.container(key="panel_dataset"):
            st.markdown(
                theme.panel_head_html(
                    "database",
                    "Dataset labels",
                    "The training labels for all 150 districts",
                    "Synthetic labels stored in the dataset (training targets), not model predictions. Sort any column.",
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(engine.dataset_labels_table(df), hide_index=True, width="stretch", height=360)


def main() -> None:
    """Entry point with a single error boundary: users never see a traceback."""
    st.set_page_config(
        page_title="Nigehban · Multi-Hazard Risk (simulated)", page_icon=str(FAVICON_PATH), layout="wide"
    )
    try:
        render_dashboard()
    except engine.RiskEngineError as exc:
        logger.warning("User-facing engine error: %s", exc)
        st.error(str(exc))
    except Exception:
        incident = uuid.uuid4().hex[:8]
        logger.exception("Unhandled dashboard error [incident %s]", incident)
        st.error(f"Something went wrong while building this assessment. Reference: {incident}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    main()
