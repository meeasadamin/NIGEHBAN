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
    sidebar.markdown(theme.sidebar_brand_html(), unsafe_allow_html=True)  # static markup only

    with sidebar.container(key="sb_district"):
        st.markdown(theme.sidebar_card_head_html("District"), unsafe_allow_html=True)
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
            st.markdown(theme.sidebar_card_head_html(group), unsafe_allow_html=True)
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
            theme.sidebar_card_head_html("Scenario", f"{changed} of {len(engine.TUNABLE_FEATURES)} changed"),
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
            width=0.55,  # thin bars with air between them (dataviz mark spec)
            text=text,
            textposition="outside",
            textfont={"color": theme.TEXT_SECONDARY},
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
        height=90 + 44 * len(labels),
        margin={"l": 4, "r": 12, "t": 8, "b": 8},
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
        theme.section_head_html(
            f"Why the {hazard.lower()} score looks like this", "SHAP explanation of the High score", panel=True
        ),
        unsafe_allow_html=True,
    )
    st.markdown(f"**Why:** {assessment.insights[target]}")
    # Mode bar hidden: it overlapped the top bar at phone width. Hover and pinch-zoom still work.
    st.plotly_chart(
        waterfall_figure(explanation), width="stretch", config={"displayModeBar": False, "responsive": True}
    )
    st.markdown(
        "Red bars with **+** raise the High score; blue bars with **−** lower it. The chart explains the "
        "uncalibrated base model; the probabilities below are after temperature calibration, which rescales "
        "confidence but never changes the predicted level."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Level": [f"{theme.LEVEL_SYMBOLS[c]} {c}" for c in engine.CLASS_LABELS],
                "Calibrated probability": prediction.probabilities,
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Calibrated probability": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0)
        },
    )
    with st.expander("Table view of this chart"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": [c.label for c in explanation.contributions],
                    "Value": [c.display_value for c in explanation.contributions],
                    "Contribution": [
                        engine.format_contribution(c.value, explanation.output_space) for c in explanation.contributions
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )


def render_map_panel(
    national: pd.DataFrame, assessment: engine.Assessment, target: str, layers: engine.MapLayers
) -> None:
    """National map for the selected hazard, its legend, and a table-view twin."""
    hazard = engine.TARGET_LABELS[target]
    rows = national[national["target"] == target]
    high_count = int((rows["label"] == "High").sum())
    st.markdown(
        theme.section_head_html(
            f"Where High {hazard.lower()} risk is modelled",
            f"{high_count} of {len(rows)} districts predicted High with unmodified inputs",
            panel=True,
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
        st.dataframe(
            high_table[["district_name", "province", "p_high"]].rename(
                columns={"district_name": "District", "province": "Province", "p_high": "P(High)"}
            ),
            hide_index=True,
            width="stretch",
            column_config={"P(High)": st.column_config.NumberColumn(format="percent")},
        )


def render_profile_panel(assessment: engine.Assessment, df: pd.DataFrame) -> None:
    """Full-width district profile: the table has seven columns and needs the width (checked in Edge)."""
    st.markdown(
        theme.section_head_html(
            "District profile",
            f"{assessment.district} compared with {assessment.province} and all {len(df)} districts. "
            "Real = reference data; Synthetic = simulated.",
            panel=True,
        ),
        unsafe_allow_html=True,
    )
    st.dataframe(
        engine.district_profile(df, assessment),
        hide_index=True,
        width="stretch",
        column_config={
            "Changed": st.column_config.CheckboxColumn(
                "Changed", help="Simulated input differs from the dataset value"
            ),
            "Source": st.column_config.TextColumn("Source", help="Real reference data or synthetic"),
        },
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
        theme.planning_label_html(
            "what-if analysis, not a projection or forecast",
            "Every district's summer temperature and rainfall are shifted by the amounts below and re-scored with the "
            "same deployed models. Nothing else changes. Results flagged extrapolated go beyond the data the models "
            "were trained on and deserve less confidence.",
        ),
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    temp_shift = left.slider(
        "Summer temperature shift (°C)", *engine.STRESS_TEMP_SHIFT_RANGE, value=2.0, step=0.5, key="stress_temp"
    )
    rain_factor = right.slider(
        "Annual rainfall multiplier (×)", *engine.STRESS_RAIN_FACTOR_RANGE, value=1.5, step=0.1, key="stress_rain"
    )
    results = get_stress_test(bundle.sha256, data_sha256, float(temp_shift), float(rain_factor), bundle, df, domain)

    newly = results[results["becomes_high"]].sort_values("change_pts", ascending=False)
    counts = ", ".join(f"{engine.TARGET_LABELS[t]} {int((newly['target'] == t).sum())}" for t in bundle.targets)
    st.markdown(
        theme.section_head_html(
            f"{len(newly)} district-hazard results would become High",
            f"+{temp_shift:g} °C and rainfall ×{rain_factor:g} · by hazard: {counts} · largest rise in P(High) first",
            panel=True,
        ),
        unsafe_allow_html=True,
    )
    if newly.empty:
        st.markdown("No district moves to High under this scenario.")
    else:
        st.dataframe(
            pd.DataFrame(
                {
                    "District": newly["district_name"],
                    "Province": newly["province"],
                    "Hazard": newly["target"].map(engine.TARGET_LABELS),
                    "Now": newly["current_label"]
                    + " · "
                    + (newly["current_p_high"] * 100).round(0).astype(int).astype(str)
                    + "%",
                    "Under scenario": "High · "
                    + (newly["stressed_p_high"] * 100).round(0).astype(int).astype(str)
                    + "%",
                    "Rise in P(High)": newly["change_pts"].map(lambda v: f"+{v:.1f} pts"),
                    "Confidence": newly["extrapolated"].map(
                        {True: "⚠ Extrapolated: input beyond training range", False: "Within training range"}
                    ),
                }
            ),
            hide_index=True,
            width="stretch",
            height=min(38 * (len(newly) + 1) + 4, 400),
        )
    extrapolated_share = results.loc[results["target"].isin(["flood_risk", "heatwave_risk"]), "extrapolated"].mean()
    st.markdown(
        f"Seismic results never change here: the seismic model does not use temperature or rainfall. "
        f"{extrapolated_share:.0%} of flood and heatwave results under this scenario are extrapolated."
    )

    hazard = st.segmented_control(
        "Compare hazard",
        options=[t for t in bundle.targets if t != "seismic_risk"],
        format_func=lambda t: engine.TARGET_LABELS[t],
        default="heatwave_risk",
        required=True,
        key="stress_hazard",
    )
    now_col, stress_col = st.columns(2, gap="large")
    for column, stressed, title in ((now_col, False, "Current conditions"), (stress_col, True, "Under the scenario")):
        with column:
            rows = _map_rows(results, hazard, stressed)
            emphasized = rows[f"{'stressed' if stressed else 'current'}_label"] == "High"
            st.markdown(
                theme.section_head_html(
                    title,
                    f"{int(emphasized.sum())} districts High for {engine.TARGET_LABELS[hazard].lower()}",
                    panel=True,
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
    st.markdown(
        theme.section_head_html(
            f"{len(hotspots)} districts are High for two or more hazards",
            "Compound risk: one district facing several hazards at once needs coordinated preparedness.",
            panel=True,
        ),
        unsafe_allow_html=True,
    )
    table_col, map_col = st.columns([7, 5], gap="large")
    with table_col:
        st.dataframe(
            hotspots.rename(
                columns={
                    "district_name": "District",
                    "province": "Province",
                    "high_count": "High hazards",
                    "high_hazards": "Which hazards",
                    "combined_p_high": "Sum P(High)",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Sum P(High)": st.column_config.NumberColumn(
                    format="%.2f", help="Sum of P(High) across the High hazards; used to rank ties"
                )
            },
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
    st.markdown(
        theme.section_head_html(
            "Provenance checks", "Run at every start-up against the files being served", panel=True
        ),
        unsafe_allow_html=True,
    )
    checklist = engine.provenance_checklist(bundle, issues)
    st.markdown(
        theme.checklist_html([(item.label, item.passed, item.detail) for item in checklist]), unsafe_allow_html=True
    )

    search = meta["search"]
    st.markdown(
        theme.section_head_html(
            "Performance (nested cross-validation)",
            f"{search['outer_folds']} outer × {search['inner_folds']} inner folds, {search['n_trials']} Optuna trials, "
            f"class-weighted {', '.join(search['families'])}. Mean ± standard deviation over outer folds.",
            panel=True,
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
                "Inputs": len(bundle.features_for(target)),
                "Model": meta["model_families"].get(target, "?"),
                "High at P(High) ≥": f"{bundle.high_thresholds[target]:.0%}",
                "High recall (argmax → tuned)": f"{cal['high_recall']['mean']:.2f} → {tuned['high_recall']['mean']:.2f}",
                "Missed / false High (argmax → tuned)": (
                    f"{held_out['argmax']['high_missed']}/{held_out['argmax']['high_false_alarms']} → "
                    f"{held_out['recall_tuned']['high_missed']}/{held_out['recall_tuned']['high_false_alarms']}"
                ),
                "Macro-F1": f"{tuned['macro_f1']['mean']:.3f} ± {tuned['macro_f1']['std']:.3f}",
                "Brier (raw → cal.)": f"{raw['brier']['mean']:.3f} → {cal['brier']['mean']:.3f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "**Decision rule:** a district is flagged High when its calibrated P(High) reaches the hazard's threshold, "
        "chosen on training folds to favour recall (F2). A missed High district is treated as costlier than a false "
        "alarm; the missed/false-alarm counts above show that trade-off on held-out predictions."
    )
    with st.expander("Inputs each hazard model uses (isolated by design)"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Hazard": [engine.TARGET_LABELS[t] for t in bundle.targets],
                    "Inputs": [
                        ", ".join(engine.FEATURE_LABELS[f] for f in bundle.features_for(t)) for t in bundle.targets
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown(theme.section_head_html("Limitations", panel=True), unsafe_allow_html=True)
    st.markdown(
        "- Climate, exposure, event counts, and all labels are synthetic. Real inputs: district names, coordinates, "
        "point elevation, active-fault and Makran subduction geometry, coastline (`data/reference/SOURCES.md`).\n"
        "- Labels are relative ranks within Pakistan (top ~15% = High), not absolute hazard levels.\n"
        "- **No historical baseline:** there is no historical event dataset in this project, so changes are shown "
        "against the district's unmodified dataset inputs only.\n"
        "- About 150 rows: every metric has wide uncertainty; see the ± values.\n"
        "- Not for operational disaster response."
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
                1, "Risk overview", "Predicted risk level for each hazard under the inputs set in the sidebar."
            ),
            unsafe_allow_html=True,
        )
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
        note_col, csv_col, txt_col = st.columns([6, 2, 2], vertical_alignment="center")
        note_col.markdown(
            '<div class="toolbar-note">Download this assessment for your records.</div>', unsafe_allow_html=True
        )
        slug = re.sub(r"[^a-z0-9]+", "-", district.lower()).strip("-")
        csv_col.download_button(
            "CSV",
            engine.assessment_to_csv(assessment),
            f"{slug}-assessment.csv",
            "text/csv",
            on_click="ignore",
            width="stretch",
            icon=":material/download:",
            help="Download this assessment as CSV (one row per hazard)",
        )
        txt_col.download_button(
            "Report",
            engine.assessment_to_text(assessment),
            f"{slug}-assessment.txt",
            "text/plain",
            on_click="ignore",
            width="stretch",
            icon=":material/description:",
            help="Download this assessment as a plain-text report",
        )

    # 02 · Why this assessment: hazard picker drives the explanation and the national map.
    with st.container(key="section_detail"):
        st.markdown(
            theme.page_section_head_html(
                2,
                "Assessment detail",
                "Choose a hazard: the explanation and the national map below both follow it.",
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
        left, right = st.columns([7, 5], gap="medium")
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
                "National what-if tools: which districts would become High, and where hazards compound.",
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
                4, "Model and data", "How the models were trained, checked and validated, and what they cannot do."
            ),
            unsafe_allow_html=True,
        )
        model_tab, data_tab = st.tabs(["Model card", "Dataset labels"])
        with model_tab, st.container(key="panel_model_card"):
            render_model_card(bundle, issues)
        with data_tab, st.container(key="panel_dataset"):
            st.markdown("Synthetic labels stored in the dataset (training targets), not model predictions.")
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
