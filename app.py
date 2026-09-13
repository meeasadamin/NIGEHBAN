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
    sidebar.markdown('<div class="sidebar-group first">District</div>', unsafe_allow_html=True)
    districts = sorted(df["district_name"].tolist())
    district = sidebar.selectbox(
        "District",
        districts,
        index=districts.index(DEFAULT_DISTRICT) if DEFAULT_DISTRICT in districts else 0,
        key="district",
        label_visibility="collapsed",
    )
    baseline = engine.build_scenario(df, district)
    sidebar.markdown(f"{baseline['province'].iloc[0]} · GeoNames id {int(baseline['geonames_id'].iloc[0])}")

    sidebar.markdown('<div class="sidebar-group">What-if scenario</div>', unsafe_allow_html=True)
    # st.markdown rather than st.caption: caption text is a faded grey whose contrast is not verified.
    sidebar.markdown("Sliders stay within the training range: beyond it, tree models silently freeze their scores.")
    overrides: dict[str, float] = {}
    for group, names in SLIDER_GROUPS.items():
        sidebar.markdown(f"**{group}**")
        for name in names:
            spec = engine.TUNABLE_BY_NAME[name]
            lower, upper = domain.lower[name], domain.upper[name]
            dataset_value = float(baseline[name].iloc[0])
            overrides[name] = sidebar.slider(
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
    sidebar.markdown(f"**{changed} of {len(engine.TUNABLE_FEATURES)} inputs changed**")
    sidebar.button(
        "Reset to dataset values",
        on_click=_reset_sliders,
        args=(district,),
        width="stretch",
        icon=":material/restart_alt:",
        disabled=changed == 0,
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

    others, highs = rows[rows["label"] != "High"], rows[rows["label"] == "High"]
    chosen = rows[selected]
    chosen_color = theme.MAP_HIGH if scenario_prediction.label == "High" else theme.MAP_OTHER
    figure = go.Figure(
        [
            trace(others, theme.PAGE_BACKGROUND, 11, "ring", hoverable=False),  # 2px surface ring under each mark
            trace(others, theme.MAP_OTHER, 7, "Medium or Low"),
            trace(highs, theme.PAGE_BACKGROUND, 13, "ring", hoverable=False),
            trace(highs, theme.MAP_HIGH, 9, "High"),
            trace(chosen, theme.MAP_SELECTED_RING, 20, "Selected district ring", hoverable=False),
            trace(chosen, theme.PAGE_BACKGROUND, 15, "ring", hoverable=False),
            trace(chosen, chosen_color, 10, "Selected district"),
        ]
    )
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
        height=430,
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
    st.markdown(theme.header_html(district, assessment.province, model_label), unsafe_allow_html=True)
    for message in critical:
        st.error(message)

    # Scenario chips sit directly above the numbers they describe.
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
    _, csv_col, txt_col = st.columns([6, 2, 2], vertical_alignment="center")
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

    st.markdown(
        theme.section_head_html(
            "Assessment detail", "Choose a hazard: the explanation and the national map below both follow it."
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
    with left, st.container(border=True):
        render_explanation(assessment, target)
    with right, st.container(border=True):
        render_map_panel(national, assessment, target, layers)
    with st.container(border=True):
        render_profile_panel(assessment, df)

    st.markdown(theme.section_head_html("Model and data"), unsafe_allow_html=True)
    model_tab, data_tab = st.tabs(["Model card", "Dataset labels"])
    with model_tab:
        render_model_card(bundle, issues)
    with data_tab:
        st.markdown("Synthetic labels stored in the dataset (training targets), not model predictions.")
        st.dataframe(engine.dataset_labels_table(df), hide_index=True, width="stretch", height=360)


def main() -> None:
    """Entry point with a single error boundary: users never see a traceback."""
    st.set_page_config(page_title="Nigehban · Multi-Hazard Risk (simulated)", page_icon="🛡️", layout="wide")
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
