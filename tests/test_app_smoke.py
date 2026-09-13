"""End-to-end smoke tests of the Streamlit app with Streamlit's AppTest runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import risk_engine as engine

APP = str(engine.PROJECT_ROOT / "app.py")
TIMEOUT = 240


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=TIMEOUT).run()


def _markdown(app: AppTest) -> list[str]:
    return [m.value for m in app.markdown]


def test_app_renders_without_errors(app: AppTest) -> None:
    assert not app.exception
    assert [e.value for e in app.error] == []
    header = next(m for m in _markdown(app) if 'class="app-header"' in m)
    assert 'class="app-title" role="heading" aria-level="1">Islamabad<' in header


def test_layout_order_brand_then_banner_then_district(app: AppTest) -> None:
    keys = ("brand-header", "ndma-banner", 'class="app-header"')
    order = [next(i for i, m in enumerate(_markdown(app)) if key in m) for key in keys]
    assert order == sorted(order)


def test_simulated_scenario_banner_is_always_shown(app: AppTest) -> None:
    assert any("SIMULATED SCENARIO — not an operational forecast" in m.value for m in app.markdown)


def test_three_kpi_cards_charts_and_two_downloads(app: AppTest) -> None:
    cards = next(m for m in _markdown(app) if 'class="kpi-grid"' in m)
    assert cards.count('class="kpi-card"') == 3
    assert cards.count('class="kpi-meter"') == 3
    assert len(app.get("download_button")) == 2
    assert len(app.get("plotly_chart")) == 2  # SHAP waterfall + national map


def test_hazard_selector_scopes_explanation_and_map(app: AppTest) -> None:
    selector = app.segmented_control[0]
    assert selector.value == "flood_risk"
    selector.set_value("seismic_risk").run()
    assert not app.exception
    heads = [m for m in _markdown(app) if 'class="section-head panel"' in m]
    assert any("Why the seismic score looks like this" in m for m in heads)
    assert any("Where High seismic risk is modelled" in m for m in heads)


def test_model_card_checklist_shows_every_check_passing(app: AppTest) -> None:
    checklist = next(m for m in _markdown(app) if 'class="checklist"' in m)
    assert checklist.count('<li class="pass">') == len(engine.PROVENANCE_CHECKS)
    assert '<li class="fail">' not in checklist


def test_sliders_are_bounded_to_the_training_range(app: AppTest, df, bundle) -> None:
    domain = engine.ApplicabilityDomain.fit(df, bundle.numeric_features)
    sliders = app.sidebar.slider
    assert len(sliders) == len(engine.TUNABLE_FEATURES)
    for slider, spec in zip(sliders, engine.TUNABLE_FEATURES, strict=True):
        assert slider.min == pytest.approx(domain.lower[spec.name])
        assert slider.max == pytest.approx(domain.upper[spec.name])


def test_what_if_changes_update_the_page(app: AppTest) -> None:
    app.sidebar.selectbox[0].select("Gwadar").run()
    temperature = next(s for s in app.sidebar.slider if s.label.startswith("Summer"))
    temperature.set_value(temperature.max).run()
    assert not app.exception
    chips = next(m for m in _markdown(app) if 'class="chip-row"' in m)
    assert "Simulated inputs" in chips and "Summer max temperature" in chips
    cards = next(m for m in _markdown(app) if 'class="kpi-grid"' in m)
    assert cards.count('class="kpi-chip"') == 3 and "unmodified" in cards


def test_missing_model_fails_gracefully(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NDMA_MODEL_PATH", str(tmp_path / "missing.pkl"))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception  # no traceback reaches the user
    assert any("was not found" in e.value for e in at.error)


def test_no_matplotlib_figure_is_ever_created(app: AppTest) -> None:
    import matplotlib.pyplot as plt  # test-only: proves the app created no figures (audit F3.1)

    assert plt.get_fignums() == []
