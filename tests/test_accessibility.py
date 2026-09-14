"""WCAG 2.2 AA contrast and colour-independence checks for every rendered colour pair (audit F5)."""

from __future__ import annotations

import pytest

import ui_theme as theme


@pytest.mark.parametrize("requirement", theme.contrast_requirements(), ids=lambda r: r.name)
def test_contrast_meets_wcag_aa(requirement: theme.ContrastRequirement) -> None:
    worst = min(theme.contrast_ratio(requirement.foreground, bg) for bg in requirement.backgrounds)
    assert worst >= requirement.minimum, f"{requirement.name}: {worst:.2f}:1 < {requirement.minimum}:1"


def test_palette_is_the_specified_colour_blind_safe_set() -> None:
    assert theme.LEVEL_COLORS == {"Low": "#0077BB", "Medium": "#CC6677", "High": "#CC3311"}


def test_levels_never_rely_on_colour_alone() -> None:
    assert len(set(theme.LEVEL_SYMBOLS.values())) == 3  # a distinct shape per level (SC 1.4.1)
    html = theme.kpi_card_html("Flood", "Medium", 0.40, 0.35)
    assert "◆" in html and "Medium" in html and "▲ +5.0 pts" in html and "unmodified 35.0%" in html


def test_map_uses_emphasis_encoding_not_the_level_palette() -> None:
    # Medium and High level colours are too close for normal vision when marks sit side by side,
    # so the map encodes High vs everything else, and the legend always names both.
    assert theme.LEVEL_COLORS["High"] == theme.MAP_HIGH
    assert theme.MAP_OTHER not in theme.LEVEL_COLORS.values()
    legend = theme.map_legend_html("Flood")
    assert "High flood risk" in legend and "Medium or Low" in legend and "Selected district" in legend


def test_checklist_state_is_not_colour_only() -> None:
    html = theme.checklist_html([("Integrity", True, "ok"), ("Data", False, "mismatch")])
    assert "✓" in html and "Pass" in html and "✕" in html and "Fail" in html


def test_medium_colour_is_only_used_as_large_text() -> None:
    # #CC6677 fails 4.5:1 on the card surfaces, so the level word must stay "large" (>= 18.66px bold).
    assert min(theme.contrast_ratio(theme.LEVEL_COLORS["Medium"], s) for s in theme.card_surfaces()) < 4.5
    assert theme.LEVEL_FONT_PX >= 19
    assert "font-weight: 700" in theme.page_css()


def test_card_html_escapes_dynamic_text() -> None:
    html = theme.kpi_card_html("<script>alert(1)</script>", "High", 0.5, None)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_gauge_draws_value_and_high_zone_from_the_threshold() -> None:
    card = theme.kpi_card_html("Flood", "High", 0.62, None, high_threshold=0.40)
    assert 'stroke-dasharray="62.00 100"' in card  # value arc covers 62% of the dial
    assert 'stroke-dasharray="0 40.00 60.00 100"' in card  # red High zone starts at the threshold
    assert "--angle: 111.6deg" in card  # needle: 0.62 x 180 degrees
    assert "62.0%" in card and "High zone from <strong>40%</strong>" in card and "▲" in card


def test_tables_escape_plain_text_and_keep_trusted_markup() -> None:
    table = theme.data_table_html(
        [theme.Column("A"), theme.Column("B")],
        [["<img src=x onerror=alert(1)>", theme.level_pill_html("Low")]],
        caption="<b>cap</b>",
    )
    assert "<img" not in table and "&lt;img" in table and "&lt;b&gt;cap" in table
    assert 'class="pill"' in table and "●" in table
    assert '<th scope="col">A</th>' in table


def test_contrast_ratio_reference_values() -> None:
    assert theme.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert theme.contrast_ratio("#FFFFFF", "#FFFFFF") == pytest.approx(1.0)
    assert theme.composite("#000000", 0.5, "#FFFFFF") == "#808080"


def test_nigehban_header_uses_rtl_nastaliq_and_no_state_symbols() -> None:
    header = theme.brand_header_html()
    assert '<div class="brand-urdu" dir="rtl" lang="ur">نگہبان</div>' in header
    assert "NIGEHBAN" in header and "<svg" in header and "not an official NDMA" in header
    css = theme.page_css()
    assert "Noto Nastaliq Urdu" in css and "line-height: 2.6" in css
    assert theme.PROJECT_STATIC_FONT.is_file()
    assert "crescent" not in header.lower()
