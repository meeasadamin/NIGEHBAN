"""Design tokens, CSS, HTML fragments, and WCAG contrast utilities for the dashboard.

Every colour pair rendered as text or as a meaningful graphic is listed in
``contrast_requirements()`` and checked in ``tests/test_accessibility.py`` against
the *composited* surface actually drawn, not asserted (audit F5).

Assumption the checks rely on: the Streamlit theme is locked to a white page
background in ``.streamlit/config.toml``. A semi-transparent card over a solid
page blurs to the same solid colour, so compositing each gradient stop over the
page colour gives the real background behind the text.

Palette note (validated with the dataviz six-checks validator during the UI pass):
the brief's level palette passes contrast and colour-blind separation, but Medium
``#CC6677`` and High ``#CC3311`` are only ΔE 11.8 apart for *normal* vision (floor 15).
The level colours are therefore used only where the level word and shape are always
printed next to them (KPI cards). The national map, where marks sit side by side,
uses emphasis encoding instead: High in ``#CC3311`` against a neutral slate for
everything else (normal-vision ΔE 22.5, CVD ΔE 16.2).
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# --------------------------------------------------------------------------
# Tokens
# --------------------------------------------------------------------------

PAGE_BACKGROUND: Final[str] = "#FFFFFF"  # locked by .streamlit/config.toml [theme] backgroundColor
SIDEBAR_BACKGROUND: Final[str] = "#F1F5F9"  # config.toml [theme] secondaryBackgroundColor
TEXT_PRIMARY: Final[str] = "#0F172A"  # slate-900
TEXT_SECONDARY: Final[str] = "#334155"  # slate-700
TEXT_MUTED: Final[str] = "#475569"  # slate-600: smallest text still >= 4.5:1 on white
HAIRLINE: Final[str] = "#E2E8F0"  # slate-200: gridlines and dividers
CHIP_BACKGROUND: Final[str] = "#F1F5F9"
CHIP_BORDER: Final[str] = "#CBD5E1"

#: Nigehban brand band: Pakistan flag green (#01411C) deepening to a forest green. Chosen over the
#: crescent-and-star: that is a national state symbol, and this is explicitly not an official system.
HEADER_STOPS: Final[tuple[str, str]] = ("#01411C", "#0A5C36")
HEADER_TITLE: Final[str] = "#FFFFFF"  # Urdu and Latin wordmarks
HEADER_SUBTITLE: Final[str] = "#DCEFE3"  # tagline
HEADER_EYEBROW: Final[str] = "#CFE5D8"  # small caps line
EMBLEM_ACCENT: Final[str] = "#E2B93B"  # muted gold for the watchful eye in the shield emblem
URDU_FONT_URL: Final[str] = "app/static/fonts/NotoNastaliqUrdu-Variable.ttf"  # served by Streamlit static serving
PROJECT_STATIC_FONT: Final[Path] = (
    Path(__file__).resolve().parent / "static" / "fonts" / "NotoNastaliqUrdu-Variable.ttf"
)

#: Colour-blind-safe level palette (Paul Tol members), specified by the brief.
LEVEL_COLORS: Final[dict[str, str]] = {"Low": "#0077BB", "Medium": "#CC6677", "High": "#CC3311"}
#: Redundant shape per level so colour is never the only cue (WCAG 1.4.1).
LEVEL_SYMBOLS: Final[dict[str, str]] = {"Low": "●", "Medium": "◆", "High": "▲"}

#: Card gradient as (hex colour, alpha) stops. High alpha keeps the "glass" look
#: subtle while guaranteeing contrast; 0.92/0.80 are judgment values.
CARD_GRADIENT_STOPS: Final[tuple[tuple[str, float], ...]] = (("#FFFFFF", 0.92), ("#E2E8F0", 0.80))
CARD_BORDER_RGBA: Final[str] = "rgba(148, 163, 184, 0.45)"  # slate-400 hairline
CARD_SHADOW: Final[str] = "0 10px 30px rgba(15, 23, 42, 0.10), 0 1px 3px rgba(15, 23, 42, 0.08)"
CARD_RADIUS_PX: Final[int] = 16

#: Level word size. 28px bold is "large text" in WCAG 2.2 (>= 18.66px bold), so its
#: AA threshold is 3:1. #CC6677 cannot reach 4.5:1 on a light surface.
LEVEL_FONT_PX: Final[int] = 28
BODY_FONT_PX: Final[int] = 15
SMALL_FONT_PX: Final[int] = 13

#: P(High) meter: the fill is the High hue on every card (red always means High);
#: the track is a light step of the same hue, per the dataviz meter spec.
METER_FILL: Final[str] = LEVEL_COLORS["High"]
METER_TRACK: Final[str] = "#FBE3DD"

BANNER_BACKGROUND: Final[str] = "#FEF3C7"  # amber-100
BANNER_TEXT: Final[str] = "#78350F"  # amber-900
BANNER_ACCENT: Final[str] = "#B45309"  # amber-700, left rule

#: Plotly waterfall: SHAP contributions that raise vs lower the explained score.
RAISES_COLOR: Final[str] = LEVEL_COLORS["High"]
LOWERS_COLOR: Final[str] = LEVEL_COLORS["Low"]
TOTAL_COLOR: Final[str] = "#475569"  # slate-600

#: National map (emphasis encoding; see module docstring).
MAP_HIGH: Final[str] = LEVEL_COLORS["High"]
MAP_OTHER: Final[str] = "#64748B"  # slate-500
MAP_SELECTED_RING: Final[str] = TEXT_PRIMARY
MAP_FAULT: Final[str] = "#78716C"  # stone-500: context layer, recessive
MAP_COAST: Final[str] = "#94A3B8"  # slate-400: context layer, recessive

#: Layout surfaces: each page section is a soft grey band; panels inside it are white cards.
SECTION_BACKGROUND: Final[str] = "#F8FAFC"  # slate-50
PANEL_SHADOW: Final[str] = "0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04)"

CHECK_OK_COLOR: Final[str] = "#006300"  # dataviz "success text" green
CHECK_FAIL_COLOR: Final[str] = "#B91C1C"  # red-700


# --------------------------------------------------------------------------
# WCAG 2.2 contrast
# --------------------------------------------------------------------------


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` into integer channels."""
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB, got {color!r}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def composite(foreground: str, alpha: float, background: str) -> str:
    """Alpha-blend ``foreground`` over an opaque ``background``; returns ``#RRGGBB``."""
    fg, bg = hex_to_rgb(foreground), hex_to_rgb(background)
    mixed = (round(alpha * f + (1 - alpha) * b) for f, b in zip(fg, bg, strict=True))
    return "#" + "".join(f"{channel:02X}" for channel in mixed)


def relative_luminance(color: str) -> float:
    """WCAG relative luminance of an sRGB colour."""

    def linear(channel: int) -> float:
        c = channel / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in hex_to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(first: str, second: str) -> float:
    """WCAG contrast ratio between two opaque colours (1.0 to 21.0)."""
    lighter, darker = sorted((relative_luminance(first), relative_luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def card_surfaces() -> tuple[str, ...]:
    """Opaque colours actually behind card content: each gradient stop composited over the page."""
    return tuple(composite(color, alpha, PAGE_BACKGROUND) for color, alpha in CARD_GRADIENT_STOPS)


@dataclass(frozen=True, slots=True)
class ContrastRequirement:
    """One rendered colour pair and the WCAG AA threshold it must meet.

    Attributes:
        name: What is rendered.
        foreground: Text or graphic colour.
        backgrounds: Every opaque surface it is drawn on.
        minimum: 4.5 normal text, 3.0 large text (SC 1.4.3) or meaningful graphics (SC 1.4.11).
    """

    name: str
    foreground: str
    backgrounds: tuple[str, ...]
    minimum: float


def contrast_requirements() -> tuple[ContrastRequirement, ...]:
    """All colour pairs the dashboard renders, with their AA thresholds."""
    surfaces = card_surfaces()
    white = (PAGE_BACKGROUND,)
    requirements = [
        ContrastRequirement("brand wordmarks, Urdu and Latin", HEADER_TITLE, HEADER_STOPS, 4.5),
        ContrastRequirement("brand tagline (15px)", HEADER_SUBTITLE, HEADER_STOPS, 4.5),
        ContrastRequirement("brand small caps line (12-13px)", HEADER_EYEBROW, HEADER_STOPS, 4.5),
        ContrastRequirement("brand emblem eye (graphic)", EMBLEM_ACCENT, HEADER_STOPS, 3.0),
        ContrastRequirement("district bar title and meta", TEXT_PRIMARY, (PAGE_BACKGROUND,), 4.5),
        ContrastRequirement("section number badge (15px bold)", HEADER_TITLE, (HEADER_STOPS[0],), 4.5),
        ContrastRequirement("section title (22px bold)", TEXT_PRIMARY, (SECTION_BACKGROUND,), 4.5),
        ContrastRequirement("section subtitle (14px)", TEXT_MUTED, (SECTION_BACKGROUND,), 4.5),
        ContrastRequirement("sidebar card labels (12px)", TEXT_SECONDARY, (PAGE_BACKGROUND,), 4.5),
        ContrastRequirement("banner text (15-16px)", BANNER_TEXT, (BANNER_BACKGROUND,), 4.5),
        ContrastRequirement("banner accent rule (graphic)", BANNER_ACCENT, (BANNER_BACKGROUND,), 3.0),
        ContrastRequirement("scenario chip text (13px)", TEXT_PRIMARY, (CHIP_BACKGROUND,), 4.5),
        ContrastRequirement("sidebar text (14-16px)", TEXT_PRIMARY, (SIDEBAR_BACKGROUND,), 4.5),
        ContrastRequirement("card hazard name (13px)", TEXT_SECONDARY, surfaces, 4.5),
        ContrastRequirement("card detail text (15px)", TEXT_SECONDARY, surfaces, 4.5),
        ContrastRequirement("card change chip text (13px)", TEXT_PRIMARY, white, 4.5),
        ContrastRequirement("P(High) meter fill vs track (graphic)", METER_FILL, (METER_TRACK,), 3.0),
        ContrastRequirement("section subtitles and captions (14px)", TEXT_MUTED, white, 4.5),
        ContrastRequirement("waterfall labels, ticks, values (13px)", TEXT_SECONDARY, white, 4.5),
        ContrastRequirement("waterfall 'raises' bars (graphic)", RAISES_COLOR, white, 3.0),
        ContrastRequirement("waterfall 'lowers' bars (graphic)", LOWERS_COLOR, white, 3.0),
        ContrastRequirement("waterfall total bars (graphic)", TOTAL_COLOR, white, 3.0),
        ContrastRequirement("map High markers (graphic)", MAP_HIGH, white, 3.0),
        ContrastRequirement("map other markers (graphic)", MAP_OTHER, white, 3.0),
        ContrastRequirement("map selected-district ring (graphic)", MAP_SELECTED_RING, white, 3.0),
        ContrastRequirement("checklist pass mark (graphic + word)", CHECK_OK_COLOR, white, 3.0),
        ContrastRequirement("checklist fail mark (graphic + word)", CHECK_FAIL_COLOR, white, 3.0),
    ]
    for level, color in LEVEL_COLORS.items():
        requirements.append(
            ContrastRequirement(f"{level} level word ({LEVEL_FONT_PX}px bold = large text)", color, surfaces, 3.0)
        )
        requirements.append(ContrastRequirement(f"{level} card accent border (graphic)", color, surfaces, 3.0))
    return tuple(requirements)


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------


def _rgba(color: str, alpha: float) -> str:
    r, g, b = hex_to_rgb(color)
    return f"rgba({r}, {g}, {b}, {alpha})"


def page_css() -> str:
    """Return the ``<style>`` block injected once per page."""
    (c0, a0), (c1, a1) = CARD_GRADIENT_STOPS
    h0, h1 = HEADER_STOPS
    return f"""
<style>
/* Streamlit's fixed toolbar is 3.75rem tall; 4.25rem keeps the header band fully visible (measured in Edge). */
[data-testid="stMainBlockContainer"] {{ padding-top: 4.25rem; padding-bottom: 3rem; max-width: 1320px; }}
@font-face {{
  font-family: "Noto Nastaliq Urdu"; src: url("{URDU_FONT_URL}") format("truetype");
  font-weight: 400 700; font-display: swap;
}}
.brand-header {{
  background: linear-gradient(135deg, {h0} 0%, {h1} 100%); border-radius: 16px;
  padding: 0.6rem 1.25rem 1rem 1.25rem; margin: 0 0 0.75rem 0; text-align: center;
  display: flex; flex-direction: column; align-items: center;
}}
.brand-header .brand-row {{ display: flex; align-items: center; justify-content: center; gap: 0.9rem; flex-wrap: wrap; }}
.brand-header .brand-emblem {{ width: 54px; height: 62px; flex: none; }}
/* Nastaliq needs generous line-height (2.5+) or its stacked letterforms clip. */
.brand-header .brand-urdu {{
  font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", "Urdu Typesetting", serif;
  color: {HEADER_TITLE}; font-size: 46px; line-height: 2.6; margin: 0; padding: 0 0.2rem;
}}
.brand-header .brand-latin {{ color: {HEADER_TITLE}; font-size: 22px; font-weight: 700; letter-spacing: 0.42em;
  margin: -0.35rem 0 0.1rem 0.42em; }}
.brand-header .brand-tagline {{ color: {HEADER_SUBTITLE}; font-size: {BODY_FONT_PX}px; margin: 0.15rem 0 0 0; }}
.brand-header .brand-note {{ color: {HEADER_EYEBROW}; font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase;
  margin-top: 0.25rem; }}
@media (max-width: 640px) {{
  .brand-header .brand-urdu {{ font-size: 36px; }}
  .brand-header .brand-latin {{ font-size: 18px; letter-spacing: 0.3em; }}
  .brand-header .brand-emblem {{ width: 44px; height: 50px; }}
}}
.app-header {{
  background: {PAGE_BACKGROUND}; border: 1px solid {HAIRLINE}; border-left: 6px solid {h0}; border-radius: 12px;
  padding: 0.7rem 1.1rem; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline;
  gap: 0.25rem 1.5rem; margin: 0 0 0.9rem 0;
}}
.app-header .app-title {{ color: {TEXT_PRIMARY}; font-size: 28px; font-weight: 700; line-height: 1.2; margin: 0; }}
.app-header .app-subtitle {{ color: {TEXT_SECONDARY}; font-size: {BODY_FONT_PX}px; }}
.app-header .app-meta {{ color: {TEXT_MUTED}; font-size: {SMALL_FONT_PX}px; text-align: right; line-height: 1.5; }}
@media (max-width: 640px) {{
  .app-header .app-title {{ font-size: 24px; }}
  .app-header .app-meta {{ text-align: left; }}
}}
.ndma-banner {{
  background: {BANNER_BACKGROUND}; color: {BANNER_TEXT}; border-left: 6px solid {BANNER_ACCENT};
  border-radius: 10px; padding: 0.6rem 1rem; margin: 0 0 1rem 0; font-size: {BODY_FONT_PX}px; line-height: 1.45;
}}
.ndma-banner strong {{ letter-spacing: 0.02em; }}
.chip-row {{ display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; min-height: 2.5rem; }}
.chip-row .chip-label {{ color: {TEXT_MUTED}; font-size: {SMALL_FONT_PX}px; font-weight: 600; margin-right: 0.2rem; }}
.chip-row .chip {{ background: {CHIP_BACKGROUND}; border: 1px solid {CHIP_BORDER}; color: {TEXT_PRIMARY};
  border-radius: 999px; padding: 0.15rem 0.65rem; font-size: {SMALL_FONT_PX}px; }}
.section-head {{ margin: 1.25rem 0 0.35rem 0; }}
.section-head .section-title {{ color: {TEXT_PRIMARY}; font-size: 20px; font-weight: 700; }}
.section-head .section-sub {{ color: {TEXT_MUTED}; font-size: 14px; }}
.section-head.panel {{ margin: 0.1rem 0 0.6rem 0; }}
.section-head.panel .section-title {{ font-size: 17px; }}
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; margin: 0.25rem 0 0.5rem 0; }}
.kpi-card {{
  background: linear-gradient(135deg, {_rgba(c0, a0)} 0%, {_rgba(c1, a1)} 100%);
  -webkit-backdrop-filter: blur(10px); backdrop-filter: blur(10px);
  border: 1px solid {CARD_BORDER_RGBA}; border-left: 6px solid var(--level);
  border-radius: {CARD_RADIUS_PX}px; box-shadow: {CARD_SHADOW}; padding: 0.95rem 1.1rem 1rem 1.1rem;
}}
/* <div> (not <p>) plus three-class selectors: Streamlit's markdown styles set p font-size, which
   silently shrank the level word to 16px (measured in Edge) and broke the large-text contrast rule. */
.kpi-grid .kpi-card .kpi-top {{ display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; }}
.kpi-grid .kpi-card .kpi-hazard {{ color: {TEXT_SECONDARY}; font-size: {SMALL_FONT_PX}px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase; }}
.kpi-grid .kpi-card .kpi-chip {{ background: {PAGE_BACKGROUND}; border: 1px solid {CHIP_BORDER}; color: {TEXT_PRIMARY};
  border-radius: 999px; padding: 0.05rem 0.55rem; font-size: {SMALL_FONT_PX}px; font-weight: 600; white-space: nowrap; }}
.kpi-grid .kpi-card .kpi-level {{ color: var(--level); font-size: {LEVEL_FONT_PX}px; font-weight: 700; line-height: 1.2;
  margin: 0.2rem 0 0.45rem 0; }}
.kpi-grid .kpi-card .kpi-meter {{ height: 8px; border-radius: 999px; background: {METER_TRACK}; overflow: hidden; position: relative; }}
.kpi-grid .kpi-card .kpi-meter > i {{ position: absolute; top: 0; bottom: 0; width: 2px; background: {TEXT_PRIMARY}; }}
.kpi-grid .kpi-card .kpi-rule {{ font-size: {SMALL_FONT_PX}px; margin-top: 0.1rem; }}
.kpi-grid .kpi-card .kpi-meter > span {{ display: block; height: 100%; background: {METER_FILL}; border-radius: 999px; }}
.kpi-grid .kpi-card .kpi-detail {{ color: {TEXT_SECONDARY}; font-size: {BODY_FONT_PX}px; line-height: 1.45; margin: 0.45rem 0 0 0; }}
.kpi-grid .kpi-card .kpi-detail strong {{ color: {TEXT_PRIMARY}; }}
.map-legend {{ display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; color: {TEXT_SECONDARY}; font-size: {SMALL_FONT_PX}px; margin: 0.35rem 0 0 0; }}
.map-legend .key {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
.map-legend .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
.map-legend .ring {{ width: 12px; height: 12px; border-radius: 50%; border: 3px solid {MAP_SELECTED_RING}; display: inline-block; box-sizing: border-box; }}
.map-legend .line {{ width: 18px; height: 2px; display: inline-block; }}
.checklist {{ list-style: none; padding: 0; margin: 0.25rem 0 0.75rem 0; }}
.checklist li {{ display: flex; gap: 0.6rem; padding: 0.55rem 0; border-bottom: 1px solid {HAIRLINE}; }}
.checklist .mark {{ font-size: 18px; font-weight: 700; line-height: 1.3; width: 1.2rem; }}
.checklist .pass .mark {{ color: {CHECK_OK_COLOR}; }}
.checklist .fail .mark {{ color: {CHECK_FAIL_COLOR}; }}
.checklist .check-label {{ color: {TEXT_PRIMARY}; font-weight: 600; font-size: {BODY_FONT_PX}px; }}
.checklist .check-state {{ color: {TEXT_MUTED}; font-weight: 600; font-size: {SMALL_FONT_PX}px; margin-left: 0.4rem; }}
.checklist .check-detail {{ color: {TEXT_MUTED}; font-size: 14px; overflow-wrap: anywhere; }}

/* ---- Page sections: numbered grey bands so headings never run into data ---- */
[class*="st-key-section_"] {{
  background: {SECTION_BACKGROUND}; border: 1px solid {HAIRLINE}; border-radius: 18px;
  padding: 1.1rem 1.25rem 1.3rem 1.25rem; margin-top: 1.4rem; gap: 0.9rem;
}}
.page-section-head {{
  display: flex; align-items: flex-start; gap: 0.85rem; padding-bottom: 0.85rem;
  border-bottom: 1px solid {HAIRLINE}; margin-bottom: 0.1rem;
}}
.page-section-head .num {{
  flex: none; width: 2.1rem; height: 2.1rem; border-radius: 10px; background: {h0}; color: {HEADER_TITLE};
  font-size: 15px; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-top: 0.1rem;
}}
.page-section-head .title {{ color: {TEXT_PRIMARY}; font-size: 22px; font-weight: 700; line-height: 1.25; }}
.page-section-head .sub {{ color: {TEXT_MUTED}; font-size: 14px; line-height: 1.45; margin-top: 0.15rem; }}
@media (max-width: 640px) {{
  [class*="st-key-section_"] {{ padding: 0.9rem 0.8rem 1rem 0.8rem; border-radius: 14px; }}
  .page-section-head .title {{ font-size: 19px; }}
}}
/* Panels inside a section: white cards. */
[class*="st-key-panel_"] {{
  background: {PAGE_BACKGROUND}; border: 1px solid {HAIRLINE}; border-radius: 14px;
  box-shadow: {PANEL_SHADOW}; padding: 1rem 1.1rem;
}}
@media (max-width: 640px) {{ [class*="st-key-panel_"] {{ padding: 0.8rem 0.75rem; }} }}
[class*="st-key-section_"] .app-header {{ margin-bottom: 0; }}
[class*="st-key-section_"] [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 0.25rem; }}
/* Phones: Streamlit collapses the sidebar, so say where the controls are. Hidden on wider screens. */
.phone-hint {{ display: none; background: {CHIP_BACKGROUND}; border: 1px solid {CHIP_BORDER}; color: {TEXT_PRIMARY};
  border-radius: 10px; padding: 0.5rem 0.75rem; font-size: 14px; line-height: 1.45; }}
@media (max-width: 640px) {{ .phone-hint {{ display: block; }} }}
.toolbar-note {{ color: {TEXT_MUTED}; font-size: {SMALL_FONT_PX}px; padding-top: 0.55rem; }}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {{ background: {SECTION_BACKGROUND}; border-right: 1px solid {HAIRLINE}; }}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 0.5rem; }}
.sb-brand {{
  display: flex; align-items: center; gap: 0.7rem; background: linear-gradient(135deg, {h0} 0%, {h1} 100%);
  border-radius: 14px; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem;
}}
.sb-brand .brand-emblem {{ width: 30px; height: 35px; flex: none; }}
.sb-brand .sb-name {{ color: {HEADER_TITLE}; font-size: 17px; font-weight: 700; letter-spacing: 0.2em; line-height: 1.2; }}
.sb-brand .sb-sub {{ color: {HEADER_TITLE}; font-size: 12px; opacity: 1; letter-spacing: 0.04em; }}
[data-testid="stSidebar"] [class*="st-key-sb_"] {{
  background: {PAGE_BACKGROUND}; border: 1px solid {HAIRLINE}; border-radius: 14px;
  box-shadow: {PANEL_SHADOW}; padding: 0.8rem 0.9rem 0.9rem 0.9rem; gap: 0.55rem;
}}
.sb-card-head {{ display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;
  padding-bottom: 0.45rem; margin-bottom: 0.35rem; border-bottom: 1px solid {HAIRLINE}; }}
.sb-card-head .sb-label {{ color: {TEXT_SECONDARY}; font-size: {SMALL_FONT_PX}px; font-weight: 700;
  letter-spacing: 0.06em; text-transform: uppercase; }}
.sb-card-head .sb-count {{ background: {CHIP_BACKGROUND}; border: 1px solid {CHIP_BORDER}; color: {TEXT_PRIMARY};
  border-radius: 999px; padding: 0.02rem 0.55rem; font-size: 12px; font-weight: 600; white-space: nowrap; }}
.sb-meta {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
.sb-meta span {{ background: {CHIP_BACKGROUND}; border: 1px solid {CHIP_BORDER}; color: {TEXT_PRIMARY};
  border-radius: 999px; padding: 0.05rem 0.55rem; font-size: 12px; }}
.sb-foot {{ color: {TEXT_MUTED}; font-size: 12px; line-height: 1.5; padding: 0.2rem 0.2rem 0 0.2rem; }}
</style>
"""


# --------------------------------------------------------------------------
# HTML fragments (all dynamic text is escaped)
# --------------------------------------------------------------------------


def emblem_svg() -> str:
    """Nigehban emblem: a shield (protection) holding a watchful eye (the watchman), drawn inline.

    Pure SVG so it renders identically on any deployment with no image asset to host.
    """
    white, gold = HEADER_TITLE, EMBLEM_ACCENT
    return (
        '<svg class="brand-emblem" viewBox="0 0 54 62" role="img" aria-label="Nigehban emblem: shield with a watchful eye">'
        f'<path d="M27 3 L50 11 V29 C50 44 40 54 27 59 C14 54 4 44 4 29 V11 Z" fill="none" stroke="{white}" '
        'stroke-width="3" stroke-linejoin="round"/>'
        f'<path d="M11 31 C17 21 37 21 43 31 C37 41 17 41 11 31 Z" fill="none" stroke="{white}" stroke-width="2.6" '
        'stroke-linejoin="round"/>'
        f'<circle cx="27" cy="31" r="5.6" fill="{gold}"/>'
        f'<circle cx="27" cy="31" r="2.2" fill="{HEADER_STOPS[0]}"/>'
        "</svg>"
    )


def brand_header_html() -> str:
    """Centered Nigehban header: emblem, Nastaliq Urdu wordmark (RTL), Latin wordmark, tagline.

    The Urdu wordmark sits in its own ``dir="rtl" lang="ur"`` element so it never inherits LTR layout.
    """
    return (
        '<div class="brand-header" role="banner">'
        '<div class="brand-row">'
        f"{emblem_svg()}"
        '<div class="brand-urdu" dir="rtl" lang="ur">نگہبان</div>'
        "</div>"
        '<div class="brand-latin" lang="en" role="heading" aria-level="1">NIGEHBAN</div>'
        '<div class="brand-tagline">Multi-hazard risk intelligence for Pakistan — flood, heatwave, seismic</div>'
        '<div class="brand-note">Independent portfolio project · not an official NDMA or PDMA system</div>'
        "</div>"
    )


def header_html(district: str, province: str, model_label: str) -> str:
    """District bar under the brand header: district as the page title, province, and model provenance."""
    return (
        '<div class="app-header">'
        "<div>"
        # Level 3: page h1 is the Nigehban wordmark, and this bar sits inside the level-2 "Risk overview" section.
        f'<div class="app-title" role="heading" aria-level="3">{html.escape(district)}</div>'
        f'<div class="app-subtitle">{html.escape(province)} · Flood, heatwave and seismic risk assessment</div>'
        "</div>"
        f'<div class="app-meta">{html.escape(model_label)}</div>'
        "</div>"
    )


def banner_html() -> str:
    """Persistent simulated-scenario banner (new feature request)."""
    return (
        '<div class="ndma-banner" role="status">'
        "<strong>SIMULATED SCENARIO — not an operational forecast.</strong> "
        "Synthetic climate and exposure data; real district locations and fault geometry. "
        "Not an official NDMA system."
        "</div>"
    )


def chips_html(label: str, chips: Sequence[str]) -> str:
    """A labelled row of neutral chips (text never wears a data colour)."""
    items = "".join(f'<span class="chip">{html.escape(chip)}</span>' for chip in chips)
    return f'<div class="chip-row"><span class="chip-label">{html.escape(label)}</span>{items}</div>'


def section_head_html(title: str, subtitle: str = "", panel: bool = False) -> str:
    """Section title with an optional one-line subtitle; ``panel`` uses the compact size inside cards."""
    sub = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    css = "section-head panel" if panel else "section-head"
    return f'<div class="{css}"><div class="section-title">{html.escape(title)}</div>{sub}</div>'


def page_section_head_html(number: int, title: str, subtitle: str = "") -> str:
    """Numbered top-level section heading (used as the first element of a ``section_*`` container)."""
    sub = f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ""
    return (
        '<div class="page-section-head">'
        f'<div class="num" aria-hidden="true">{number:02d}</div>'
        f'<div><div class="title" role="heading" aria-level="2">{html.escape(title)}</div>{sub}</div>'
        "</div>"
    )


def phone_hint_html() -> str:
    """Phone-only note pointing to the collapsed sidebar (display is controlled by a CSS media query)."""
    return (
        '<div class="phone-hint" role="note"><strong>To change the district or inputs,</strong> '
        "tap the <strong>»</strong> button at the top left to open the controls.</div>"
    )


def sidebar_brand_html() -> str:
    """Compact brand block at the top of the sidebar."""
    return (
        '<div class="sb-brand">'
        f"{emblem_svg()}"
        '<div><div class="sb-name">NIGEHBAN</div><div class="sb-sub">Scenario controls</div></div>'
        "</div>"
    )


def sidebar_card_head_html(label: str, count: str = "") -> str:
    """Heading row of a sidebar card: an uppercase label and an optional count chip."""
    badge = f'<span class="sb-count">{html.escape(count)}</span>' if count else ""
    return f'<div class="sb-card-head"><span class="sb-label">{html.escape(label)}</span>{badge}</div>'


def sidebar_meta_html(items: Sequence[str]) -> str:
    """A row of small neutral chips (province, identifiers) under the district picker."""
    return '<div class="sb-meta">' + "".join(f"<span>{html.escape(i)}</span>" for i in items) + "</div>"


def kpi_card_html(
    hazard: str, level: str, p_high: float, baseline_p_high: float | None, high_threshold: float | None = None
) -> str:
    """Render one KPI stat tile.

    Args:
        hazard: Hazard display name.
        level: ``Low``/``Medium``/``High``.
        p_high: Calibrated probability of High, 0-1.
        baseline_p_high: P(High) for the unmodified inputs, or None when the inputs are unmodified.
        high_threshold: P(High) at which this hazard is flagged High; drawn as a tick on the meter so a
            "High" label with P(High) below 50% is explained on the card itself.
    """
    if level not in LEVEL_COLORS:
        raise ValueError(f"Unknown level {level!r}")
    width = f"{max(0.0, min(1.0, p_high)) * 100:.1f}%"
    chip = detail_extra = tick = ""
    if baseline_p_high is not None:
        change = (p_high - baseline_p_high) * 100
        arrow = "▲" if change > 0.05 else "▼" if change < -0.05 else "■"
        chip = f'<span class="kpi-chip">{arrow} {change:+.1f} pts</span>'
        detail_extra = f" · unmodified {baseline_p_high:.1%}"
    threshold_note = ""
    if high_threshold is not None:
        tick = f'<i style="left: {high_threshold * 100:.1f}%"></i>'
        threshold_note = f'<div class="kpi-detail kpi-rule">Flagged High at P(High) ≥ {high_threshold:.0%}</div>'
    return (
        f'<div class="kpi-card" style="--level: {LEVEL_COLORS[level]}" role="group" '
        f'aria-label="{html.escape(hazard)} risk: {html.escape(level)}, probability of High {p_high:.1%}">'
        f'<div class="kpi-top"><span class="kpi-hazard">{html.escape(hazard)} risk</span>{chip}</div>'
        f'<div class="kpi-level"><span aria-hidden="true">{LEVEL_SYMBOLS[level]}</span> {html.escape(level)}</div>'
        f'<div class="kpi-meter" aria-hidden="true"><span style="width: {width}"></span>{tick}</div>'
        f'<div class="kpi-detail">P(High) <strong>{p_high:.1%}</strong>{html.escape(detail_extra)}</div>'
        f"{threshold_note}"
        "</div>"
    )


def planning_label_html(title: str, body: str) -> str:
    """Label shown on each planning tool: hypothetical what-if analysis, never a forecast."""
    return (
        '<div class="ndma-banner" role="note">'
        f"<strong>HYPOTHETICAL SCENARIO — {html.escape(title)}.</strong> {html.escape(body)}"
        "</div>"
    )


def map_legend_html(
    hazard: str, emphasized_label: str | None = None, other_label: str = "Medium or Low", selected: bool = True
) -> str:
    """Legend for the national map (always present: the map has more than one mark type)."""
    emphasized_label = emphasized_label or f"High {hazard.lower()} risk"
    ring = '<span class="key"><span class="ring"></span>Selected district</span>' if selected else ""
    return (
        '<div class="map-legend">'
        f'<span class="key"><span class="dot" style="background: {MAP_HIGH}"></span>{html.escape(emphasized_label)}</span>'
        f'<span class="key"><span class="dot" style="background: {MAP_OTHER}"></span>{html.escape(other_label)}</span>'
        f"{ring}"
        f'<span class="key"><span class="line" style="background: {MAP_FAULT}"></span>Active fault (GEM)</span>'
        f'<span class="key"><span class="line" style="background: {MAP_COAST}"></span>Coastline</span>'
        "</div>"
    )


def checklist_html(items: Sequence[tuple[str, bool, str]]) -> str:
    """Provenance checklist: mark + word + evidence, so state never relies on colour."""
    rows = []
    for label, passed, detail in items:
        state = "pass" if passed else "fail"
        rows.append(
            f'<li class="{state}"><span class="mark" aria-hidden="true">{"✓" if passed else "✕"}</span><div>'
            f'<div class="check-label">{html.escape(label)}<span class="check-state">{"Pass" if passed else "Fail"}</span></div>'
            f'<div class="check-detail">{html.escape(detail)}</div></div></li>'
        )
    return f'<ul class="checklist">{"".join(rows)}</ul>'
