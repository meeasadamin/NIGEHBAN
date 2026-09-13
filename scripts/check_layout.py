"""Measure the running dashboard's responsive layout in a real browser.

Not part of CI (it needs a browser and a running server). It exists because a
static contrast test cannot see the CSS cascade: during the rebuild it showed
Streamlit's markdown styles shrinking the KPI level word to 16px, which broke
the large-text contrast rule for #CC6677.

Usage (Playwright is a one-off tool dependency, not a project requirement):
    streamlit run app.py --server.port 8502
    uv run --no-project --python 3.12 --with playwright==1.55.0 python scripts/check_layout.py --browser msedge

Exits 1 if any viewport shows horizontal overflow, a KPI level word below the
WCAG large-text threshold (18.66px bold), unstacked cards on phones, or missing content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

VIEWPORTS: dict[str, dict[str, object]] = {
    "phone_360x740": {
        "viewport": {"width": 360, "height": 740},
        "device_scale_factor": 2,
        "is_mobile": True,
        "has_touch": True,
    },
    "phone_390x844": {
        "viewport": {"width": 390, "height": 844},
        "device_scale_factor": 2,
        "is_mobile": True,
        "has_touch": True,
    },
    "tablet_768x1024": {
        "viewport": {"width": 768, "height": 1024},
        "device_scale_factor": 1,
        "is_mobile": True,
        "has_touch": True,
    },
    "desktop_1440x900": {"viewport": {"width": 1440, "height": 900}, "device_scale_factor": 1},
}

MEASURE_JS = """() => {
  const level = document.querySelector('.kpi-level');
  const style = level ? getComputedStyle(level) : null;
  const sidebar = document.querySelector('[data-testid="stSidebar"]');
  return {
    viewport_width: window.innerWidth,
    horizontal_overflow_px: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
    card_x: [...document.querySelectorAll('.kpi-card')].map(c => Math.round(c.getBoundingClientRect().x)),
    level_font_px: style ? parseFloat(style.fontSize) : null,
    level_font_weight: style ? parseInt(style.fontWeight, 10) : null,
    banner_present: !!document.querySelector('.ndma-banner'),
    sidebar_expanded: sidebar ? sidebar.getAttribute('aria-expanded') : null,
    plotly_charts: document.querySelectorAll('.js-plotly-plot').length,
    download_buttons: document.querySelectorAll('[data-testid="stDownloadButton"]').length,
  };
}"""


def problems_for(name: str, m: dict[str, object]) -> list[str]:
    """Return failed layout expectations for one viewport measurement."""
    found = []
    if m["horizontal_overflow_px"]:
        found.append(f"{name}: horizontal overflow of {m['horizontal_overflow_px']}px")
    if not m["level_font_px"] or m["level_font_px"] < 18.66 or (m["level_font_weight"] or 0) < 700:
        found.append(f"{name}: level word {m['level_font_px']}px/{m['level_font_weight']} is not WCAG large text")
    if len(m["card_x"]) != 3 or not m["banner_present"] or m["plotly_charts"] < 2 or m["download_buttons"] != 2:
        found.append(f"{name}: missing content {m}")
    if name.startswith("phone") and len(set(m["card_x"])) != 1:
        found.append(f"{name}: KPI cards are not stacked in one column ({m['card_x']})")
    return found


def main() -> int:
    """Measure every viewport, save screenshots, and report problems."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8502/")
    parser.add_argument("--browser", default="msedge", help="Playwright channel: msedge, chrome, or chromium.")
    parser.add_argument("--screenshots", type=Path, default=Path("layout-screenshots"))
    args = parser.parse_args()
    args.screenshots.mkdir(parents=True, exist_ok=True)

    measurements: dict[str, dict[str, object]] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel=None if args.browser == "chromium" else args.browser, headless=True
        )
        for name, options in VIEWPORTS.items():
            context = browser.new_context(**options)
            page = context.new_page()
            page.goto(args.url, wait_until="domcontentloaded")
            page.wait_for_selector(".kpi-card", timeout=120_000)
            page.wait_for_selector(".js-plotly-plot", timeout=120_000)
            page.wait_for_timeout(2_500)  # let Plotly finish its resize pass
            measurements[name] = page.evaluate(MEASURE_JS)
            page.screenshot(path=str(args.screenshots / f"{name}.png"), full_page=True)
            context.close()
        browser.close()

    print(json.dumps(measurements, indent=2))
    problems = [p for name, m in measurements.items() for p in problems_for(name, m)]
    print("\n".join(problems) if problems else "All layout checks passed.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
