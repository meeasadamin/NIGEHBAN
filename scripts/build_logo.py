"""Render the Nigehban sidebar logo PNG (emblem + wordmark in flag green) for ``st.logo``.

``st.logo`` places an image in Streamlit's own sidebar header row, level with the collapse button, and
keeps it visible at the top left when the sidebar is collapsed. It needs an image file, so the same SVG
emblem the page header uses is rendered once at 3x to ``static/nigehban-logo.png`` (committed). Re-run
only if the emblem or wordmark changes:

    uv run --no-project --python 3.12 --with playwright==1.55.0 python scripts/build_logo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ui_theme as theme  # noqa: E402  (path set above)

HEIGHT_PX = 32  # st.logo size="large" renders the image 2rem (32px) tall
SCALE = 4  # rendered at 4x so the logo stays sharp on high-density screens


def logo_html() -> str:
    """Emblem and letter-spaced wordmark on a transparent background."""
    green = theme.HEADER_STOPS[0]
    emblem = theme.emblem_svg(stroke=green).replace('class="brand-emblem"', 'width="27" height="31"')
    return (
        "<html><body style='margin:0;background:transparent'>"
        f"<div id='logo' style='display:inline-flex;align-items:center;gap:9px;height:{HEIGHT_PX}px;padding:0 2px;"
        f'font-family:"Segoe UI","Source Sans Pro",Arial,sans-serif;color:{green}\'>{emblem}'
        "<span style='font-size:19px;font-weight:700;letter-spacing:0.24em'>NIGEHBAN</span></div>"
        "</body></html>"
    )


def main() -> None:
    """Render the logo element only, with a transparent background."""
    theme.LOGO_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 400, "height": 80}, device_scale_factor=SCALE)
        page.set_content(logo_html())
        page.locator("#logo").screenshot(path=str(theme.LOGO_PATH), omit_background=True)
        browser.close()
    print(f"Wrote {theme.LOGO_PATH}")


if __name__ == "__main__":
    main()
