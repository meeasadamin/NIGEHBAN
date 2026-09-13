"""Render the Nigehban favicon PNG from the same SVG emblem the header uses.

Browsers need a raster favicon and Streamlit's ``page_icon`` accepts an image file, so the emblem is
rendered once to ``static/nigehban-favicon.png`` (committed). Re-run only if the emblem changes:

    uv run --no-project --python 3.12 --with playwright==1.55.0 python scripts/build_favicon.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ui_theme as theme  # noqa: E402  (path set above)

OUT = ROOT / "static" / "nigehban-favicon.png"
SIZE = 256


def favicon_html() -> str:
    """Emblem centred on a rounded flag-green tile; the tile keeps the white strokes visible on light tabs."""
    emblem = theme.emblem_svg().replace('class="brand-emblem"', 'width="150" height="172"')
    h0, h1 = theme.HEADER_STOPS
    return (
        "<html><body style='margin:0;background:transparent'>"
        f"<div style='width:{SIZE}px;height:{SIZE}px;border-radius:52px;display:flex;align-items:center;"
        f"justify-content:center;background:linear-gradient(135deg,{h0},{h1})'>{emblem}</div>"
        "</body></html>"
    )


def main() -> None:
    """Render the favicon with a transparent background outside the rounded tile."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": SIZE, "height": SIZE})
        page.set_content(favicon_html())
        page.screenshot(path=str(OUT), omit_background=True)
        browser.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
