"""Render an HTML/SVG file to PNG with Playwright (Chromium).

usage: python render.py <in.html|in.svg> <out.png> <width> <height> [scale] [--opaque]
"""
import pathlib
import sys

from playwright.sync_api import sync_playwright


def render(src: str, out: str, w: int, h: int, scale: float = 1, transparent: bool = True) -> None:
    path = pathlib.Path(src).resolve()
    txt = path.read_text()
    if path.suffix == ".svg":
        txt = ("<html><head><style>html,body{margin:0;background:transparent}"
               "svg{display:block;width:100vw;height:100vh}</style></head>"
               f"<body>{txt}</body></html>")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
        pg.goto(path.as_uri()) if path.suffix == ".html" else pg.set_content(txt)
        pg.wait_for_timeout(400)
        pg.screenshot(path=out, omit_background=transparent)
        b.close()


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    render(a[0], a[1], int(a[2]), int(a[3]), float(a[4]) if len(a) > 4 else 1,
           transparent="--opaque" not in sys.argv)
