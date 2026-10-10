"""Regenerate the README product screenshots (docs/img/screen-*.png).

1. Build the web app and start the server, e.g.
     cd web && npm run build && cd ..
     cd backend && FEELABLE_DATA=/tmp/shots PORT=8205 python -m uvicorn app.main:app --port 8205
2. From the repo root:
     python docs/img/make_screens.py                  # uses http://localhost:8205
     python docs/img/make_screens.py --base https://feelable.onrender.com

Needs: playwright (Chromium) + Pillow. Each output is framed (browser window or
phone) on the cream brand background and palette-optimised to stay small.
If the touch-mode selectors or layout change, tweak TOUCH_POINT / the clicks below.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import tempfile
import urllib.request

from PIL import Image
from playwright.sync_api import sync_playwright

OUT = pathlib.Path(__file__).resolve().parent
SAMPLE = "heart"
TOUCH_POINT = (240, 470)  # a point on the left ventricle in a 390x844 viewport
CREAM, INK, BANANA = "#FFFBF0", "#17171C", "#FFD23F"


def api(base: str, path: str, method: str = "GET") -> dict:
    req = urllib.request.Request(base + path, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def optimise(path: pathlib.Path, max_kb: int = 400) -> None:
    """Palette-quantise (like pngquant) and shrink until under max_kb."""
    img = Image.open(path).convert("RGB")
    img.save(path, optimize=True)  # lossless first: keep full colour if it already fits
    if path.stat().st_size <= max_kb * 1024:
        print(f"{path.name}: {img.size} {path.stat().st_size // 1024} KB (lossless)")
        return
    for colors in (256, 192, 128):
        q = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
        q.save(path, optimize=True)
        if path.stat().st_size <= max_kb * 1024:
            break
    while path.stat().st_size > max_kb * 1024 and img.width > 900:
        img = img.resize((int(img.width * 0.85), int(img.height * 0.85)), Image.LANCZOS)
        img.quantize(colors=192, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
    print(f"{path.name}: {Image.open(path).size} {path.stat().st_size // 1024} KB")


BROWSER_FRAME = """<!doctype html><html><head><meta charset=utf-8><style>
html,body{{margin:0;background:{cream}}}
.wrap{{padding:56px 64px 64px;background:linear-gradient(160deg,#FFFBF0,#FFF1BF)}}
.win{{border-radius:18px;overflow:hidden;background:#fff;box-shadow:0 6px 0 #E9E2CF,0 30px 60px rgba(138,106,0,.18);border:1.5px solid #E9E2CF}}
.bar{{height:46px;display:flex;align-items:center;gap:9px;padding:0 18px;background:#FBF3DE;border-bottom:1.5px solid #E9E2CF}}
.d{{width:13px;height:13px;border-radius:50%;background:#E3D7B5}}
.url{{margin-left:18px;flex:1;max-width:520px;height:28px;border-radius:999px;background:#fff;border:1.5px solid #E9E2CF;
  font:600 14px Inter,sans-serif;color:#6E6C76;display:flex;align-items:center;padding:0 14px}}
img{{display:block;width:100%}}
</style></head><body><div class=wrap id=f><div class=win><div class=bar><span class=d></span><span class=d></span><span class=d></span>
<span class=url>{url}</span></div><img src="{src}"></div></div></body></html>"""

PHONE_FRAME = """<!doctype html><html><head><meta charset=utf-8><style>
html,body{{margin:0}}
.wrap{{display:flex;gap:56px;justify-content:center;align-items:center;padding:56px 72px;background:linear-gradient(160deg,#FFFBF0,#FFF1BF);width:max-content}}
.phone{{width:390px;padding:14px;border-radius:58px;background:{ink};box-shadow:0 30px 60px rgba(60,40,0,.25),inset 0 0 0 2px #3A3A44}}
.screen{{border-radius:44px;overflow:hidden;position:relative}}
.screen img{{display:block;width:390px}}
.cap{{font:600 17px Inter,sans-serif;color:#4A4852;text-align:center;margin-top:18px}}
</style></head><body><div class=wrap id=f>{phones}</div></body></html>"""


def frame(page, html: str, out: pathlib.Path, tmp: pathlib.Path) -> None:
    f = tmp / (out.stem + ".html")
    f.write_text(html)
    page.goto(f.as_uri())
    page.wait_for_timeout(400)
    page.locator("#f").screenshot(path=str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8205")
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--display-host", default="feelable.onrender.com", help="host shown in the browser frame")
    a = ap.parse_args()
    base = a.base.rstrip("/")

    job = api(base, f"/api/jobs/from-sample/{a.sample}", "POST")
    jid = job["id"]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ts-shots-"))
    for key, name in (("url", "tactile.png"), ("legend_url", "legend.png")):
        urllib.request.urlretrieve(base + job["tactile"][key], tmp / name)

    with sync_playwright() as p:
        b = p.chromium.launch()
        # desktop pages
        d = b.new_page(viewport={"width": 1440, "height": 900})
        d.goto(base + "/#/"); d.wait_for_timeout(2500)
        d.screenshot(path=str(tmp / "home.png"))
        tok = job.get("edit_token", "")
        d.goto(f"{base}/#/job/{jid}" + (f"?t={tok}" if tok else "")); d.wait_for_timeout(3500)
        d.screenshot(path=str(tmp / "review.png"))

        # phone: start screen + exploring (touching a part)
        m = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        m.goto(f"{base}/#/touch/{jid}"); m.wait_for_timeout(2500)
        m.screenshot(path=str(tmp / "touch-start.png"))
        m.mouse.click(195, 400); m.wait_for_timeout(1500)
        x, y = TOUCH_POINT
        m.mouse.move(x, y); m.mouse.down(); m.mouse.move(x + 2, y + 2); m.wait_for_timeout(800)
        m.screenshot(path=str(tmp / "touch.png"))
        m.mouse.up()

        f = b.new_page(viewport={"width": 1600, "height": 1000}, device_scale_factor=1)
        host = a.display_host
        frame(f, BROWSER_FRAME.format(cream=CREAM, url=host + "/", src=(tmp / "home.png").as_uri()),
              OUT / "screen-home.png", tmp)
        frame(f, BROWSER_FRAME.format(cream=CREAM, url=f"{host}/#/job/{jid}", src=(tmp / "review.png").as_uri()),
              OUT / "screen-review.png", tmp)
        phones = "".join(
            f'<div><div class=phone><div class=screen><img src="{(tmp / n).as_uri()}"></div></div><div class=cap>{c}</div></div>'
            for n, c in (("touch-start.png", "One tap to start"), ("touch.png", "Touch a part, hear its name"))
        )
        frame(f, PHONE_FRAME.format(ink=INK, phones=phones), OUT / "screen-touch.png", tmp)

        # tactile PDF pages, shown as two sheets of paper
        sheets = "".join(
            f'<figure><div class=sheet style="transform:rotate({r}deg)"><img src="{(tmp / n).as_uri()}"></div><figcaption>{c}</figcaption></figure>'
            for n, c, r in (("tactile.png", "Page 1 · the raised drawing (black rises on swell paper)", -1.2),
                            ("legend.png", "Page 2 · braille key", 1.0))
        )
        frame(f, f"""<!doctype html><html><head><meta charset=utf-8><style>
          html,body{{margin:0}}
          #f{{display:flex;gap:48px;padding:56px 64px 48px;background:linear-gradient(160deg,#FFFBF0,#FFF1BF);width:max-content;align-items:flex-start}}
          figure{{margin:0;width:620px}} .sheet{{background:#fff;padding:14px;border-radius:6px;box-shadow:0 4px 0 #E9E2CF,0 24px 50px rgba(138,106,0,.16)}}
          .sheet img{{display:block;width:100%}}
          figcaption{{font:600 17px Inter,sans-serif;color:#4A4852;text-align:center;margin-top:22px}}
        </style></head><body><div id=f>{sheets}</div></body></html>""", OUT / "screen-tactile.png", tmp)
        b.close()

    for n in ("home", "review", "touch", "tactile"):
        optimise(OUT / f"screen-{n}.png")


if __name__ == "__main__":
    main()
