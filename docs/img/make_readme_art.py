#!/usr/bin/env python3
"""README art: for each featured lesson, the original picture beside Feelable's tactile page.
    python3 docs/img/make_readme_art.py CACHE_DIR   (a prepared sample cache, e.g. FEELABLE_SAMPLE_CACHE/a4)"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "img", "readme")
W, H, PAD = 760, 300, 14
INK, BG, YEL = (23, 23, 28), (255, 255, 255), (255, 210, 63)

def fit(im, w, h):
    im = im.convert("RGB"); im.thumbnail((w, h), Image.LANCZOS)
    c = Image.new("RGB", (w, h), BG); c.paste(im, ((w - im.width) // 2, (h - im.height) // 2)); return c

def font(sz):
    for p in ("/usr/share/fonts/opentype/inter/Inter-SemiBold.otf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p): return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

def card(cache, sid):
    d = os.path.join(cache, sid)
    a, b = Image.open(os.path.join(d, "original.png")), Image.open(os.path.join(d, "tactile.png"))
    half = (W - 3 * PAD) // 2
    c = Image.new("RGB", (W, H), (246, 244, 236))
    c.paste(fit(a, half, H - 2 * PAD - 26), (PAD, PAD + 26)); c.paste(fit(b, half, H - 2 * PAD - 26), (2 * PAD + half, PAD + 26))
    g = ImageDraw.Draw(c); f = font(17)
    g.text((PAD, PAD - 2), "Original", fill=INK, font=f); g.text((2 * PAD + half, PAD - 2), "Feelable tactile page", fill=INK, font=f)
    g.polygon([(W // 2 + 9, H // 2), (W // 2 - 6, H // 2 - 13), (W // 2 - 6, H // 2 + 13)], fill=YEL)
    g.rectangle((0, 0, W - 1, H - 1), outline=(220, 216, 200), width=2)
    c.save(os.path.join(OUT, f"{sid}.jpg"), quality=84, optimize=True)

if __name__ == "__main__":
    cache = sys.argv[1]
    m = json.load(open(os.path.join(ROOT, "samples", "manifest.json")))
    m = m if isinstance(m, list) else m.get("samples", m)
    for s in m:
        if s.get("featured"): card(cache, s["id"]); print("wrote", s["id"])
