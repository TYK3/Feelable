"""Palette-optimise the rendered README art (run after re-rendering with render.py)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from make_screens import optimise  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
for rel, kb in (("docs/img/banner.png", 400), ("docs/img/banner@2x.png", 700),
                ("docs/img/how-it-works.png", 400), ("web/public/og-image.png", 300)):
    optimise(ROOT / rel, kb)
