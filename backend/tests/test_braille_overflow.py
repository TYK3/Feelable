"""F03: long Braille titles wrap onto more rows; text that cannot print in full blocks readiness."""
from engine import braille, pipeline, tactile
from engine.tactile import TactileOptions
from tests.conftest import fake_analysis


def _flat(lines):
    return [c for line in lines for c in line]


def test_wrap_keeps_every_word_and_respects_width():
    text = "Plant cell structure and function of organelles"
    lines = tactile._wrap_cells(text, 14)
    assert all(len(line) <= 14 for line in lines)
    # every word's cells appear, in order, across the wrapped lines
    expected_words = [braille.to_cells(w) for w in text.split()]
    flat = _flat(lines)
    for w in expected_words:
        idx = next(i for i in range(len(flat)) if flat[i:i + len(w)] == w)
        flat = flat[idx + len(w):]


def test_long_word_is_split_not_dropped():
    lines = tactile._wrap_cells("Supercalifragilisticexpialidocious", 10)
    assert _flat(lines) == braille.to_cells("Supercalifragilisticexpialidocious")


def test_title_within_three_rows_is_not_cut_or_flagged():
    a = fake_analysis()
    a.title = "Plant cell structure and function of organelles in a leaf"
    tactile.render_tactile(a, TactileOptions(title_in_braille=True))
    assert not any(str(w).startswith(tactile.BRAILLE_OVERFLOW_PREFIX) for w in a.warnings)


def test_overlong_title_is_blocked_by_readiness():
    a = fake_analysis()
    a.title = " ".join(["overlongword"] * 60)
    tactile.render_tactile(a, TactileOptions(title_in_braille=True))
    rd = pipeline.build_readiness(a, [{"rule": "ok", "passed": True, "detail": ""}], True, None)
    assert rd["status"] == "blocked"
    assert any(i["code"] == "braille_overflow" for i in rd["issues"])


# ---------------------------------------------------------------- audit probes (11 Oct 2026)
import cv2
import numpy as np

from engine import export
from engine.types import PAGE_MARGIN_MM, PX_PER_MM

LONG_TITLE = "Human anatomy diagram showing the ascending and descending colon in high detail"
LONG_NAMES = ["ascending colon boundary measurement reference label section",
              "descending colon boundary measurement reference label region"]


def _dots(cells):
    return sum(bin(c).count("1") for c in cells)


def _blobs(ink):
    return cv2.connectedComponents((ink > 0).astype(np.uint8), connectivity=8)[0] - 1


def test_long_key_names_are_complete_in_every_key_page():
    """braille-long probe: both 63-cell rows used to be cut to 44 cells. Every raised dot the full
    names need must now be on the key pages (counted on the raster, not from the translator)."""
    a = fake_analysis()
    for p, nm in zip(a.parts, LONG_NAMES + ["chloroplast"]):
        p.name = nm
    # many more long parts so the one-page print layout cannot fit
    extra = []
    for i in range(14):
        q = a.parts[2].__class__(**{**a.parts[2].__dict__, "id": f"x{i}", "name": f"{LONG_NAMES[i % 2]} {i}"})
        extra.append(q)
    a.parts += extra
    r = tactile.render_tactile(a, TactileOptions(title_in_braille=True))
    pages = export.legend_pages(r)
    want = 0
    for e in r.legend:
        p = next(q for q in a.parts if q.id == e["part_id"])
        want += _dots(braille.to_cells(p.key)) + _dots(braille.to_cells(p.name))
    want += len(pages) * _dots(braille.to_cells("key"))
    assert sum(_blobs(pg) for pg in pages) == want
    assert not any(e.get("cut") for e in r.legend)
    assert not any(str(w).startswith(tactile.BRAILLE_OVERFLOW_PREFIX) for w in a.warnings)


def test_long_title_is_complete_on_page_one():
    """braille-long probe: an 80-cell title used to be cut to 44 cells."""
    a = fake_analysis()
    a.title = LONG_TITLE
    r = tactile.render_tactile(a, TactileOptions(title_in_braille=True))
    m = int(round(PAGE_MARGIN_MM * PX_PER_MM))
    rows = tactile._wrap_cells(LONG_TITLE, 10 ** 6)
    band_h = int(round(tactile._title_block_mm(3) * PX_PER_MM)) + 2
    band = r.ink[m - 2:m + band_h, :]
    lines = tactile._wrap_cells(LONG_TITLE, int((r.width - 2 * m) // (6.2 * PX_PER_MM)))
    assert len(lines) >= 2
    assert _blobs(band) == _dots(rows[0])


# independent UEB expectations recorded by the audit (braille-translator-check.json)
UEB_EXPECTED = {"nucleus": "⠝⠥⠉⠇⠑⠥⠎", "Nucleus": "⠠⠝⠥⠉⠇⠑⠥⠎", "DNA": "⠠⠠⠙⠝⠁", "10": "⠼⠁⠚",
                "1.5": "⠼⠁⠲⠑", "a1b": "⠁⠼⠁⠰⠃", "50%": "⠼⠑⠚⠨⠴", ".5": "⠼⠲⠑", '"cell"': "⠦⠉⠑⠇⠇⠴"}


def test_translator_matches_independent_ueb_expectations():
    for text, cells in UEB_EXPECTED.items():
        assert braille.to_unicode(text) == cells, text
