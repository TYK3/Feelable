"""Unified English Braille, grade 1 (uncontracted), English + raster drawing of braille cells.

Supported: a-z with capital indicators, digits (numeric indicator), . , ? ! ' - : ; ( ) / % + = & " – — °.
Anything else (accents, other scripts, maths beyond + = /, emoji) is left out and reported by
unsupported_chars(); the tactile renderer turns that into a warning.

OWNER: Tactile team.

Cells are ints: bit i set = dot (i+1) raised.  Dot layout inside a cell:
    1 4
    2 5
    3 6
"""
from __future__ import annotations

import cv2
import numpy as np

from engine.types import (
    BRAILLE_CELL_SPACING_MM,
    BRAILLE_DOT_DIAMETER_MM,
    BRAILLE_DOT_SPACING_MM,
)


def _cell(dots: str) -> int:
    v = 0
    for d in dots:
        v |= 1 << (int(d) - 1)
    return v


LETTERS: dict[str, int] = {
    ch: _cell(d)
    for ch, d in {
        "a": "1", "b": "12", "c": "14", "d": "145", "e": "15", "f": "124", "g": "1245",
        "h": "125", "i": "24", "j": "245", "k": "13", "l": "123", "m": "134", "n": "1345",
        "o": "135", "p": "1234", "q": "12345", "r": "1235", "s": "234", "t": "2345",
        "u": "136", "v": "1236", "w": "2456", "x": "1346", "y": "13456", "z": "1356",
    }.items()
}
# digits use the a-j patterns after the numeric indicator
DIGITS: dict[str, int] = {d: LETTERS["jabcdefghi"[int(d)]] for d in "0123456789"}
NUMERIC_INDICATOR = _cell("3456")   # ⠼
GRADE1_INDICATOR = _cell("56")      # ⠰
PUNCT: dict[str, list[int]] = {
    ".": [_cell("256")],
    ",": [_cell("2")],
    "?": [_cell("236")],
    "!": [_cell("235")],
    "'": [_cell("3")],
    "-": [_cell("36")],
    ":": [_cell("25")],
    ";": [_cell("23")],
    "(": [_cell("5"), _cell("126")],
    ")": [_cell("5"), _cell("345")],
    "’": [_cell("3")],
    "/": [_cell("456"), _cell("34")],
    "%": [_cell("46"), _cell("356")],
    "+": [_cell("5"), _cell("235")],
    "=": [_cell("5"), _cell("2356")],
    "&": [_cell("4"), _cell("12346")],
    '"': [_cell("6"), _cell("2356")],
    "–": [_cell("6"), _cell("36")],     # en dash
    "—": [_cell("6"), _cell("36")],     # em dash
    "°": [_cell("45"), _cell("245")],
}
CAPITAL_INDICATOR = _cell("6")      # ⠠ (doubled: the whole word is in capitals)
# dot index (0..5) -> (column, row)
_DOT_POS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

BRAILLE_CODE = "UEB grade 1 (uncontracted), English"


def _supported(ch: str) -> bool:
    return ch.isspace() or (ch.isascii() and ch.lower() in LETTERS) or ch in DIGITS or ch in PUNCT


def unsupported_chars(text: str) -> set[str]:
    """Characters of `text` with no braille in our code (BRAILLE_CODE); to_cells leaves them out.
    E.g. accented letters, non-Latin scripts, most maths/currency symbols, emoji."""
    return {ch for ch in (text or "") if not _supported(ch)}


def to_cells(text: str) -> list[int]:
    """UEB grade-1 cells for `text`. Capitals get the capital indicator (two for an all-capital
    word of 2+ letters); characters in unsupported_chars(text) are skipped."""
    cells: list[int] = []
    s = text or ""
    n = len(s)
    numeric = False
    caps_word = False
    isletter = lambda c: c.isascii() and c.lower() in LETTERS
    for i, ch in enumerate(s):
        nxt = s[i + 1] if i + 1 < n else ""
        if ch.isdigit() and ch in DIGITS:
            if not numeric:
                cells.append(NUMERIC_INDICATOR)
                numeric = True
            cells.append(DIGITS[ch])
        elif isletter(ch):
            low = ch.lower()
            if ch.isupper() and not caps_word:
                j = i
                while j < n and isletter(s[j]):
                    j += 1
                word = s[i:j]
                at_start = i == 0 or not isletter(s[i - 1])
                if at_start and len(word) >= 2 and word.isupper():
                    cells += [CAPITAL_INDICATOR, CAPITAL_INDICATOR]
                    caps_word = True
                else:
                    cells.append(CAPITAL_INDICATOR)
                numeric = False                  # the capital indicator ends numeric mode
            if numeric and low in "abcdefghij":
                cells.append(GRADE1_INDICATOR)   # letter a-j right after digits
            numeric = False
            cells.append(LETTERS[low])
            if caps_word and not isletter(nxt):
                caps_word = False
        elif ch.isspace():
            numeric = False
            caps_word = False
            if cells and cells[-1] != 0:          # collapse runs of spaces
                cells.append(0)
        elif ch in PUNCT:
            # decimal point / thousands comma inside a number keep numeric mode
            if not (numeric and ch in ".," and nxt.isdigit()):
                numeric = False
            cells.extend(PUNCT[ch])
        # anything else: skipped (numeric mode unchanged); reported by unsupported_chars()
    while cells and cells[-1] == 0:
        cells.pop()
    while cells and cells[0] == 0:
        cells.pop(0)
    return cells


def to_unicode(text: str) -> str:
    return "".join(chr(0x2800 + c) for c in to_cells(text))


def cells_width_mm(n_cells: int) -> float:
    """Physical width (mm) of the raised extent of `n_cells` consecutive cells."""
    if n_cells <= 0:
        return 0.0
    return (n_cells - 1) * BRAILLE_CELL_SPACING_MM + BRAILLE_DOT_SPACING_MM + BRAILLE_DOT_DIAMETER_MM


def cell_height_mm() -> float:
    return 2 * BRAILLE_DOT_SPACING_MM + BRAILLE_DOT_DIAMETER_MM


def text_size_px(text: str, px_per_mm: float) -> tuple[int, int]:
    """(w, h) px that draw_braille would use for `text`."""
    n = len(to_cells(text))
    if n == 0:
        return 0, 0
    return int(round(cells_width_mm(n) * px_per_mm)), int(round(cell_height_mm() * px_per_mm))


def dot_centres(cells: list[int], x: float, y: float, px_per_mm: float) -> list[tuple[float, float]]:
    r = BRAILLE_DOT_DIAMETER_MM / 2
    out = []
    for i, c in enumerate(cells):
        ox = x + (i * BRAILLE_CELL_SPACING_MM + r) * px_per_mm
        oy = y + r * px_per_mm
        for b, (cx, cy) in enumerate(_DOT_POS):
            if c & (1 << b):
                out.append((ox + cx * BRAILLE_DOT_SPACING_MM * px_per_mm, oy + cy * BRAILLE_DOT_SPACING_MM * px_per_mm))
    return out


_SHIFT = 4


def draw_cells(ink: np.ndarray, cells: list[int], x: int, y: int, px_per_mm: float) -> tuple[int, int]:
    # cv2's filled circle comes out ~0.7 px wider than 2*r: compensate so dots measure 1.5 mm
    rad = int(round((BRAILLE_DOT_DIAMETER_MM / 2 * px_per_mm - 0.35) * (1 << _SHIFT)))
    for cx, cy in dot_centres(cells, x, y, px_per_mm):
        # pixel centres are at integer coords: shift by -0.5 so a dot covers [x, x+d)
        c = (int(round((cx - 0.5) * (1 << _SHIFT))), int(round((cy - 0.5) * (1 << _SHIFT))))
        cv2.circle(ink, c, rad, 255, -1, cv2.LINE_8, _SHIFT)
    if not cells:
        return 0, 0
    return int(round(cells_width_mm(len(cells)) * px_per_mm)), int(round(cell_height_mm() * px_per_mm))


def draw_braille(ink: np.ndarray, text: str, x: int, y: int, px_per_mm: float) -> tuple[int, int]:
    """Draw raised dots (255) for `text` with the top-left of the first cell at (x, y). Returns (w, h) px."""
    return draw_cells(ink, to_cells(text), x, y, px_per_mm)


# F03: prefix for braille text that could not be printed in full (read by readiness)
OVERFLOW_PREFIX = "Braille overflow: "
