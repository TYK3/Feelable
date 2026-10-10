import numpy as np
import pytest

from engine import braille
from engine.types import BRAILLE_CELL_SPACING_MM, PX_PER_MM


def test_nucleus_unicode():
    assert braille.to_unicode("nucleus") == "⠝⠥⠉⠇⠑⠥⠎"


def test_capitals_marked():
    # UEB: ⠠ before a capital letter, ⠠⠠ before a word in capitals
    assert braille.to_unicode("Nucleus") == "⠠" + braille.to_unicode("nucleus")
    assert braille.to_unicode("DNA") == "⠠⠠⠙⠝⠁"
    assert braille.to_unicode("mRNA") == "⠍⠠⠗⠠⠝⠠⠁"
    assert braille.to_unicode("3D") == "⠼⠉⠠⠙"          # capital indicator ends numeric mode


def test_unsupported_chars_reported():
    assert braille.unsupported_chars("café 5 µm") == {"é", "µ"}
    assert braille.unsupported_chars("cell wall (2)") == set()
    assert braille.unsupported_chars("") == set()
    assert braille.to_unicode("café") == braille.to_unicode("caf")
    assert "UEB grade 1" in braille.BRAILLE_CODE and "English" in braille.BRAILLE_CODE


def test_common_symbols():
    assert braille.to_unicode("50%") == "⠼⠑⠚⠨⠴"
    assert braille.to_unicode("a/b") == "⠁⠸⠌⠃"
    assert braille.to_unicode("1+1=2") == "⠼⠁⠐⠖⠼⠁⠐⠶⠼⠃"


def test_abc_bits():
    assert braille.to_cells("abc") == [0b000001, 0b000011, 0b001001]


def test_all_letters_distinct():
    cells = braille.to_cells("abcdefghijklmnopqrstuvwxyz")
    assert len(cells) == 26 and len(set(cells)) == 26
    assert braille.to_unicode("w") == "⠺" and braille.to_unicode("z") == "⠵"


def test_digit_uses_numeric_indicator():
    assert braille.to_unicode("3") == "⠼⠉"
    assert braille.to_unicode("10") == "⠼⠁⠚"


def test_grade1_indicator_after_digit():
    assert braille.to_unicode("a1b") == "⠁⠼⠁⠰⠃"
    # letters k-z after a digit need no indicator; a space ends numeric mode
    assert braille.to_unicode("1k") == "⠼⠁⠅"
    assert braille.to_unicode("1 b") == "⠼⠁⠀⠃"


def test_punctuation():
    assert braille.to_unicode("a.") == "⠁⠲"
    assert braille.to_unicode(",?!'-:;") == "⠂⠦⠖⠄⠤⠒⠆"
    assert braille.to_unicode("(a)") == "⠐⠣⠁⠐⠜"
    assert braille.to_unicode("1.5") == "⠼⠁⠲⠑"


def test_space_and_unknown():
    assert braille.to_unicode("cell wall") == "⠉⠑⠇⠇⠀⠺⠁⠇⠇"
    assert braille.to_unicode("a#b") == "⠁⠃"
    assert braille.to_cells("") == []


def test_widths():
    assert braille.cells_width_mm(0) == 0
    assert braille.cells_width_mm(1) == pytest.approx(4.0)
    assert braille.cells_width_mm(3) - braille.cells_width_mm(2) == pytest.approx(BRAILLE_CELL_SPACING_MM)


def test_draw_braille_dots():
    ink = np.zeros((100, 200), np.uint8)
    w, h = braille.draw_braille(ink, "abc", 10, 20, PX_PER_MM)
    assert w == round(braille.cells_width_mm(3) * PX_PER_MM)
    assert h == round(6.5 * PX_PER_MM)
    assert set(np.unique(ink)) <= {0, 255}
    ys, xs = np.nonzero(ink)
    assert xs.min() >= 10 - 1 and xs.max() <= 10 + w and ys.min() >= 20 - 1 and ys.max() <= 20 + h
    import cv2
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    assert n - 1 == 1 + 2 + 2          # a=1 dot, b=2 dots, c=2 dots
    # each dot ~1.5 mm in diameter
    d = st[1:, cv2.CC_STAT_WIDTH] / PX_PER_MM
    assert np.all(np.abs(d - 1.5) <= 0.25)


def test_draw_empty():
    ink = np.zeros((10, 10), np.uint8)
    assert braille.draw_braille(ink, "", 0, 0, PX_PER_MM) == (0, 0)
    assert not ink.any()
