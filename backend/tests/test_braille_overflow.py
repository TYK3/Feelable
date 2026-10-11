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
