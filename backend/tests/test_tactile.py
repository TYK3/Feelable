import time

import cv2
import numpy as np
import pytest

from engine import braille
from engine.tactile import assign_keys, render_tactile
from engine.types import MIN_LINE_MM, MIN_PART_MM, PAGE_MARGIN_MM, PX_PER_MM, Part, TactileOptions
try:
    from tests.fixtures_tactile import grid_analysis, lineart_analysis, plant_cell_analysis, single_part_analysis
except ImportError:  # pragma: no cover - tests/ itself on sys.path
    from fixtures_tactile import grid_analysis, lineart_analysis, plant_cell_analysis, single_part_analysis

RULES = ["Lines at least 1 mm", "Outlines 1.5 mm", "Small parts enlarged", "Neighbours have different textures",
         "2 mm gap around textures", "Braille keys clear of lines", "8 parts or fewer"]


def _render(make, **opt):
    a = make()
    assign_keys(a.parts)
    t = time.perf_counter()
    r = render_tactile(a, TactileOptions(**opt))
    return a, r, time.perf_counter() - t


@pytest.fixture(scope="module")
def plant():
    return _render(plant_cell_analysis)


@pytest.fixture(scope="module")
def grid():
    return _render(grid_analysis)


@pytest.fixture(scope="module")
def single():
    return _render(single_part_analysis)


@pytest.fixture(scope="module")
def lineart():
    return _render(lineart_analysis)


ALL = ["plant", "grid", "single", "lineart"]


def _checks(r):
    return {c.rule: c for c in r.checks}


# ------------------------------------------------------------------ keys
def _part(pid, name, include=True):
    return Part(pid, name, "", "", (0, 0, 0), np.zeros((4, 4), np.uint8), [], (0, 0, 0, 0), 0, include=include)


def test_assign_keys_rules():
    ps = [_part("p1", "nucleus"), _part("p2", "cell wall"), _part("p3", "cytoplasm"), _part("p4", "part 3"),
          _part("p5", "chloroplast"), _part("p6", "Vacuole"), _part("p7", "ribosome", include=False)]
    assign_keys(ps)
    keys = {p.name: p.key for p in ps}
    assert keys["nucleus"] == "n" and keys["Vacuole"] == "v" and keys["part 3"] == "p"
    # equal areas -> list order decides: the first "c" part keeps the single letter
    assert keys["cell wall"] == "c" and keys["cytoplasm"] == "cy" and keys["chloroplast"] == "ch"
    assert keys["ribosome"] == ""


def test_assign_keys_biggest_part_keeps_letter_and_keys_come_from_name():
    ps = [_part("p1", "nucleolus"), _part("p2", "nucleus"), _part("p3", "left atrium"), _part("p4", "left ventricle")]
    ps[0].area, ps[1].area, ps[2].area, ps[3].area = 100, 5000, 3000, 4000
    assign_keys(ps)
    keys = {p.name: p.key for p in ps}
    assert keys["nucleus"] == "n" and keys["nucleolus"] == "nu"
    assert keys["left ventricle"] == "l" and keys["left atrium"] == "la"


def test_assign_keys_fallback_unique_deterministic():
    names = ["part 1", "part 2", "part 3", "123", "pa", "pb"]
    ps = [_part(f"p{i}", n) for i, n in enumerate(names)]
    assign_keys(ps)
    keys = [p.key for p in ps]
    assert len(set(keys)) == len(keys)
    assert all(k.isalpha() and k.islower() and k.isascii() and 1 <= len(k) <= 2 for k in keys)
    ps2 = [_part(f"p{i}", n) for i, n in enumerate(names)]
    assign_keys(ps2)
    assert [p.key for p in ps2] == keys


def test_assign_keys_many_parts():
    ps = [_part(f"p{i}", "part") for i in range(40)]
    assign_keys(ps)
    assert len({p.key for p in ps}) == 40


# ------------------------------------------------------------------ result shape
@pytest.mark.parametrize("name", ALL)
def test_all_checks_present(name, request):
    a, r, _ = request.getfixturevalue(name)
    assert [c.rule for c in r.checks] == RULES
    assert all(isinstance(c.passed, bool) and c.detail for c in r.checks)


@pytest.mark.parametrize("name", ALL)
def test_canvas_and_layers(name, request):
    a, r, _ = request.getfixturevalue(name)
    assert (r.width, r.height) == (round(297 * PX_PER_MM), round(210 * PX_PER_MM))
    assert r.ink.shape == (r.height, r.width) and r.ink.dtype == np.uint8
    assert set(np.unique(r.ink)) <= {0, 255}
    assert r.legend_ink.shape == r.ink.shape
    assert r.print_layer.shape == (r.height, r.width, 3)
    # raised ink is black on the print layer, and print text never covers raised ink
    assert np.all(r.print_layer[r.ink > 0] == 0)
    assert r.ink.any()


@pytest.mark.parametrize("name", ALL)
def test_keys_unique_and_on_page(name, request):
    a, r, _ = request.getfixturevalue(name)
    inc = [p for p in a.parts if p.include]
    assert set(r.key_positions) == {p.id for p in inc}
    keys = [p.key for p in inc]
    assert len(set(keys)) == len(keys)
    m = PAGE_MARGIN_MM * PX_PER_MM
    for p in inc:
        x, y = r.key_positions[p.id]
        w, h = braille.text_size_px(p.key, PX_PER_MM)
        assert m - 1 <= x and x + w <= r.width - m + 1 and m - 1 <= y and y + h <= r.height - m + 1
        assert r.ink[y:y + h, x:x + w].any()          # dots really drawn there


@pytest.mark.parametrize("name", ALL)
def test_part_polygons_align_with_ink(name, request):
    a, r, _ = request.getfixturevalue(name)
    inc = [p for p in a.parts if p.include]
    assert set(r.part_polygons) == {p.id for p in inc}
    for p in inc:
        polys = r.part_polygons[p.id]
        assert polys
        outline = np.zeros_like(r.ink)
        for poly in polys:
            arr = np.array(poly)
            assert arr.shape[1] == 2 and len(arr) >= 3
            assert arr[:, 0].min() >= 0 and arr[:, 0].max() < r.width
            assert arr[:, 1].min() >= 0 and arr[:, 1].max() < r.height
            cv2.polylines(outline, [arr.astype(np.int32)], True, 255, 1)
        # the polygon boundary runs along raised outline ink (touch mode hit-tests these)
        on = np.count_nonzero(r.ink[outline > 0]) / np.count_nonzero(outline)
        assert on > 0.9, (p.name, on)


def test_excluded_part_not_drawn(plant):
    a, r, _ = plant
    excl = [p for p in a.parts if not p.include]
    assert excl
    for p in excl:
        assert p.id not in r.part_polygons and p.id not in r.key_positions and p.id not in r.textures
        assert p.id not in [e["part_id"] for e in r.legend]


def test_legend_order_and_ink(plant):
    a, r, _ = plant
    inc = [p for p in a.parts if p.include]
    assert [e["part_id"] for e in r.legend] == [p.id for p in inc]
    assert all(e["key"] == p.key and e["name"] == p.name for e, p in zip(r.legend, inc))
    assert r.legend_ink.any()
    # one braille row per part, 10 mm pitch
    ys = [e["y"] for e in r.legend]
    assert all(abs((b - a_) - 10 * PX_PER_MM) <= 1 for a_, b in zip(ys, ys[1:]))


def test_legend_wraps_into_columns():
    from engine.tactile import _legend
    parts = [_part(f"p{i}", f"structure {i}") for i in range(30)]
    assign_keys(parts)
    W, H = int(297 * PX_PER_MM), int(210 * PX_PER_MM)
    m = int(PAGE_MARGIN_MM * PX_PER_MM)
    ink, legend, _more = _legend(parts, W, H, m, PX_PER_MM)
    assert [e["part_id"] for e in legend] == [p.id for p in parts]
    assert len({e["x"] for e in legend}) >= 2                      # second column used
    for e in legend:
        assert e["y"] + 6.5 * PX_PER_MM <= H - m + 1
    ys, xs = np.nonzero(ink)
    assert xs.max() < W - m + 2 and ys.max() < H - m + 2


# ------------------------------------------------------------------ tactile rules
def _poly_mask(r, pid):
    m = np.zeros_like(r.ink)
    cv2.fillPoly(m, [np.array(p, np.int32) for p in r.part_polygons[pid]], 255)
    return m


@pytest.mark.parametrize("name", ["plant", "grid"])
def test_neighbour_textures_differ(name, request):
    a, r, _ = request.getfixturevalue(name)
    inc = [p for p in a.parts if p.include]
    masks = {p.id: _poly_mask(r, p.id) for p in inc}
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))      # 3 mm
    near = {i: cv2.dilate(m, k) for i, m in masks.items()}
    for i, p in enumerate(inc):
        for q in inc[i + 1:]:
            if np.any(near[p.id] & masks[q.id]):
                tp, tq = r.textures[p.id], r.textures[q.id]
                assert tp == "none" or tp != tq, (p.name, q.name, tp)
    assert _checks(r)["Neighbours have different textures"].passed


def test_plant_textures(plant):
    a, r, _ = plant
    t = {p.name: r.textures[p.id] for p in a.parts if p.include}
    assert t["cell wall"] == "none" and t["cytoplasm"] == "none"            # containers
    assert t["mitochondrion"] == "solid"                                      # tiny
    inner = [t[n] for n in ("vacuole", "nucleus", "chloroplast")]
    assert len(set(inner)) == 3 and not set(inner) & {"none", "solid"}


def test_small_parts_enlarged(plant):
    a, r, _ = plant
    c = _checks(r)["Small parts enlarged"]
    assert c.passed and "mitochondrion" in c.detail
    mito = next(p for p in a.parts if p.name == "mitochondrion")
    for poly in r.part_polygons[mito.id]:
        (_, _), (w, h), _ = cv2.minAreaRect(np.array(poly, np.int32))
        assert min(w, h) >= MIN_PART_MM * PX_PER_MM * 0.85


def test_min_line_width_measured(plant, grid, lineart):
    for a, r, _ in (plant, grid, lineart):
        c = _checks(r)["Lines at least 1 mm"]
        assert c.passed, c.detail
        # independent measure: every raised component has a core at least ~1 mm wide
        dt = cv2.distanceTransform(r.ink, cv2.DIST_L2, 5)
        n, cc = cv2.connectedComponents(r.ink)
        core = np.zeros(n)
        np.maximum.at(core, cc.ravel(), dt.ravel())
        assert core[1:].min() >= MIN_LINE_MM * PX_PER_MM / 2 - 0.5


@pytest.mark.parametrize("name", ["plant", "grid", "single"])
def test_core_checks_pass(name, request):
    a, r, _ = request.getfixturevalue(name)
    c = _checks(r)
    for rule in ("Outlines 1.5 mm", "2 mm gap around textures", "Braille keys clear of lines"):
        assert c[rule].passed, (rule, c[rule].detail)


def test_too_many_parts_warns(grid):
    a, r, _ = grid
    c = _checks(r)["8 parts or fewer"]
    assert not c.passed and "split into two graphics" in c.detail
    assert len(set(r.textures.values())) == 6


def test_lineart_only(lineart):
    a, r, _ = lineart
    assert r.part_polygons == {} and r.key_positions == {} and r.legend == []
    s, ox, oy = r.transform
    # the 1-px speck at (1150, 860) is clutter: nothing raised there
    x, y = int(1150 * s + ox), int(860 * s + oy)
    assert not r.ink[max(0, y - 10):y + 10, max(0, x - 10):x + 10].any()
    assert all(c.passed for c in r.checks)


def test_transform_maps_source_into_drawing(single):
    a, r, _ = single
    s, ox, oy = r.transform
    x0, y0, x1, y1 = a.parts[0].bbox
    m = PAGE_MARGIN_MM * PX_PER_MM
    assert x0 * s + ox >= m and x1 * s + ox <= r.width - m
    assert y0 * s + oy >= m and y1 * s + oy <= r.height - m


def test_title_option_and_letter_paper():
    a = single_part_analysis()
    assign_keys(a.parts)
    r1 = render_tactile(a, TactileOptions(paper="letter", title_in_braille=False))
    assert (r1.width, r1.height) == (round(279.4 * PX_PER_MM), round(215.9 * PX_PER_MM))
    r2 = render_tactile(a, TactileOptions(paper="letter", title_in_braille=True))
    m = int(PAGE_MARGIN_MM * PX_PER_MM)
    band = (slice(m, m + int(7 * PX_PER_MM)), slice(m, m + int(40 * PX_PER_MM)))
    assert not r1.ink[band].any() and r2.ink[band].any()


def test_keys_assigned_defensively():
    a = single_part_analysis()
    for p in a.parts:
        p.key = ""
    r = render_tactile(a)
    assert a.parts[0].key == "r" and r.legend[0]["key"] == "r"


def test_deterministic(plant):
    a, r, _ = plant
    r2 = render_tactile(a)
    assert np.array_equal(r.ink, r2.ink) and r.key_positions == r2.key_positions and r.textures == r2.textures


def test_empty_analysis():
    a = lineart_analysis()
    a.linework[:] = 0
    r = render_tactile(a)
    assert r.ink.any()            # title only
    assert len(r.checks) == len(RULES)


def test_render_speed(plant):
    assert plant[2] < 3.0


def test_braille_warning_for_unsupported_characters():
    """F15: characters with no braille sign are listed once, with the declared code."""
    a = single_part_analysis()
    a.parts[0].name = "café µ"
    a.warnings = []
    render_tactile(a)
    w = [x for x in a.warnings if x.startswith("Braille (")]
    assert len(w) == 1 and "é" in w[0] and "µ" in w[0]
    assert "UEB grade 1 (uncontracted), English" in w[0]
    render_tactile(a)                      # re-render after an edit: no duplicate
    assert len([x for x in a.warnings if x.startswith("Braille (")]) == 1
    a.parts[0].name = "cafe"
    render_tactile(a)
    assert not [x for x in a.warnings if x.startswith("Braille (")]
