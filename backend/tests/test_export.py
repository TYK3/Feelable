import os
import re
import time

import cv2
import numpy as np
import pytest
import trimesh

from engine.export import build_relief, export_all, stl_size_mm
from engine.tactile import assign_keys, render_tactile
from engine.types import TactileOptions

try:
    from tests.fixtures_tactile import lineart_analysis, plant_cell_analysis, single_part_analysis
except ImportError:  # pragma: no cover
    from fixtures_tactile import lineart_analysis, plant_cell_analysis, single_part_analysis

KEYS = {"original", "png", "print_png", "legend_png", "pdf", "stl"}
NAMES = {"original": "original.png", "png": "tactile.png", "print_png": "tactile_print.png",
         "legend_png": "legend.png", "pdf": "tactile.pdf", "stl": "relief.stl"}


@pytest.fixture(scope="module")
def plant_export(tmp_path_factory):
    out = tmp_path_factory.mktemp("plant")
    a = plant_cell_analysis()
    assign_keys(a.parts)
    t0 = time.perf_counter()
    r = render_tactile(a)
    t1 = time.perf_counter()
    files = export_all(a, r, str(out))
    t2 = time.perf_counter()
    return a, r, files, (t1 - t0, t2 - t1)


def _pdf_pages(path):
    data = open(path, "rb").read()
    assert data.startswith(b"%PDF-")
    try:
        import pypdf
        rd = pypdf.PdfReader(path)
        return len(rd.pages), [(float(p.mediabox.width), float(p.mediabox.height)) for p in rd.pages]
    except ImportError:
        boxes = re.findall(rb"/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]", data)
        return len(re.findall(rb"/Type /Page\b(?!s)", data)), [(float(w), float(h)) for w, h in boxes]


def test_all_files_written(plant_export):
    a, r, files, _ = plant_export
    assert set(files) == KEYS
    for k, path in files.items():
        assert os.path.isabs(path) and os.path.basename(path) == NAMES[k]
        assert os.path.getsize(path) > 0


def test_pngs(plant_export):
    a, r, files, _ = plant_export
    orig = cv2.imread(files["original"])
    assert orig.shape[:2] == (a.height, a.width)
    t = cv2.imread(files["png"], cv2.IMREAD_GRAYSCALE)
    assert t.shape == (r.height, r.width)
    assert np.array_equal(t == 0, r.ink > 0)                     # black = raised
    pr = cv2.imread(files["print_png"])
    assert pr.shape == (r.height, r.width, 3)
    grey = (pr[:, :, 0] > 100) & (pr[:, :, 0] < 200)            # some grey print text present
    assert grey.sum() > 100
    lg = cv2.imread(files["legend_png"])
    assert lg.shape == (r.height, r.width, 3)
    assert (lg.max(axis=2) == 0).sum() > 100 and ((lg[:, :, 1] > 100) & (lg[:, :, 1] < 200)).sum() > 100


def test_pdf_two_landscape_pages(plant_export):
    a, r, files, _ = plant_export
    n, boxes = _pdf_pages(files["pdf"])
    assert n == 2
    for w, h in boxes:
        assert w == pytest.approx(297 / 25.4 * 72, abs=1) and h == pytest.approx(210 / 25.4 * 72, abs=1)
    assert os.path.getsize(files["pdf"]) < 8_000_000


def test_stl_watertight(plant_export):
    a, r, files, _ = plant_export
    size = os.path.getsize(files["stl"])
    assert size < 25 * 1024 * 1024
    m = trimesh.load(files["stl"], force="mesh")
    assert m.is_watertight
    zmin, zmax = m.bounds[0][2], m.bounds[1][2]
    assert zmin == pytest.approx(0.0, abs=1e-3)
    assert zmax - zmin == pytest.approx(3.0, abs=0.05)
    # x/y extents in millimetres within the page
    assert m.bounds[1][0] <= 297.0 + 1e-3 and m.bounds[1][1] <= 210.0 + 1e-3


def test_stl_build_time_and_domes(plant_export):
    a, r, files, _ = plant_export
    t = time.perf_counter()
    mesh = build_relief(r.ink, r.px_per_mm)
    assert time.perf_counter() - t < 10
    assert mesh.is_watertight
    # braille / dot texture domes reach 2.6 mm, lines 3.0 mm
    z = np.round(mesh.vertices[:, 2], 2)
    assert np.any(np.isclose(z, 2.6)) and np.any(np.isclose(z, 3.0))


def test_timing_budget(plant_export):
    render_s, export_s = plant_export[3]
    assert render_s + export_s < 4.0, (render_s, export_s)


def test_letter_lineart_and_single(tmp_path):
    for make, paper in ((lineart_analysis, "letter"), (single_part_analysis, "a4")):
        a = make()
        assign_keys(a.parts)
        r = render_tactile(a, TactileOptions(paper=paper))
        files = export_all(a, r, str(tmp_path / f"{paper}_{make.__name__}"))
        n, boxes = _pdf_pages(files["pdf"])
        assert n == 2
        if paper == "letter":
            assert boxes[0][0] == pytest.approx(279.4 / 25.4 * 72, abs=1)
        assert trimesh.load(files["stl"], force="mesh").is_watertight


def test_empty_ink_relief():
    ink = np.zeros((100, 150), np.uint8)
    m = build_relief(ink, 5.0)
    assert m.is_watertight and m.bounds[1][2] == pytest.approx(2.0)
    assert stl_size_mm(ink, 5.0) == [30.0, 20.0, 2.0]


def test_stl_size_matches_built_mesh(plant_export):
    """F16: the size we report equals the bounds of the STL file actually built."""
    a, r, files, _ = plant_export
    size = stl_size_mm(files["png"], r.px_per_mm)
    mesh = trimesh.load(files["stl"], force="mesh")
    ext = mesh.bounds[1] - mesh.bounds[0]
    assert size[0] == pytest.approx(ext[0], abs=0.25)
    assert size[1] == pytest.approx(ext[1], abs=0.25)
    assert size[2] == pytest.approx(ext[2], abs=0.05)
    assert size[0] < 297.0                          # the plate is content + margin, not the paper
    assert size == stl_size_mm(r.ink, r.px_per_mm)  # path or array give the same answer


def test_stl_size_dots_only():
    ink = np.zeros((200, 300), np.uint8)
    cv2.circle(ink, (100, 100), 3, 255, -1)          # one 1.4 mm braille-like dot
    size = stl_size_mm(ink, 5.0)
    ext = build_relief(ink, 5.0).bounds
    assert size[2] == pytest.approx(ext[1][2] - ext[0][2], abs=0.05)


def test_relief_is_manifold_when_holes_touch_outline():
    """TS-05: a texture hole touching the outline at a point must not create 4-face edges."""
    import numpy as np
    from collections import Counter
    from engine.export import build_relief
    ink = np.zeros((200, 300), np.uint8)
    ink[40:160, 40:260] = 255
    for x in range(60, 240, 12):                 # grid of holes, the first column touching the border
        for y in range(60, 140, 12):
            ink[y:y + 6, x:x + 6] = 0
    ink[60:66, 40:46] = 0                         # hole touching the outer edge
    ink[90:96, 46:52] = 0
    ink[96:102, 52:58] = 0                        # diagonal hole chain (corner touches)
    m = build_relief(ink, 5.0)
    c = Counter(map(tuple, np.sort(m.edges, axis=1)))
    assert sum(v > 2 for v in c.values()) == 0
    assert sum(v == 1 for v in c.values()) == 0
