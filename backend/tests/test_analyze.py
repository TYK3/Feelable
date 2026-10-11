"""Vision team tests: engine.analyze / engine.ocr / engine.vision_utils.

Synthetic fixtures are drawn in code (deterministic). The real prototype image lives in
tests/data/vision_plant_cell.png. To LOOK at results while tuning:

    cd backend && python -m tests.test_analyze /tmp/out_dir      # writes overlay PNGs
"""
from __future__ import annotations

import io
import json
import os
import sys

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from engine import ai
from engine.analyze import analyze, load_image
from engine.types import WORKING_LONG_SIDE, Analysis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FONT = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"


def _font(size, bold=False):
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:  # pragma: no cover
        return ImageFont.load_default()


# =============================================================================== fixtures


def make_colour_diagram() -> np.ndarray:
    """Animal-cell style diagram: nested coloured shapes, leaders, 60+ clutter dots, title."""
    W, H = 1000, 700
    rng = np.random.default_rng(7)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Fig. 2.1 An animal cell", font=_font(30, True), fill=(25, 25, 25))
    d.ellipse((150, 140, 650, 600), fill=(250, 200, 160))            # cytoplasm / cell
    d.ellipse((330, 260, 480, 410), fill=(90, 110, 210))             # nucleus (nested)
    d.ellipse((380, 310, 420, 350), fill=(60, 70, 160))              # nucleolus (shading detail)
    d.rounded_rectangle((210, 430, 330, 490), 20, fill=(230, 120, 40))   # mitochondrion
    d.polygon([(520, 200), (600, 250), (560, 330), (480, 290)], fill=(80, 170, 90))  # golgi
    arr = np.array(im)
    n = 0
    while n < 70:  # clutter dots inside the cell only
        x, y = int(rng.integers(170, 630)), int(rng.integers(160, 580))
        if tuple(arr[y, x]) == (250, 200, 160) and tuple(arr[max(0, y - 6), x]) == (250, 200, 160) \
                and tuple(arr[min(H - 1, y + 6), x]) == (250, 200, 160):
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(70, 70, 70))
            n += 1
    lab = _font(22)
    for text, tx, ty, px, py in [("Nucleus", 760, 250, 450, 300), ("Mitochondrion", 30, 520, 250, 470),
                                 ("Golgi body", 760, 170, 570, 260), ("Cytoplasm", 760, 520, 600, 480)]:
        tw = d.textlength(text, font=lab)
        sx = tx - 8 if tx > 500 else tx + tw + 8
        d.line((sx, ty + 13, px, py), fill=(30, 30, 30), width=2)
        d.text((tx, ty), text, font=lab, fill=(20, 20, 20))
    return np.array(im)


def make_lineart_diagram() -> np.ndarray:
    """Black outlines of 3 closed shapes, an arrow, labels (one with a leader)."""
    W, H = 1100, 700
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 30), "Figure 5 Simple shapes", font=_font(30, True), fill=(0, 0, 0))
    d.ellipse((90, 200, 330, 440), outline=(0, 0, 0), width=4)                 # circle
    d.rectangle((470, 200, 690, 420), outline=(0, 0, 0), width=4)              # square
    d.polygon([(820, 440), (1010, 440), (915, 220)], outline=(0, 0, 0), width=4)  # triangle
    # arrow from circle to square (between the shapes, below them)
    d.line((200, 560, 560, 560), fill=(0, 0, 0), width=4)
    d.polygon([(560, 545), (590, 560), (560, 575)], fill=(0, 0, 0))
    lab = _font(24)
    d.text((165, 460), "Circle", font=lab, fill=(0, 0, 0))
    d.text((535, 440), "Square", font=lab, fill=(0, 0, 0))
    # label with a leader pointing into the triangle
    d.text((860, 600), "Triangle", font=lab, fill=(0, 0, 0))
    d.line((905, 596, 915, 380), fill=(0, 0, 0), width=2)
    return np.array(im)


def make_photo(diagram: np.ndarray) -> np.ndarray:
    """Photograph-like version: page warped onto a dark table, uneven light, noise, JPEG."""
    rng = np.random.default_rng(11)
    h, w = diagram.shape[:2]
    page = cv2.copyMakeBorder(diagram, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    ph, pw = page.shape[:2]
    OW, OH = 1600, 1200
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    dst = np.float32([[230, 160], [1390, 120], [1470, 1060], [150, 1010]])
    M = cv2.getPerspectiveTransform(src, dst)
    table = np.zeros((OH, OW, 3), np.float32)
    table[:] = (70, 60, 50)
    table += rng.normal(0, 6, table.shape)
    warped = cv2.warpPerspective(page, M, (OW, OH), flags=cv2.INTER_LINEAR).astype(np.float32)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (OW, OH)) > 127
    out = table.copy()
    out[mask] = warped[mask]
    yy, xx = np.mgrid[0:OH, 0:OW]
    # uneven lighting: a side gradient plus a vignette (corners ~40 % darker than the brightest spot)
    r2 = ((xx - 0.7 * OW) / OW) ** 2 + ((yy - 0.3 * OH) / OH) ** 2
    light = (0.62 + 0.38 * (xx / OW)) * np.exp(-1.2 * r2)
    light = light / light.max() * 0.97
    out = out * light[:, :, None] * np.array([1.0, 0.97, 0.9])           # warm cast
    out += rng.normal(0, 3.0, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _png_bytes(arr: np.ndarray, mode="RGB") -> bytes:
    bio = io.BytesIO()
    Image.fromarray(arr, mode=mode if arr.ndim == 3 else None).save(bio, format="PNG")
    return bio.getvalue()


def _names(a: Analysis) -> list[str]:
    return [p.name for p in a.parts]


def _part(a: Analysis, name: str):
    for p in a.parts:
        if p.name == name:
            return p
    return None


@pytest.fixture(autouse=True)
def _no_ai(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture(scope="module")
def colour_result():
    return analyze(make_colour_diagram(), use_ai=False)


@pytest.fixture(scope="module")
def plant_result():
    with open(os.path.join(DATA, "vision_plant_cell.png"), "rb") as f:
        return analyze(load_image(f.read()), use_ai=False)


# =============================================================================== load_image


def test_load_image_garbage_raises():
    with pytest.raises(ValueError, match="Not an image"):
        load_image(b"this is definitely not an image")
    with pytest.raises(ValueError):
        load_image(b"")


def test_load_image_rgba_composited_over_white():
    rgba = np.zeros((40, 60, 4), np.uint8)
    rgba[:, :30] = (255, 0, 0, 255)       # opaque red
    rgba[:, 30:] = (0, 0, 255, 0)         # fully transparent
    out = load_image(_png_bytes(rgba, "RGBA"))
    assert out.shape == (40, 60, 3) and out.dtype == np.uint8
    assert tuple(out[5, 5]) == (255, 0, 0)
    assert tuple(out[5, 50]) == (255, 255, 255)


def test_load_image_grey_and_exif():
    grey = np.full((30, 50), 128, np.uint8)
    out = load_image(_png_bytes(grey))
    assert out.shape == (30, 50, 3)
    im = Image.fromarray(np.zeros((20, 40, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    bio = io.BytesIO()
    im.save(bio, format="JPEG", exif=exif)
    assert load_image(bio.getvalue()).shape == (40, 20, 3)


# =============================================================================== colour mode


def test_colour_diagram_parts_and_names(colour_result):
    a = colour_result
    assert a.width == WORKING_LONG_SIDE or a.height == WORKING_LONG_SIDE
    assert len(a.parts) >= 3
    names = _names(a)
    for nm in ("nucleus", "mitochondrion", "golgi body", "cytoplasm"):
        assert nm in names, names
    # nucleolus folded into nucleus, nucleus nested in cytoplasm
    nuc = _part(a, "nucleus")
    assert "Inside the cytoplasm" in nuc.description
    assert nuc.count == 1
    # ids / order / shape invariants
    assert [p.id for p in a.parts] == [f"p{i + 1}" for i in range(len(a.parts))]
    assert all(a.parts[i].area >= a.parts[i + 1].area for i in range(len(a.parts) - 1))
    for p in a.parts:
        assert p.mask.shape == (a.height, a.width) and p.mask.dtype == np.uint8
        assert set(np.unique(p.mask)).issubset({0, 255})
        assert p.key == "" and p.polygons and p.area == int((p.mask > 0).sum())


def test_colour_diagram_clutter_title_and_text(colour_result):
    a = colour_result
    assert a.title == "animal cell"
    assert any("small dots" in d for d in a.dropped), a.dropped
    assert not a.ai_used and "analyze" in a.timings
    # text and leaders never end up in linework
    for lb in a.labels:
        x0, y0, x1, y1 = lb.bbox
        assert a.linework[y0:y1, x0:x1].sum() == 0, lb.text
    roles = {lb.text: lb.role for lb in a.labels}
    assert any(r == "title" for r in roles.values())
    assert sum(1 for lb in a.labels if lb.part_id) >= 4


# =============================================================================== line-art


def test_lineart_diagram():
    a = analyze(make_lineart_diagram(), use_ai=False)
    assert a.diagram_type == "lineart"
    assert len(a.parts) >= 3
    names = _names(a)
    for nm in ("circle", "square", "triangle"):
        assert nm in names, names
    # the arrow (between y=545..575 in the 1100-wide drawing -> scaled) is linework
    s = a.width / 1100
    ys, xs = slice(int(530 * s), int(590 * s)), slice(int(220 * s), int(580 * s))
    assert a.linework[ys, xs].sum() > 255 * 100
    assert a.title == "simple shapes"


# =============================================================================== photo


def test_photo_is_flattened_and_parts_found():
    photo = make_photo(make_colour_diagram())
    a = analyze(photo, use_ai=False)
    assert a.page_flattened
    assert len(a.parts) >= 3
    names = _names(a)
    assert sum(nm in names for nm in ("nucleus", "mitochondrion", "golgi body", "cytoplasm")) >= 3, names
    # paper became white-ish
    border = np.concatenate([a.image[:10].reshape(-1, 3), a.image[-10:].reshape(-1, 3)])
    assert np.median(border) > 200


def test_clean_screenshot_not_flattened(colour_result):
    assert colour_result.page_flattened is False


# =============================================================================== degenerate inputs


def test_blank_image():
    a = analyze(np.full((600, 800, 3), 255, np.uint8), use_ai=False)
    assert a.parts == []
    assert any("No parts found" in w for w in a.warnings)
    assert a.title == "diagram"


def test_random_noise():
    rng = np.random.default_rng(0)
    a = analyze(rng.integers(0, 256, (500, 700, 3), dtype=np.uint8), use_ai=False)
    assert len(a.parts) == 0
    assert any("No parts found" in w for w in a.warnings)


def test_tiny_and_grey_inputs():
    a = analyze(np.full((5, 5, 3), 200, np.uint8), use_ai=False)
    assert isinstance(a, Analysis)
    g = analyze(np.full((300, 300), 255, np.uint8), use_ai=False)
    assert g.parts == []


# =============================================================================== real prototype image


def test_plant_cell(plant_result):
    a = plant_result
    names = _names(a)
    for nm in ("cell wall", "nucleus", "vacuole", "chloroplast", "mitochondrion", "cytoplasm"):
        assert nm in names, names
    assert _part(a, "chloroplast").count == 5
    assert _part(a, "mitochondrion").count == 3
    # precision audit: a plural label pointing among dots names every dot instance, it is not
    # dropped as "too small" (the tactile step enlarges each dot to a feelable size)
    rib = _part(a, "ribosomes")
    assert rib is not None and rib.count >= 3, a.dropped
    assert not any(d.startswith("ribosomes") for d in a.dropped), a.dropped
    assert a.title in ("plant cell", "typical plant cell")
    assert _part(a, "nucleus").description == "Nucleus. Inside the cytoplasm."
    assert "There are" not in _part(a, "chloroplast").description   # shape counts are not object counts
    assert a.timings["analyze"] < 6.0


def test_ai_enrichment_applied(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def fake_enrich(image, analysis):
        parts = [{"id": p.id, "name": p.name, "description": f"{p.name.capitalize()} described.", "drop": False}
                 for p in analysis.parts]
        parts[-1]["drop"] = True
        return ai.AIEnrichment(title="animal cell (ai)", diagram_type="biology", parts=parts)

    monkeypatch.setattr(ai, "enrich", fake_enrich)
    a = analyze(make_colour_diagram(), use_ai=True)
    assert a.ai_used and a.diagram_type == "biology" and a.title == "animal cell (ai)"
    assert a.parts[-1].include is False
    assert a.parts[0].description.endswith("described.")


def test_ai_failure_keeps_offline(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(ai, "enrich", lambda image, analysis: None)
    a = analyze(make_colour_diagram(), use_ai=True)
    assert not a.ai_used
    assert any(w.startswith("AI description unavailable") and w.endswith("used labels from the image")
               for w in a.warnings)
    assert "nucleus" in _names(a)


def test_use_ai_false_never_calls_ai(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(ai, "enrich", boom)
    a = analyze(make_colour_diagram(), use_ai=False)
    assert not a.ai_used


# =============================================================================== unlabelled parts / charts


def make_scene_diagram() -> np.ndarray:
    """Framed scene: unlabelled land touching the frame, two same-shaped clouds (one labelled)."""
    W, H = 1100, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 25), "Fig. 4 A rainy day", font=_font(28, True), fill=(20, 20, 20))
    d.rectangle((40, 90, 1060, 700), outline=(0, 0, 0), width=3)
    d.rectangle((43, 520, 1057, 697), fill=(120, 180, 90))                    # land, unlabelled
    for (cx, cy) in ((300, 230), (760, 230)):                                  # two grey clouds
        for dx, dy, r in ((-60, 10, 45), (0, -15, 60), (60, 10, 45), (0, 25, 45)):
            d.ellipse((cx + dx - r, cy + dy - r, cx + dx + r, cy + dy + r), fill=(150, 160, 175))
    d.ellipse((520, 300, 600, 380), fill=(250, 200, 50))                     # sun
    lab = _font(22)
    d.text((240, 380), "Cloud", font=lab, fill=(0, 0, 0))
    d.line((270, 376, 290, 260), fill=(0, 0, 0), width=2)
    d.text((620, 420), "Sun", font=lab, fill=(0, 0, 0))
    d.line((634, 415, 578, 362), fill=(0, 0, 0), width=2)
    return np.array(im)


def make_globe_diagram() -> np.ndarray:
    """Unlabelled big container holding two labelled parts; title names the object."""
    W, H = 1000, 720
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 25), "Fig. 6 Inside the planet", font=_font(28, True), fill=(20, 20, 20))
    d.ellipse((200, 120, 680, 600), fill=(90, 150, 215))                       # planet, unlabelled
    d.ellipse((380, 300, 500, 420), fill=(250, 210, 80))                      # core
    d.rectangle((260, 200, 340, 260), fill=(110, 180, 90))                    # island
    lab = _font(22)
    for text, tx, ty, px, py in (("Core", 780, 340, 470, 360), ("Island", 780, 160, 320, 230)):
        d.line((tx - 8, ty + 13, px, py), fill=(30, 30, 30), width=2)
        d.text((tx, ty), text, font=lab, fill=(20, 20, 20))
    return np.array(im)


def make_bar_chart() -> np.ndarray:
    """Bars with values above them, y-axis tick numbers, a rotated y-axis title, category labels."""
    W, H = 1100, 760
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 25), "Fig. 5 Pets in our class", font=_font(28, True), fill=(20, 20, 20))
    x_axis_y, left = 640, 200
    d.line((left, 120, left, x_axis_y), fill=(0, 0, 0), width=3)
    d.line((left, x_axis_y, 1040, x_axis_y), fill=(0, 0, 0), width=3)
    lab, small = _font(22), _font(20)
    for v in (0, 5, 10):
        y = x_axis_y - v * 45
        d.line((left - 8, y, left, y), fill=(0, 0, 0), width=2)
        d.text((left - 40, y - 12), str(v), font=small, fill=(60, 60, 60))
    bars = [("Cats", 7, (220, 90, 60)), ("Dogs", 10, (60, 120, 210)), ("Fish", 3, (80, 170, 90)),
            ("Birds", 5, (230, 170, 30))]
    for i, (name, v, col) in enumerate(bars):
        x0 = 280 + i * 190
        d.rectangle((x0, x_axis_y - v * 45, x0 + 110, x_axis_y - 2), fill=col)
        d.text((x0 + 45, x_axis_y - v * 45 - 32), str(v), font=small, fill=(60, 60, 60))
        d.text((x0 + 55 - d.textlength(name, font=lab) / 2, x_axis_y + 14), name, font=lab, fill=(0, 0, 0))
    t = Image.new("RGB", (300, 34), "white")
    ImageDraw.Draw(t).text((5, 3), "Number of pets", font=lab, fill=(60, 60, 60))
    im.paste(t.rotate(90, expand=True), (100, 230))
    return np.array(im)


def test_unlabelled_copy_and_background():
    a = analyze(make_scene_diagram(), use_ai=False)
    cloud = _part(a, "cloud")
    assert cloud is not None and cloud.count == 2, _names(a)
    bg = _part(a, "background")
    assert bg is not None and bg.include is False
    assert "large unlabelled background (switch it back on in review if needed)" in a.dropped
    assert not any(n.startswith("part ") for n in _names(a)), _names(a)


def test_unlabelled_container_named_from_title():
    a = analyze(make_globe_diagram(), use_ai=False)
    planet = _part(a, "planet")
    assert planet is not None, _names(a)
    assert planet.description.startswith("Planet. Contains the ")
    assert "core" in planet.description and "island" in planet.description
    assert "Inside the planet" in _part(a, "core").description


def test_bar_chart_values():
    a = analyze(make_bar_chart(), use_ai=False)
    assert a.diagram_type == "chart"
    dogs, fish, cats = _part(a, "dogs"), _part(a, "fish"), _part(a, "cats")
    assert dogs and fish and cats, _names(a)
    assert dogs.description.startswith("Dogs. 10") and "Bar 2 of 4, the tallest." in dogs.description
    assert fish.description.startswith("Fish. 3") and "the shortest" in fish.description
    assert cats.description.startswith("Cats. 7")
    assert "number of pets" in dogs.description
    assert len(dogs.description.split()) <= 15
    # axis tick numbers are not attached to bars
    for lb in a.labels:
        if lb.text.strip() in ("0", "5") and lb.bbox[0] < 200 * a.width / 1100:
            assert lb.part_id is None


def test_inside_phrase_only_for_real_containers():
    from engine.analyze import _describe
    from engine.types import Part

    def part(pid, name, box):
        m = np.zeros((200, 200), np.uint8)
        m[box[1]:box[3], box[0]:box[2]] = 255
        return Part(id=pid, name=name, description="", key="", color=(0, 0, 0), mask=m, polygons=[[(0, 0)]],
                    bbox=box, area=int((m > 0).sum()))
    # a thin wall around a big chamber: the wall is barely visible, so no "Inside the wall"
    wall = part("p1", "septum", (10, 10, 190, 190))
    chamber = part("p2", "left ventricle", (20, 20, 180, 180))
    _describe([wall, chamber])
    assert chamber.description == "Left ventricle."
    # a roomy container keeps the phrase
    cyto = part("p1", "cytoplasm", (10, 10, 190, 190))
    nuc = part("p2", "nucleus", (80, 80, 120, 120))
    _describe([cyto, nuc])
    assert nuc.description == "Nucleus. Inside the cytoplasm."


def test_respacing_of_joined_ocr_words():
    from engine.vision_utils import clean_name, main_noun
    assert clean_name("Outercore") == "outer core"
    assert clean_name("Leftventricle") == "left ventricle"
    assert clean_name("Mitochondrion") == "mitochondrion"
    assert main_noun("inside the earth") == "earth"
    assert main_noun("parts of a flower") == "flower"


# =============================================================================== samples (Content team)

MANIFEST = os.path.join(REPO, "samples", "manifest.json")


def _load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


@pytest.mark.skipif(not os.path.exists(MANIFEST), reason="samples/manifest.json not created yet")
def test_samples_manifest():
    failures = []
    for s in _load_manifest():
        path = os.path.join(REPO, "samples", s["file"])
        with open(path, "rb") as f:
            a = analyze(load_image(f.read()), use_ai=False)
        names = _names(a)
        exp = s.get("expected_parts", [])
        sq = lambda t: t.lower().replace(" ", "")    # "outercore" == "outer core"
        hit = sum(1 for e in exp if any(sq(e) in sq(n) or sq(n) in sq(e) for n in names if not n.startswith("part ")))
        if len(a.parts) < s.get("min_parts", 0):
            failures.append(f"{s['id']}: {len(a.parts)} parts < min {s.get('min_parts')}: {names}")
        if exp and hit < 0.6 * len(exp):
            failures.append(f"{s['id']}: matched {hit}/{len(exp)} expected {exp} in {names}")
    assert not failures, "\n".join(failures)


# =============================================================================== debug entry point

if __name__ == "__main__":  # pragma: no cover
    from engine.analyze import _analyze_impl
    from engine.vision_utils import save_overlay

    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    cases = {"colour": make_colour_diagram(), "lineart": make_lineart_diagram(),
             "photo": make_photo(make_colour_diagram())}
    for name, img in cases.items():
        cv2.imwrite(os.path.join(out, f"in_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        a, dbg = _analyze_impl(img, use_ai=False)
        save_overlay(a, os.path.join(out, f"ov_{name}.png"), dbg)
        print(name, dbg.get("mode"), [(p.name, p.count) for p in a.parts], a.title, a.dropped, a.warnings, a.timings)


def test_titles_with_ocr_glued_words():
    from engine.vision_utils import clean_title
    assert clean_title("Fig. 9.1Thewatercycle") == "water cycle"
    assert clean_title("Fig.3.2Atypicalplantcell") == "plant cell"
    assert clean_title("Fig. 7.4The human heart (front view)") == "human heart"
    assert clean_title("Fig. 12.3Asimpleseriescircuit") == "simple series circuit"
    assert clean_title("Fig. 9.1 The water cycle") == "water cycle"
