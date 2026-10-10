"""Robust image intake (CONTRACT-v2 §1, audit F11/F12): odd formats, huge photos, non-diagram photos.

OWNER: Engine team.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import textwrap

import cv2
import numpy as np
import pytest
from PIL import Image

from engine import analyze as A
from engine.analyze import ImageTooBig, NotAnImage, analyze, load_image
from engine.types import Analysis

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPO = os.path.dirname(BACKEND)


def _save(im: Image.Image, fmt: str, **kw) -> bytes:
    bio = io.BytesIO()
    im.save(bio, format=fmt, **kw)
    return bio.getvalue()


def _drawing(w=300, h=200) -> np.ndarray:
    a = np.full((h, w, 3), 255, np.uint8)
    cv2.circle(a, (w // 3, h // 2), min(w, h) // 4, (220, 60, 60), -1)
    cv2.rectangle(a, (w // 2, h // 4), (w - 20, 3 * h // 4), (60, 160, 60), -1)
    return a


# ----------------------------------------------------------------------------- formats

@pytest.mark.parametrize("fmt,mode", [
    ("PNG", "RGB"), ("PNG", "RGBA"), ("PNG", "P"), ("PNG", "L"), ("PNG", "LA"), ("PNG", "1"),
    ("JPEG", "RGB"), ("JPEG", "L"), ("JPEG", "CMYK"), ("WEBP", "RGB"), ("WEBP", "RGBA"),
    ("TIFF", "RGB"), ("TIFF", "CMYK"), ("BMP", "RGB"), ("GIF", "P"),
])
def test_formats_and_modes_decode(fmt, mode):
    im = Image.fromarray(_drawing()).convert(mode)
    out = load_image(_save(im, fmt))
    assert out.dtype == np.uint8 and out.shape == (200, 300, 3)
    # the red circle stays reddish (CMYK / palette conversions keep colours roughly right)
    if mode not in ("L", "LA", "1"):
        r, g, b = out[100, 100].astype(int)
        assert r > g + 60 and r > b + 60


def test_palette_with_transparency_over_white():
    im = Image.fromarray(_drawing()).convert("P")
    im.info["transparency"] = int(im.getpixel((5, 5)))     # page colour becomes transparent
    out = load_image(_save(im, "PNG", transparency=im.info["transparency"]))
    assert tuple(out[5, 5]) == (255, 255, 255)


def test_16bit_grey_png():
    a = (np.linspace(0, 65535, 200 * 300).reshape(200, 300)).astype(np.uint16)
    im = Image.fromarray(a, mode="I;16")
    out = load_image(_save(im, "PNG"))
    assert out.shape == (200, 300, 3) and out[0, 0, 0] < 10 and out[-1, -1, 0] > 245


def test_animated_gif_first_frame():
    frames = [Image.new("RGB", (60, 40), c).convert("P") for c in ((255, 0, 0), (0, 0, 255))]
    data = _save(frames[0], "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    out = load_image(data)
    assert out.shape == (40, 60, 3) and out[20, 30, 0] > 200 and out[20, 30, 2] < 50


def test_tiny_image_analyzes():
    out = load_image(_save(Image.fromarray(_drawing(20, 20)), "PNG"))
    assert out.shape == (20, 20, 3)
    a = analyze(out, use_ai=False)
    assert isinstance(a, Analysis)


def test_extreme_aspect_ratio_clean_error():
    with pytest.raises(ValueError, match="thin strip"):
        load_image(_save(Image.new("RGB", (3000, 20), "white"), "PNG"))
    # 10:1 still works end to end
    a = analyze(load_image(_save(Image.fromarray(_drawing(1000, 100)), "PNG")), use_ai=False)
    assert isinstance(a, Analysis)


def test_garbage_and_truncated():
    with pytest.raises(NotAnImage):
        load_image(b"GIF89a not really")
    data = _save(Image.fromarray(_drawing()), "JPEG")
    with pytest.raises(ValueError):
        load_image(data[: len(data) // 3] + b"\x00" * 4)   # truncated photo: a clean ValueError (or decodes)


def test_bomb_guard_too_big():
    # 9000 x 9000 = 81 MP of flat grey compresses to a tiny PNG
    data = _save(Image.new("L", (9000, 9000), 200), "PNG")
    with pytest.raises(ImageTooBig) as ei:
        load_image(data)
    assert str(ei.value) == "too big" and ei.value.code == "too_big"


def test_big_png_downscaled_after_decode():
    data = _save(Image.new("RGB", (6000, 4000), (200, 200, 200)), "PNG")
    out = load_image(data)
    assert max(out.shape[:2]) <= A.LOAD_MAX_SIDE and out.shape[1] > out.shape[0]


def test_exif_rotation_survives_draft():
    im = Image.fromarray(np.zeros((3000, 4000, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6        # rotate 90 CW on display
    out = load_image(_save(im, "JPEG", exif=exif, quality=70))
    assert out.shape[0] > out.shape[1] and max(out.shape[:2]) <= A.LOAD_MAX_SIDE


def test_heic_decodes_when_pillow_heif_installed():
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    try:
        data = _save(Image.fromarray(_drawing()), "HEIF", quality=90)
    except Exception as exc:  # pragma: no cover - wheel without an encoder
        pytest.skip(f"no HEIF encoder: {exc}")
    out = load_image(data)
    assert out.shape == (200, 300, 3)


# ----------------------------------------------------------------------------- F01 / F10 four shapes

# shape centres in the 1000 x 700 fixture (the triangle's centroid sits low)
SHAPES = {"circle": (285, 255), "square": (690, 255), "triangle": (285, 565), "ellipse": (690, 535)}


def _fixture(name):
    with open(os.path.join(DATA, name), "rb") as f:
        return load_image(f.read())


@pytest.mark.parametrize("name,rotation", [("four-shapes.png", 0), ("four-shapes-grayscale.png", 0),
                                           ("four-shapes-sideways.png", 90)])
def test_four_shapes_four_named_parts(name, rotation):
    a = analyze(_fixture(name), use_ai=False)
    assert a.input_kind == "diagram" and a.rotation == rotation
    assert sorted(p.name for p in a.parts) == sorted(SHAPES), [p.name for p in a.parts]
    s = a.width / 1000.0
    for p in a.parts:
        ys, xs = np.nonzero(p.mask)
        cx, cy = xs.mean() / s, ys.mean() / s
        ex, ey = SHAPES[p.name]
        assert abs(cx - ex) < 30 and abs(cy - ey) < 30, (p.name, cx, cy)     # its own shape ...
        assert (p.bbox[2] - p.bbox[0]) / s < 230 and p.count == 1           # ... and only that one
        assert p.match_method == "leader" and p.label_text and p.description_source == "auto"
    assert a.unmatched_labels == []


def test_blank_page_fixture():
    a = analyze(_fixture("blank-page.png"), use_ai=False)
    assert a.input_kind == "blank" and a.parts == []
    assert any("No parts found" in w for w in a.warnings)


# ----------------------------------------------------------------------------- photos


def _object_photo(seed=0):
    """A dark rounded object with a light button on a patterned, noisy mat (like a phone photo)."""
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    small = rng.integers(60, 255, (12, 16, 3)).astype(np.uint8)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(np.clip(bg.astype(np.int16) + rng.normal(0, 12, (h, w, 3)), 0, 255).astype(np.uint8),
                           (3, 3), 0)
    yy, xx = np.mgrid[0:h, 0:w]
    obj = ((xx - 400) / 170.0) ** 2 + ((yy - 310) / 210.0) ** 2 <= 1
    shade = (35 + 25 * (xx - 230) / 340.0).clip(20, 70).astype(np.uint8)
    for c in range(3):
        img[:, :, c][obj] = shade[obj]
    btn = ((xx - 400) / 45.0) ** 2 + ((yy - 230) / 45.0) ** 2 <= 1
    img[btn] = (215, 210, 200)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, obj, btn


def _iou(a, b):
    return float((a & b).sum()) / max(1, float((a | b).sum()))


def test_photo_object_outline_and_detail():
    img, obj, btn = _object_photo()
    a = analyze(img, use_ai=False)
    assert a.input_kind == "photo_object" and a.diagram_type == "photo"
    assert a.title == "photo of an object"
    assert a.parts[0].name == "main object" and len(a.parts) <= 1 + 3
    m = cv2.resize(obj.astype(np.uint8), (a.width, a.height), interpolation=cv2.INTER_NEAREST) > 0
    assert _iou(a.parts[0].mask > 0, m) > 0.9
    b = cv2.resize(btn.astype(np.uint8), (a.width, a.height), interpolation=cv2.INTER_NEAREST) > 0
    assert any(_iou(p.mask > 0, b) > 0.6 for p in a.parts[1:]), "the button is a separate detail"
    assert not any(p.name.startswith("part ") or p.name == "background" for p in a.parts)


def test_photo_object_with_ai_describe_object(monkeypatch):
    from engine import ai
    from types import SimpleNamespace as NS
    img, obj, btn = _object_photo()
    calls = []

    def fake_describe(image_rgb):
        calls.append(image_rgb.shape)
        H, W = image_rgb.shape[:2]
        s = W / 800.0
        return NS(title="Computer mouse", summary="A wireless mouse seen from above.", parts=[
            NS(name="mouse", description="The whole mouse, in the middle.",
               box=(int(225 * s), int(95 * s), int(575 * s), int(525 * s))),
            NS(name="scroll button", description="Round button near the top.",
               box=(int(350 * s), int(180 * s), int(450 * s), int(280 * s))),
            NS(name="tiny speck", description="Too small.", box=(0, 0, 2, 2)),
        ])
    monkeypatch.setattr(ai, "get_provider_info", lambda: {"provider": "gemini", "model": "fake"})
    monkeypatch.setattr(ai, "describe_object", fake_describe, raising=False)
    monkeypatch.setattr(ai, "enrich", lambda *a_, **k: pytest.fail("enrich is for diagrams"))
    stages = []
    a = analyze(img, use_ai=True, progress=stages.append)
    assert calls and stages == ["read", "find", "ai"]
    assert a.input_kind == "photo_object" and a.ai_used and a.title == "computer mouse"
    names = [p.name for p in a.parts]
    assert names[:2] == ["mouse", "scroll button"] and "tiny speck" not in names
    assert all(p.match_method == "ai" and p.description_source == "ai" for p in a.parts)
    b = cv2.resize(btn.astype(np.uint8), (a.width, a.height), interpolation=cv2.INTER_NEAREST) > 0
    assert _iou(a.parts[1].mask > 0, b) > 0.6


def test_photo_ai_failure_still_gives_outline(monkeypatch):
    from engine import ai
    monkeypatch.setattr(ai, "get_provider_info", lambda: {"provider": "gemini", "model": "fake"})
    monkeypatch.setattr(ai, "describe_object", lambda img: None, raising=False)
    monkeypatch.setattr(ai, "last_error", lambda: "AI timed out", raising=False)
    a = analyze(_object_photo()[0], use_ai=True)
    assert a.parts and a.parts[0].name == "main object"
    assert any("AI description unavailable" in w and "timed out" in w for w in a.warnings)


def test_brand_text_never_names_a_part():
    img, _, _ = _object_photo()
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    d.text((300, 380), "amazonbasics", fill=(140, 140, 140), font=font)
    a = analyze(np.asarray(pil), use_ai=False)
    assert a.input_kind == "photo_object"
    assert not any("amazon" in p.name for p in a.parts)
    assert all(lb.part_id is None for lb in a.labels) and a.unmatched_labels == []


MOUSE = os.path.join(REPO, "qa", "audit", "reproduce", "mouse-photo.jpg")
if not os.path.exists(MOUSE):     # the audit folder lives beside the repo checkout
    MOUSE = os.path.join(os.path.dirname(REPO), "qa", "audit", "reproduce", "mouse-photo.jpg")


@pytest.mark.skipif(not os.path.exists(MOUSE), reason="user photo not available")
def test_real_mouse_photo():
    with open(MOUSE, "rb") as f:
        a = analyze(load_image(f.read()), use_ai=False)
    assert a.input_kind == "photo_object"
    assert [p.name for p in a.parts][0] == "main object" and len(a.parts) <= 2
    p = a.parts[0]
    assert 0.15 < p.area / (a.width * a.height) < 0.45


def test_scene_and_noise():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (500, 700, 3), dtype=np.uint8)
    a = analyze(noise, use_ai=False)
    assert a.parts == [] and any("No parts found" in w for w in a.warnings)


def test_real_scene_photo():
    data = pytest.importorskip("skimage.data")
    img = cv2.cvtColor(data.camera(), cv2.COLOR_GRAY2RGB)        # cameraman on a lawn, public domain
    s = analyze(img, use_ai=False)
    assert s.input_kind == "photo_scene" and s.title == "photo of a scene"
    assert 1 <= len(s.parts) <= 5
    assert any("scene" in w_ for w_ in s.warnings)
    assert not any(p.name.startswith("part ") for p in s.parts)


def test_text_page():
    from PIL import ImageDraw, ImageFont
    pil = Image.new("RGB", (1000, 760), "white")
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for i in range(14):
        d.text((40, 30 + i * 50), "The quick brown fox jumps over the lazy dog again", fill="black", font=font)
    a = analyze(np.asarray(pil), use_ai=False)
    assert a.input_kind == "text_page" and a.parts == []
    assert any("text" in w for w in a.warnings)


# ----------------------------------------------------------------------------- split / merge


def test_split_and_merge_round_trip():
    a = analyze(_fixture("four-shapes.png"), use_ai=False)
    ids = {p.name: p.id for p in a.parts}
    n0 = len(a.parts)
    kept = A.merge_parts(a, [ids["circle"], ids["ellipse"]])
    assert kept == ids["circle"] and len(a.parts) == n0 - 1
    merged = next(p for p in a.parts if p.id == kept)
    assert merged.count == 2 and merged.name == "circle"
    assert all(lb.part_id != ids["ellipse"] for lb in a.labels)
    a.unmatched_labels = ["ellipse"]
    for lb in a.labels:
        if lb.part_id == kept and "llipse" in lb.text:
            lb.part_id = None
    new_ids = A.split_part(a, kept)
    assert new_ids[0] == kept and len(new_ids) == 2 and len(a.parts) == n0
    byid = {p.id: p for p in a.parts}
    assert byid[kept].name == "circle" and byid[kept].count == 1
    assert byid[new_ids[1]].name == "ellipse" and a.unmatched_labels == []
    assert len({p.id for p in a.parts}) == len(a.parts)
    with pytest.raises(ValueError):
        A.split_part(a, kept)                 # one piece now
    with pytest.raises(ValueError):
        A.merge_parts(a, [kept])
    with pytest.raises(ValueError):
        A.merge_parts(a, [kept, "p99"])


def test_split_unlabelled_pieces_get_part_numbers():
    a = analyze(_fixture("four-shapes.png"), use_ai=False)
    ids = [p.id for p in a.parts]
    A.merge_parts(a, ids[:3])
    out = A.split_part(a, ids[0])
    assert len(out) == 3
    assert all(next(p for p in a.parts if p.id == i).name == f"part {i[1:]}" for i in out[1:])


# ----------------------------------------------------------------------------- misc contract


def test_progress_without_ai_and_broken_callback():
    stages = []
    analyze(_fixture("four-shapes.png"), use_ai=False, progress=stages.append)
    assert stages == ["read", "find"]

    def boom(stage):
        raise RuntimeError("reporter down")
    a = analyze(_fixture("four-shapes.png"), use_ai=False, progress=boom)
    assert len(a.parts) == 4


def test_apply_ai_marks_description_source():
    from types import SimpleNamespace as NS
    a = analyze(_fixture("four-shapes.png"), use_ai=False)
    pid = a.parts[0].id
    A._apply_ai(a, NS(parts=[{"id": pid, "description": "Square, top right."}], title="", diagram_type=""))
    p = next(p for p in a.parts if p.id == pid)
    assert p.description_source == "ai" and p.match_method == "leader"
    assert all(q.description_source == "auto" for q in a.parts if q.id != pid)


def test_old_pickles_get_new_field_defaults():
    import pickle
    a = analyze(_fixture("blank-page.png"), use_ai=False)
    state = dict(a.__dict__)
    for k in ("input_kind", "unmatched_labels", "rotation", "notes"):
        state.pop(k)
    old = Analysis.__new__(Analysis)
    old.__setstate__(state)
    b = pickle.loads(pickle.dumps(old))
    assert b.input_kind == "diagram" and b.unmatched_labels == [] and b.rotation == 0 and b.notes == []


def test_48mp_jpeg_peak_memory_bounded(tmp_path):
    """F12: a 48 MP phone photo must decode with < ~150 MB extra peak memory."""
    src = tmp_path / "big.jpg"
    make = textwrap.dedent(f"""
        import numpy as np
        from PIL import Image
        W, H = 8064, 6048
        x = np.linspace(0, 255, W, dtype=np.float32)[None, :]
        y = np.linspace(0, 255, H, dtype=np.float32)[:, None]
        arr = np.dstack([((x + y) / 2).astype(np.uint8), np.broadcast_to(x.astype(np.uint8), (H, W)),
                         np.broadcast_to(y.astype(np.uint8), (H, W))])
        Image.fromarray(arr).save({str(src)!r}, quality=85)
    """)
    subprocess.run([sys.executable, "-c", make], check=True, timeout=120)
    if not os.path.exists("/proc/self/status"):
        pytest.skip("needs Linux /proc to read the peak RSS")
    # VmHWM (peak RSS) resets on exec; ru_maxrss would inherit the big pytest parent's peak
    probe = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {BACKEND!r})
        from engine.analyze import load_image

        def kb(key):
            for line in open('/proc/self/status'):
                if line.startswith(key):
                    return int(line.split()[1]) * 1024
        data = open({str(src)!r}, 'rb').read()
        base = kb('VmRSS:')
        img = load_image(data)
        print(img.shape[0], img.shape[1], max(0, kb('VmHWM:') - base))
    """)
    out = subprocess.run([sys.executable, "-c", probe], check=True, capture_output=True, text=True, timeout=120)
    h, w, delta = map(int, out.stdout.split()[-3:])
    assert max(h, w) <= A.LOAD_MAX_SIDE
    assert delta < 150e6, f"load_image peak grew by {delta / 1e6:.0f} MB"
