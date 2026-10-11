"""Exports: PNGs, 2-page swell-paper PDF (300 dpi) and a 3D-printable relief STL.

OWNER: Tactile team.

STL design (page 1):
  * base plate 2.0 mm thick covering the CONTENT bbox + 10 mm margin (clipped to the page), so
    the print fits ordinary 220 mm beds more often than a full A4 plate would;
  * raised lines / textures / outlines: extruded 1.0 mm above the plate (top at z = 3.0 mm);
  * dot-like features (braille cells, dot texture): spherical-cap domes 0.6 mm high, Ø 1.5 mm;
  * every feature is its own closed shell, sunk 0.05 mm into the plate so slicers fuse them.
  Coordinates in mm, x to the right, y up (the page reads correctly from above).
"""
from __future__ import annotations

import math
import os
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

from engine.tactile import PRINT_GREY, _font, print_font, text_box
from engine.types import Analysis, TactileResult

DPI = 300
BASE_MM = 2.0
RAISE_MM = 1.0
DOME_MM = 0.6
SINK_MM = 0.05
STL_MARGIN_MM = 10.0


# ============================================================================ raster pages
def _ink_rgb(ink: np.ndarray) -> np.ndarray:
    out = np.full(ink.shape + (3,), 255, np.uint8)
    out[ink > 0] = 0
    return out


def _legend_print(tactile: TactileResult, img: Image.Image, scale: float, page: int = 0) -> None:
    """Grey print beside each braille legend row (scale = output px per tactile px)."""
    d = ImageDraw.Draw(img)
    ppm = tactile.px_per_mm * scale
    font = print_font(ppm)
    for e in tactile.legend:
        if int(e.get("page", 0)) != page:
            continue
        if e.get("print_lines"):                     # F03: wrapped print on a paged key
            for x, y, t in e["print_lines"]:
                d.text((x * scale, y * scale), t, font=font, fill=PRINT_GREY, anchor="lt")
        elif "print_x" in e:
            d.text((e["print_x"] * scale, e["print_y"] * scale), e["print"], font=font, fill=PRINT_GREY, anchor="lt")
    # heading print beside the braille word "key"
    margin_px = 12.0 * tactile.px_per_mm
    hx = margin_px + 4 * 6.2 * tactile.px_per_mm
    small = _font(max(8, int(round(2.6 * ppm / 0.727))))
    d.text((hx * scale, (margin_px + 1.6 * tactile.px_per_mm) * scale), "key (print for the teacher, not raised)",
           font=small, fill=PRINT_GREY, anchor="lt")


def legend_pages(tactile: TactileResult) -> list[np.ndarray]:
    """All braille key pages (raised ink), first page first."""
    return [tactile.legend_ink] + list(getattr(tactile, "legend_pages", None) or [])


def legend_rgb(tactile: TactileResult, scale: float = 1.0, page: int = 0) -> np.ndarray:
    pages = legend_pages(tactile)
    H, W = pages[page].shape
    size = (int(round(W * scale)), int(round(H * scale)))
    img = Image.new("RGB", size, "white")
    _legend_print(tactile, img, scale, page)
    arr = np.array(img)
    ink = _hires(pages[page], size)
    arr[ink > 0] = 0
    return arr


def legend_rgb_all(tactile: TactileResult, scale: float = 1.0) -> np.ndarray:
    """Every key page stacked top to bottom with a thin grey rule between pages (legend.png)."""
    imgs = [legend_rgb(tactile, scale, i) for i in range(len(legend_pages(tactile)))]
    if len(imgs) == 1:
        return imgs[0]
    rule = np.full((6, imgs[0].shape[1], 3), 200, np.uint8)
    out = []
    for i, im in enumerate(imgs):
        if i:
            out.append(rule)
        out.append(im)
    return np.concatenate(out, 0)


def _hires(ink: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upscale a binary raster to `size` with smooth edges: light AA blur, linear resize, threshold."""
    if (ink.shape[1], ink.shape[0]) == size:
        return ink
    soft = cv2.GaussianBlur(ink, (0, 0), 0.6)
    big = cv2.resize(soft, size, interpolation=cv2.INTER_LINEAR)
    return (big > 127).astype(np.uint8) * 255


def _page_hires(print_layer: np.ndarray, ink: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Grey (L) page at `size`: grey print resampled smoothly, raised ink crisp black on top."""
    grey = cv2.cvtColor(print_layer, cv2.COLOR_RGB2GRAY)
    grey[ink > 0] = 255                       # ink is redrawn crisply below
    grey = cv2.resize(grey, size, interpolation=cv2.INTER_LINEAR)
    grey[_hires(ink, size) > 0] = 0
    return grey


def _write_pdf(path: str, pages: list[np.ndarray], page_mm: tuple[float, float]) -> None:
    """Minimal PDF writer: one full-page DeviceGray image per page, Flate-compressed (lossless)."""
    w_pt, h_pt = page_mm[0] / 25.4 * 72, page_mm[1] / 25.4 * 72
    objs: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 3 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, pg in enumerate(pages):
        h, w = pg.shape
        pobj, cobj, iobj = 3 + 3 * i, 4 + 3 * i, 5 + 3 * i
        objs.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:.2f} {h_pt:.2f}] "
                     f"/Resources << /XObject << /Im0 {iobj} 0 R >> /ProcSet [/PDF /ImageB] >> "
                     f"/Contents {cobj} 0 R >>").encode())
        content = f"q {w_pt:.2f} 0 0 {h_pt:.2f} 0 0 cm /Im0 Do Q".encode()
        objs.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
        data = zlib.compress(np.ascontiguousarray(pg).tobytes(), 6)
        objs.append((f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
                     f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>\nstream\n").encode()
                    + data + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for k, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info << /Producer (Feelable) "
            f"/Title (Feelable tactile graphic) >> >>\nstartxref\n{xref}\n%%EOF\n").encode()
    with open(path, "wb") as f:
        f.write(out)


# ============================================================================ STL
def _dome_template(radius: float, height: float, seg: int = 12, rings: int = 3):
    """Closed spherical-cap mesh (base disk at z=0), apex at z=height."""
    R = (radius ** 2 + height ** 2) / (2 * height)
    zc = height - R
    verts = [(0.0, 0.0, 0.0)]                         # 0: base centre
    angles = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    for k in range(rings):
        z = height * k / rings
        r = math.sqrt(max(R ** 2 - (z - zc) ** 2, 0.0))
        verts += [(r * math.cos(a), r * math.sin(a), z) for a in angles]
    verts.append((0.0, 0.0, height))                  # apex
    apex = len(verts) - 1
    faces = []
    ring = lambda k, j: 1 + k * seg + (j % seg)
    for j in range(seg):                              # bottom (normal -z)
        faces.append((0, ring(0, j + 1), ring(0, j)))
    for k in range(rings - 1):
        for j in range(seg):
            a, b, c, d = ring(k, j), ring(k, j + 1), ring(k + 1, j), ring(k + 1, j + 1)
            faces += [(a, b, d), (a, d, c)]
    for j in range(seg):
        faces.append((ring(rings - 1, j), ring(rings - 1, j + 1), apex))
    return np.array(verts, np.float64), np.array(faces, np.int64)


def _box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int64)
    return v, f


def _plate_bounds(ink: np.ndarray, px_per_mm: float) -> tuple[float, float, float, float]:
    """Base plate (x0, y0, x1, y1) in mm, y up: content bbox + STL_MARGIN_MM, clipped to the page."""
    H, W = ink.shape[:2]
    s = 1.0 / px_per_mm
    ys, xs = np.nonzero(ink)
    if len(xs):
        return (max(0.0, xs.min() * s - STL_MARGIN_MM), max(0.0, (H - ys.max() - 1) * s - STL_MARGIN_MM),
                min(W * s, (xs.max() + 1) * s + STL_MARGIN_MM), min(H * s, (H - ys.min()) * s + STL_MARGIN_MM))
    return 0.0, 0.0, W * s, H * s


def _ink_from_png(path: str) -> np.ndarray:
    png = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if png is None:
        raise FileNotFoundError(path)
    return ((png < 128).astype(np.uint8)) * 255


def stl_size_mm(tactile_png_path_or_ink, px_per_mm: float) -> list[float]:
    """F16: the real footprint [width, height, depth] in mm of the STL that build_relief /
    ensure_stl make from this tactile raster — without building the mesh (a few ms).
    Accepts the saved tactile.png path (black = raised) or the ink array (255 = raised)."""
    if isinstance(tactile_png_path_or_ink, (str, os.PathLike)):
        ink = _ink_from_png(os.fspath(tactile_png_path_or_ink))
    else:
        ink = (np.asarray(tactile_png_path_or_ink) > 0).astype(np.uint8) * 255
    x0, y0, x1, y1 = _plate_bounds(ink, px_per_mm)
    depth = BASE_MM
    if ink.any():
        # same dot test as build_relief: domes (2.6 mm) vs extruded relief (3.0 mm)
        n, _cc, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        dot_d = 1.5 * px_per_mm
        w_, h_, a_ = st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT], st[1:, cv2.CC_STAT_AREA]
        is_dot = (np.maximum(w_, h_) <= dot_d * 1.35) & (np.minimum(w_, h_) >= dot_d * 0.6) & \
                 (a_ >= 0.6 * (w_ * h_) * math.pi / 4)
        depth = BASE_MM + (RAISE_MM if (~is_dot).any() else DOME_MM)
    return [round(float(x1 - x0), 1), round(float(y1 - y0), 1), round(float(depth), 1)]


def _solid_extrusion(poly, height: float):
    """Extrude a polygon and make sure every piece is a closed, manifold solid (TS-05).

    Holes that touch each other or the outline at a point can give edges shared by 4 faces.
    A tiny morphological close (0.05 mm, far below what a finger can feel) or a 0.1 mm open
    removes those pinch points. Returns one trimesh (possibly several bodies) or None."""
    import trimesh
    from shapely.geometry.polygon import orient

    def extrude(geom):
        ms, ok = [], True
        for gg in getattr(geom, "geoms", [geom]):
            if gg.is_empty or gg.area < 0.05 or gg.geom_type != "Polygon":
                continue
            try:
                m = trimesh.creation.extrude_polygon(orient(gg), height)
            except Exception:
                continue
            ok &= bool(m.is_watertight)
            ms.append(m)
        return (trimesh.util.concatenate(ms) if ms else None), ok

    fixes = (lambda g: g,
             lambda g: g.buffer(0.05, join_style="mitre").buffer(-0.05, join_style="mitre"),
             lambda g: g.buffer(-0.1).buffer(0.1))
    first = None
    for fix in fixes:
        try:
            m, ok = extrude(fix(poly))
        except Exception:
            continue
        if first is None:
            first = m
        if m is not None and ok:
            return m
    return first


def build_relief(ink: np.ndarray, px_per_mm: float):
    """trimesh.Trimesh relief of a binary page raster (see module docstring)."""
    import trimesh
    from shapely.geometry import Polygon

    H, W = ink.shape
    s = 1.0 / px_per_mm
    to_mm = lambda pts: np.column_stack([pts[:, 0] * s, (H - pts[:, 1]) * s])
    meshes_v, meshes_f = [], []
    nv = 0

    def add(v, f):
        nonlocal nv
        meshes_v.append(v)
        meshes_f.append(f + nv)
        nv += len(v)

    ink = (ink > 0).astype(np.uint8) * 255
    n, cc, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    # dot-like components -> domes
    dot_d = 1.5 * px_per_mm
    w_, h_, a_ = st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT], st[:, cv2.CC_STAT_AREA]
    is_dot = (np.maximum(w_, h_) <= dot_d * 1.35) & (np.minimum(w_, h_) >= dot_d * 0.6) & \
             (a_ >= 0.6 * (w_ * h_) * math.pi / 4)
    is_dot[0] = False
    dv, df = _dome_template(0.75, DOME_MM + SINK_MM)
    for i in np.nonzero(is_dot)[0]:
        c = to_mm(cen[i:i + 1])[0]
        add(dv + (c[0], c[1], BASE_MM - SINK_MM), df)
    rest = np.where(is_dot[cc], 0, ink).astype(np.uint8)
    rest[cc == 0] = 0

    # everything else -> extruded polygons with holes
    if rest.any():
        from shapely.geometry.polygon import orient
        from shapely.ops import unary_union
        polys = []
        cs, hier = cv2.findContours(rest, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hier = hier[0]
        for k, c in enumerate(cs):
            if hier[k][3] != -1 or len(c) < 3:
                continue
            shell = to_mm(c.reshape(-1, 2).astype(np.float64))
            holes = []
            ch = hier[k][2]
            while ch != -1:
                if len(cs[ch]) >= 3:
                    holes.append(to_mm(cs[ch].reshape(-1, 2).astype(np.float64)))
                ch = hier[ch][0]
            poly = Polygon(shell, holes)
            # pixel-centre contours are half a pixel inside the raster edge: grow back, simplify
            polys.append(poly.buffer(0.5 * s, join_style="mitre").simplify(0.1))
        # TS-05: union first, so shapes that touch become ONE solid. Extruding touching shapes
        # separately left edges shared by 4 faces (non-manifold STL).
        merged = unary_union([q for q in polys if not q.is_empty]).buffer(0)
        geoms = getattr(merged, "geoms", [merged])
        for g in geoms:
            if g.is_empty or g.area < 0.05:
                continue
            m = _solid_extrusion(g, RAISE_MM + SINK_MM)
            if m is not None:
                add(m.vertices + (0, 0, BASE_MM - SINK_MM), m.faces)

    # base plate: content bbox + margin, clipped to the page
    x0, y0, x1, y1 = _plate_bounds(ink, px_per_mm)
    add(*_box(x0, y0, 0.0, x1, y1, BASE_MM))
    V = np.concatenate(meshes_v, 0)
    F = np.concatenate(meshes_f, 0)
    return trimesh.Trimesh(V, F, process=False)


# ============================================================================ main
def export_all(analysis: Analysis, tactile: TactileResult, outdir: str) -> dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    p = lambda name: os.path.abspath(os.path.join(outdir, name))
    files = {
        "original": p("original.png"), "png": p("tactile.png"), "print_png": p("tactile_print.png"),
        "legend_png": p("legend.png"), "pdf": p("tactile.pdf"), "stl": p("relief.stl"),
    }
    img = analysis.image
    if img is not None and img.ndim == 3:
        cv2.imwrite(files["original"], cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    elif img is not None:
        cv2.imwrite(files["original"], img)
    else:
        cv2.imwrite(files["original"], np.full((10, 10), 255, np.uint8))
    cv2.imwrite(files["png"], 255 - ((tactile.ink > 0).astype(np.uint8) * 255))
    cv2.imwrite(files["print_png"], cv2.cvtColor(tactile.print_layer, cv2.COLOR_RGB2BGR))
    cv2.imwrite(files["legend_png"], cv2.cvtColor(legend_rgb_all(tactile, 1.0), cv2.COLOR_RGB2BGR))

    # PDF: paper size (landscape) at 300 dpi
    page_mm = (tactile.width / tactile.px_per_mm, tactile.height / tactile.px_per_mm)
    size = (int(round(page_mm[0] / 25.4 * DPI)), int(round(page_mm[1] / 25.4 * DPI)))
    page1 = _page_hires(tactile.print_layer, tactile.ink, size)
    key_pages = []
    for i in range(len(legend_pages(tactile))):
        pg = cv2.cvtColor(legend_rgb(tactile, size[0] / tactile.width, i), cv2.COLOR_RGB2GRAY)
        if (pg.shape[1], pg.shape[0]) != size:
            pg = cv2.resize(pg, size, interpolation=cv2.INTER_NEAREST)
        key_pages.append(pg)
    _write_pdf(files["pdf"], [page1] + key_pages, page_mm)

    if os.environ.get("FEELABLE_LAZY_STL") == "1":
        # built on first download instead (ensure_stl) — saves seconds on slow servers
        if os.path.exists(files["stl"]):
            os.remove(files["stl"])          # stale relief from before an edit
    else:
        mesh = build_relief(tactile.ink, tactile.px_per_mm)
        mesh.export(files["stl"])
    return files


def ensure_stl(job_dir: str, px_per_mm: float) -> str:
    """Build relief.stl from the saved tactile.png if it doesn't exist yet. Returns its path."""
    stl = os.path.join(job_dir, "relief.stl")
    if not os.path.exists(stl):
        png = cv2.imread(os.path.join(job_dir, "tactile.png"), cv2.IMREAD_GRAYSCALE)
        if png is None:
            raise FileNotFoundError("tactile.png missing")
        ink = ((png < 128).astype(np.uint8)) * 255
        tmp = stl + ".tmp.stl"
        build_relief(ink, px_per_mm).export(tmp)
        os.replace(tmp, stl)
    return stl
