"""Feelable rough prototype pipeline.

Steps (real image processing, run on a sample textbook diagram):
  1. make_diagram()      - draws a colourful 'textbook' plant-cell diagram (stand-in for a photo)
  2. understand()        - structure JSON (in the real app this comes from a vision-language model;
                           here it is a fixed mock so the demo runs offline)
  3. segment()           - colour-based segmentation + morphology -> one clean mask per part
  4. tactilise()         - tactile design rules: drop clutter, thick smooth outlines,
                           enlarge tiny parts, one texture per part
  5. add_braille()       - UEB grade-1 braille key letters + legend
  6. export_pdf/stl()    - swell-paper PDF and 3D-printable relief STL
Outputs go to ./out
"""
import json, os, time
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
W, H = 1100, 720
FONT = "/usr/share/fonts/opentype/inter/Inter-%s.otf"

def font(size, weight="Regular"):
    return ImageFont.truetype(FONT % weight, size)

PAL = {
    "wall": (40, 100, 40), "cyto": (200, 235, 180), "vacuole": (150, 200, 240),
    "nucleus": (170, 120, 200), "nucleolus": (120, 70, 160),
    "chloro": (90, 190, 80), "grana": (120, 210, 105),
    "mito": (240, 150, 60), "ribo": (90, 90, 90),
}

CHLORO = [((245, 235), (40, 22), 20), ((300, 540), (42, 22), -10), ((520, 545), (40, 21), 8),
          ((626, 395), (22, 40), 0), ((214, 430), (21, 38), 8)]
MITO = [((400, 212), (16, 9), 0), ((640, 482), (15, 9), 30), ((362, 250), (15, 8), -20)]

# ---------------------------------------------------------------- 1. diagram
def make_diagram():
    rng = np.random.default_rng(3)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.text((40, 28), "Fig. 3.2   A typical plant cell", font=font(26, "SemiBold"), fill=(30, 30, 30))
    d.rounded_rectangle((176, 166, 696, 586), 64, fill=(225, 225, 225))  # drop shadow (clutter)
    d.rounded_rectangle((170, 160, 690, 580), 64, fill=PAL["wall"])
    d.rounded_rectangle((184, 174, 676, 566), 52, fill=PAL["cyto"])
    # vacuole: wobbly blob
    t = np.linspace(0, 2 * np.pi, 120)
    r = 1 + 0.06 * np.sin(3 * t) + 0.04 * np.cos(5 * t)
    pts = [(420 + 170 * r[i] * np.cos(t[i]), 400 + 105 * r[i] * np.sin(t[i])) for i in range(len(t))]
    d.polygon(pts, fill=PAL["vacuole"])
    d.ellipse((330, 330, 400, 360), fill=(185, 220, 248))  # highlight (clutter)
    d.ellipse((535, 185, 635, 285), fill=PAL["nucleus"])
    d.ellipse((568, 218, 604, 254), fill=PAL["nucleolus"])
    for (cx, cy), (a, b), ang in CHLORO:
        lay = Image.new("L", (W, H), 0); ld = ImageDraw.Draw(lay)
        ld.ellipse((cx - a, cy - b, cx + a, cy + b), fill=255)
        lay = lay.rotate(ang, center=(cx, cy))
        im.paste(PAL["chloro"], mask=lay)
        for k in range(-2, 3):  # grana stacks (clutter detail)
            gx = cx + k * a * 0.32 if a > b else cx
            gy = cy if a > b else cy + k * b * 0.32
            d.line((gx - 4, gy - 4, gx + 4, gy + 4) if a > b else (gx - 6, gy, gx + 6, gy), fill=PAL["grana"], width=3)
    for (cx, cy), (a, b), ang in MITO:
        lay = Image.new("L", (W, H), 0); ld = ImageDraw.Draw(lay)
        ld.ellipse((cx - a, cy - b, cx + a, cy + b), fill=255)
        im.paste(PAL["mito"], mask=lay.rotate(ang, center=(cx, cy)))
    cy_mask = np.array(im)
    for _ in range(140):  # ribosomes (clutter)
        x, y = rng.integers(195, 665), rng.integers(185, 555)
        if tuple(cy_mask[y, x]) == PAL["cyto"]:
            d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=PAL["ribo"])
    # labels with leader lines
    lab = font(19, "Medium")
    for text, tx, ty, px, py in [
        ("Cell wall", 735, 168, 688, 200), ("Nucleus", 735, 228, 630, 238),
        ("Mitochondrion", 735, 300, 655, 478), ("Chloroplast", 735, 380, 648, 395),
        ("Cytoplasm", 735, 520, 650, 548), ("Vacuole", 40, 395, 300, 400),
        ("Ribosomes", 40, 255, 205, 300)]:
        d.line((tx - 6 if tx > 500 else tx + d.textlength(text, font=lab) + 6, ty + 12, px, py), fill=(40, 40, 40), width=1)
        d.text((tx, ty), text, font=lab, fill=(20, 20, 20))
    d.text((40, 650), "The vacuole is surrounded by the tonoplast. Not to scale.", font=font(15), fill=(120, 120, 120))
    im.save(os.path.join(OUT, "01_diagram.png"))
    return im

# ---------------------------------------------------------------- 2. understanding (mocked VLM)
STRUCTURE = {
    "diagram_type": "biology / cell structure",
    "title": "plant cell",
    "parts": [
        {"id": "w", "name": "cell wall", "say": "Cell wall. A tough outer layer that gives the cell its shape.", "box": [170, 160, 690, 580]},
        {"id": "n", "name": "nucleus", "say": "Nucleus. Controls the cell and holds its DNA.", "box": [535, 185, 635, 285]},
        {"id": "v", "name": "vacuole", "say": "Vacuole. A large sac of water that keeps the cell firm.", "box": [245, 285, 600, 515]},
        {"id": "c", "name": "chloroplast", "say": "Chloroplast. Uses sunlight to make food. There are five.", "box": [200, 200, 665, 570]},
        {"id": "m", "name": "mitochondrion", "say": "Mitochondrion. Releases energy for the cell.", "box": [315, 195, 660, 495]},
    ],
    "drop": ["ribosome dots (too small to feel)", "shading & highlights", "drop shadow", "figure caption"],
}

def understand():
    with open(os.path.join(OUT, "02_structure.json"), "w") as f:
        json.dump(STRUCTURE, f, indent=2)
    return STRUCTURE

# ---------------------------------------------------------------- 3. segmentation
def segment(im):
    a = np.array(im).astype(np.int32)
    keys = ["wall", "cyto", "vacuole", "nucleus", "nucleolus", "chloro", "grana", "mito"]
    cols = np.array([PAL[k] for k in keys])
    dist = np.linalg.norm(a[:, :, None, :] - cols[None, None], axis=3)
    lab = dist.argmin(2); ok = dist.min(2) < 45
    m = lambda *ks: (np.isin(lab, [keys.index(k) for k in ks]) & ok).astype(np.uint8) * 255
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    def clean(mask, min_area=120, fill=True):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
        n, cc, st, _ = cv2.connectedComponentsWithStats(mask)
        outm = np.zeros_like(mask)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= min_area:
                outm[cc == i] = 255
        if fill:
            cs, _ = cv2.findContours(outm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(outm, cs, -1, 255, -1)
        return outm
    wall = clean(m("wall"), 2000, fill=False)
    cell = clean(m("wall", "cyto", "vacuole", "nucleus", "nucleolus", "chloro", "grana", "mito") | 0, 5000)
    cell = cv2.morphologyEx(cell, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cs, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cell = np.zeros_like(cell); cv2.drawContours(cell, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    inner = cv2.erode(cell, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27)))
    masks = {
        "w": wall & cv2.bitwise_not(inner) & cell,
        "cell": cell, "inner": inner,
        "v": clean(m("vacuole"), 3000),
        "n": clean(m("nucleus", "nucleolus"), 1500),
        "c": clean(m("chloro", "grana"), 500),
        "m": clean(m("mito"), 80),
    }
    for k, v in masks.items():
        cv2.imwrite(os.path.join(OUT, f"03_mask_{k}.png"), v)
    return masks

# ---------------------------------------------------------------- 4. tactile rules
def hatch(shape, spacing, angle=45, width=3):
    h, w = shape
    img = np.zeros((h, w), np.uint8)
    diag = int(np.hypot(h, w))
    for off in range(-diag, diag, spacing):
        if angle == 45:
            cv2.line(img, (off, 0), (off + h, h), 255, width)
        else:
            cv2.line(img, (0, off), (w, off), 255, width)
    return img

def dots(shape, spacing, r):
    img = np.zeros(shape, np.uint8)
    for y in range(spacing // 2, shape[0], spacing):
        for x in range(spacing // 2 + (spacing // 2 if (y // spacing) % 2 else 0), shape[1], spacing):
            cv2.circle(img, (x, y), r, 255, -1)
    return img

def smooth_contours(mask, eps=1.5):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [cv2.approxPolyDP(c, eps, True) for c in cs]

def enlarge(mask, factor):
    """Scale each component about its own centre (tactile minimum-size rule)."""
    outm = np.zeros_like(mask)
    n, cc, st, cen = cv2.connectedComponentsWithStats(mask)
    for i in range(1, n):
        comp = (cc == i).astype(np.uint8) * 255
        M = cv2.getRotationMatrix2D(tuple(cen[i]), 0, factor)
        outm |= cv2.warpAffine(comp, M, (mask.shape[1], mask.shape[0]))
    return outm

def tactilise(masks):
    ink = np.zeros((H, W), np.uint8)  # 255 = raised
    m = dict(masks)
    m["m"] = enlarge(masks["m"], 1.7)          # tiny parts -> min tactile size
    m["c"] = enlarge(masks["c"], 1.1)
    # keep a 6px gap around inner parts so textures don't touch outlines
    gap = lambda k: cv2.dilate(m[k], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    # cell wall = double line
    cv2.drawContours(ink, smooth_contours(m["cell"]), -1, 255, 9)
    cv2.drawContours(ink, smooth_contours(m["inner"]), -1, 255, 4)
    # vacuole: outline + dot texture (cut away where other parts sit)
    vac_tex = dots((H, W), 17, 3) & cv2.erode(m["v"], np.ones((15, 15), np.uint8))
    for k in ("n", "c", "m"):
        vac_tex &= cv2.bitwise_not(gap(k))
    ink |= vac_tex
    cv2.drawContours(ink, smooth_contours(m["v"]), -1, 255, 5)
    # nucleus: outline + horizontal lines
    ink |= hatch((H, W), 9, angle=0, width=3) & cv2.erode(m["n"], np.ones((9, 9), np.uint8))
    cv2.drawContours(ink, smooth_contours(m["n"]), -1, 255, 5)
    # chloroplasts: outline + diagonal lines
    ink |= hatch((H, W), 9, 45, 3) & cv2.erode(m["c"], np.ones((7, 7), np.uint8))
    cv2.drawContours(ink, smooth_contours(m["c"]), -1, 255, 4)
    # mitochondria: solid
    ink |= m["m"]
    ink = cv2.GaussianBlur(ink, (3, 3), 0)
    ink = (ink > 110).astype(np.uint8) * 255
    tactile_masks = {"w": m["cell"] & cv2.bitwise_not(m["inner"]), "n": m["n"], "v": m["v"], "c": m["c"], "m": m["m"]}
    return ink, tactile_masks

# ---------------------------------------------------------------- 5. braille (UEB grade 1)
BR = {'a': '1', 'b': '12', 'c': '14', 'd': '145', 'e': '15', 'f': '124', 'g': '1245', 'h': '125', 'i': '24',
      'j': '245', 'k': '13', 'l': '123', 'm': '134', 'n': '1345', 'o': '135', 'p': '1234', 'q': '12345',
      'r': '1235', 's': '234', 't': '2345', 'u': '136', 'v': '1236', 'w': '2456', 'x': '1346', 'y': '13456',
      'z': '1356', ' ': ''}
DOTPOS = {'1': (0, 0), '2': (0, 1), '3': (0, 2), '4': (1, 0), '5': (1, 1), '6': (1, 2)}
DOT_R, DOT_STEP, CELL_STEP = 4, 11, 23   # ~ 2.5 mm dot pitch, 6 mm cell pitch at the print scale

def braille_unicode(text):
    out = ""
    for ch in text.lower():
        bits = 0
        for dgt in BR.get(ch, ''):
            bits |= 1 << (int(dgt) - 1)
        out += chr(0x2800 + bits)
    return out

def draw_braille(ink, text, x, y):
    for i, ch in enumerate(text.lower()):
        for dgt in BR.get(ch, ''):
            cx, cy = DOTPOS[dgt]
            cv2.circle(ink, (x + i * CELL_STEP + cx * DOT_STEP, y + cy * DOT_STEP), DOT_R, 255, -1)
    return len(text) * CELL_STEP

KEY_POS = {"w": (195, 120), "n": (560, 300), "v": (410, 455), "c": (283, 222), "m": (436, 196)}

def add_braille(ink, tmasks, structure):
    ink = ink.copy()
    plain = ink.copy()
    # clear a small white pad around each key so dots stand alone
    for p in structure["parts"]:
        x, y = KEY_POS[p["id"]]
        cv2.rectangle(ink, (x - 8, y - 8), (x + 22, y + 32), 0, -1)
        draw_braille(ink, p["id"], x, y)
    # title + legend
    draw_braille(ink, "plant cell", 40, 40)
    ly = 150
    legend = []
    for p in structure["parts"]:
        draw_braille(ink, p["id"], 725, ly)
        cv2.rectangle(ink, (725 + CELL_STEP + 6, ly + 10), (725 + CELL_STEP + 20, ly + 13), 255, -1)
        draw_braille(ink, p["name"].split()[-1] if p["name"] == "cell wall" else p["name"], 725 + CELL_STEP + 30, ly)
        legend.append((p, ly))
        ly += 78
    return ink, plain, legend

def render_tactile_rgb(ink, legend=None, show_print=True):
    im = Image.new("RGB", (W, H), "white")
    arr = np.array(im); arr[ink > 0] = (20, 20, 22)
    im = Image.fromarray(arr)
    if show_print and legend:
        d = ImageDraw.Draw(im)
        f = font(15)
        d.text((40, 76), "plant cell  (print for the teacher, not raised)", font=f, fill=(150, 150, 160))
        for p, ly in legend:
            d.text((725 + CELL_STEP + 30, ly + 32), f"{p['id']} = {p['name']}", font=f, fill=(150, 150, 160))
    return im

# ---------------------------------------------------------------- 6. exports
def export_pdf(tactile_rgb):
    page = Image.new("RGB", (2480, 3508), "white")  # A4 @ 300 dpi
    t = tactile_rgb.resize((2300, int(2300 * H / W)), Image.LANCZOS)
    page.paste(t, (90, 400))
    d = ImageDraw.Draw(page)
    d.text((90, 150), "Feelable - swell paper print (black = raised)", font=font(60, "SemiBold"), fill=(0, 0, 0))
    page.save(os.path.join(OUT, "swell-paper.pdf"), resolution=300)

def heightmap(ink, scale=0.25):
    h = cv2.resize(ink, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return cv2.GaussianBlur(h, (3, 3), 0)

def export_stl(ink, mm_per_px=0.2, scale=0.25, base=2.0, relief=1.2):
    import trimesh
    h = heightmap(ink, scale)
    gh, gw = h.shape
    step = mm_per_px / scale
    z = base + relief * h
    xs, ys = np.meshgrid(np.arange(gw) * step, (gh - 1 - np.arange(gh)) * step)
    top = np.stack([xs, ys, z], -1).reshape(-1, 3)
    bot = np.stack([xs, ys, np.zeros_like(z)], -1).reshape(-1, 3)
    V = np.vstack([top, bot]); off = gh * gw
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    F = [np.stack([a, c, b], 1), np.stack([b, c, d], 1),
         np.stack([a + off, b + off, c + off], 1), np.stack([b + off, d + off, c + off], 1)]
    def wall(edge, flip):
        e0, e1 = edge[:-1], edge[1:]
        t1 = np.stack([e0, e1, e0 + off], 1); t2 = np.stack([e1, e1 + off, e0 + off], 1)
        return [t1[:, ::-1], t2[:, ::-1]] if flip else [t1, t2]
    F += wall(idx[0, :], False) + wall(idx[-1, :], True) + wall(idx[:, 0], True) + wall(idx[:, -1], False)
    mesh = trimesh.Trimesh(V, np.vstack(F), process=True)
    mesh.fix_normals()
    mesh.export(os.path.join(OUT, "relief.stl"))
    return mesh

def run():
    t0 = time.time()
    im = make_diagram()
    t1 = time.time()
    st = understand()
    masks = segment(im)
    ink, tmasks = tactilise(masks)
    final, plain, legend = add_braille(ink, tmasks, st)
    trgb = render_tactile_rgb(final, legend)
    trgb.save(os.path.join(OUT, "05_tactile.png"))
    render_tactile_rgb(plain).save(os.path.join(OUT, "04_tactile_nobraille.png"))
    cv2.imwrite(os.path.join(OUT, "05_ink.png"), final)
    export_pdf(trgb)
    mesh = export_stl(final)
    t2 = time.time()
    np.savez_compressed(os.path.join(OUT, "masks.npz"), **{k: v for k, v in masks.items()},
                        **{"t_" + k: v for k, v in tmasks.items()})
    with open(os.path.join(OUT, "legend.json"), "w") as f:
        json.dump([[p["id"], ly] for p, ly in legend], f)
    print(f"pipeline (excl. drawing input): {t2 - t1:.2f}s | mesh {len(mesh.faces)} faces watertight={mesh.is_watertight}")
    print("braille title:", braille_unicode("plant cell"))
    return t2 - t1

if __name__ == "__main__":
    run()
