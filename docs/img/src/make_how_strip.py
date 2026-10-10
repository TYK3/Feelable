"""Render the home page's "How Feelable reads a picture" strip from a REAL engine run.

usage (repo root): python docs/img/src/make_how_strip.py
input: samples/skull-lateral.jpg (OpenStax Anatomy & Physiology, CC BY 4.0)
writes: web/public/how/{1-read,2-follow,3-trace,4-feel}.webp (each a crop of the same region)
"""
import io, pathlib, sys, tempfile

import cv2
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))
from engine import pipeline  # noqa: E402

INK, BANANA, DEEP = (23, 23, 28), (255, 210, 63), (242, 190, 0)


def main() -> None:
    src = ROOT / "samples" / "skull-lateral.jpg"
    out = ROOT / "web" / "public" / "how"
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        r = pipeline.run_job(src.read_bytes(), td + "/job", use_ai=False)
        v = pipeline.vision_trace(td + "/job")
        orig = np.array(Image.open(td + "/job/original.png").convert("RGB"))
        tact = np.array(Image.open(td + "/job/tactile.png").convert("RGB"))
    H, W = orig.shape[:2]
    # a crop around the face, where labels are densest
    x0, y0, x1, y1 = int(0.40 * W), int(0.06 * H), W, int(0.66 * H)
    base = (orig * 0.55 + 255 * 0.45).astype(np.uint8)
    s = max(1, W // 600)

    def save(img, name):
        c = img[y0:y1, x0:x1]
        Image.fromarray(c).resize((720, int(720 * c.shape[0] / c.shape[1])), Image.LANCZOS).save(out / name, quality=82)

    a = base.copy()
    for lb in v["labels"]:
        bx = lb["bbox"]
        ov = a.copy()
        cv2.rectangle(ov, (bx[0] - 2, bx[1] - 1), (bx[2] + 2, bx[3] + 1), BANANA, -1)
        a = cv2.addWeighted(ov, 0.55, a, 0.45, 0)
        cv2.rectangle(a, (bx[0] - 2, bx[1] - 1), (bx[2] + 2, bx[3] + 1), DEEP, s)
    for lb in v["labels"]:
        if lb["raw"]:
            cv2.circle(a, (lb["bbox"][2] + 3, lb["bbox"][1]), 4 * s, INK, -1)
    save(a, "1-read.webp")
    b = a.copy()
    for L in v["leaders"]:
        pts = np.array(L["points"], np.int32)
        if len(pts) >= 2:
            cv2.polylines(b, [pts], False, INK, 2 * s, cv2.LINE_AA)
            cv2.circle(b, tuple(int(t) for t in pts[-1]), 4 * s, BANANA, -1, cv2.LINE_AA)
            cv2.circle(b, tuple(int(t) for t in pts[-1]), 4 * s, INK, s, cv2.LINE_AA)
    save(b, "2-follow.webp")
    c = base.copy()
    meth = {p["id"]: p["match_method"] for p in v["parts"]}
    for p in r["parts"]:
        if not p["include"]:
            continue
        for poly in p["source_polygons"]:
            pts = np.array(poly, np.int32)
            if meth.get(p["id"]) == "carved":
                ov = c.copy()
                cv2.fillPoly(ov, [pts], BANANA)
                c = cv2.addWeighted(ov, 0.35, c, 0.65, 0)
            cv2.polylines(c, [pts], True, INK, 2 * s, cv2.LINE_AA)
    save(c, "3-trace.webp")
    # the raised print: the whole swell-paper page (its layout differs from the photo's)
    Image.fromarray(tact).resize((720, int(720 * tact.shape[0] / tact.shape[1])), Image.LANCZOS).save(out / "4-feel.webp", quality=82)
    print("steps", v["steps"])


if __name__ == "__main__":
    main()
