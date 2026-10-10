"""Live smoke test for the Gemini integration. Run it wherever GEMINI_API_KEY is set.

    cd backend && python scripts/ai_smoke.py ../samples/heart.png            # diagram: analyze + enrich
    cd backend && python scripts/ai_smoke.py photo.jpg --object              # photo: describe_object
    cd backend && python scripts/ai_smoke.py img.png --all                   # enrich, describe_parts, describe_object

Prints what the AI functions return plus timings. It never prints the API key.
Exit code 0 if every call returned a result, 1 otherwise.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from engine import ai  # noqa: E402


def _load(path: str) -> np.ndarray:
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im)


def _timed(label, fn):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    status = "ok" if out is not None else f"FAILED: {ai.last_error()}"
    print(f"\n== {label}: {status} ({dt:.1f} s)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--object", action="store_true", help="only run describe_object (photo of a thing)")
    ap.add_argument("--all", action="store_true", help="run enrich, describe_parts and describe_object")
    ap.add_argument("--model", help="override GEMINI_MODEL for this run")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.model:
        os.environ["GEMINI_MODEL"] = args.model
    info = ai.get_provider_info()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    print(f"provider={info['provider']} model={info['model']} fallback={ai.FALLBACK_MODEL} "
          f"key={'set (' + str(len(key)) + ' chars)' if key else 'NOT SET'}")
    if not info["provider"]:
        print("Set GEMINI_API_KEY first.")
        return 1

    img = _load(args.image)
    print(f"image {args.image}: {img.shape[1]}x{img.shape[0]}")
    ok = True

    if not args.object:
        from engine.analyze import analyze
        t = time.perf_counter()
        an = analyze(img, use_ai=False)
        print(f"offline analyze: {len(an.parts)} parts, title={an.title!r} ({time.perf_counter() - t:.1f} s)")
        for p in an.parts:
            print(f"  {p.id:4} {p.name!r:28} {p.description}")
        if an.parts:
            enr = _timed("enrich", lambda: ai.enrich(an.image, an))
            ok &= enr is not None
            if enr:
                print(f"title={enr.title!r} type={enr.diagram_type!r}")
                for it in enr.parts:
                    print(f"  {it['id']:4} {it['name']!r:28} drop={it['drop']!s:5} {it['description']}")
            if args.all:
                ids = [p.id for p in an.parts[:4]]
                d = _timed(f"describe_parts {ids}", lambda: ai.describe_parts(an.image, an, ids))
                ok &= d is not None
                for k, v in (d or {}).items():
                    print(f"  {k:4} {v}")
        else:
            print("no parts found offline, skipping enrich")

    if args.object or args.all:
        od = _timed("describe_object", lambda: ai.describe_object(img))
        ok &= od is not None
        if od:
            print(f"title={od.title!r} summary={od.summary!r}")
            for p in od.parts:
                print(f"  {p.name!r:22} box={p.box} {p.description}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
