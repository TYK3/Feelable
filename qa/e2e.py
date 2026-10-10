"""Feelable end-to-end QA (real server + real browser).

Usage:  ./run.sh &   then   python3 qa/e2e.py [base_url]
Writes screenshots to qa/artifacts/ and prints a PASS/FAIL table. Exit code 1 on any failure.
"""
import io
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "qa", "artifacts")
os.makedirs(ART, exist_ok=True)
results: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""), flush=True)


def get(url):
    with urllib.request.urlopen(BASE + url) as r:
        return r.status, r.headers, r.read()


def interior_point(job, part):
    """A point well inside the part's tactile polygons and outside every other (smaller) part."""
    W, H = job["tactile"]["width"], job["tactile"]["height"]
    m = np.zeros((H, W), np.uint8)
    for poly in part["tactile_polygons"]:
        cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    for q in job["parts"]:
        if q["id"] != part["id"] and q["include"] and q["area"] < part["area"]:
            for poly in q["tactile_polygons"]:
                cv2.fillPoly(m, [np.array(poly, np.int32)], 0)
    if not m.any():
        return None
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return int(x), int(y)


manifest = json.load(open(os.path.join(ROOT, "samples", "manifest.json")))

with sync_playwright() as p:
    browser = p.chromium.launch()
    desk = browser.new_context(viewport={"width": 1440, "height": 900})
    page = desk.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # 1. home
    page.goto(BASE + "/#/")
    page.wait_for_selector("text=Feelable", timeout=10000)
    page.wait_for_timeout(800)
    page.screenshot(path=f"{ART}/01_home.png")
    check("home loads with sample gallery", page.locator("img").count() >= 4)

    # 2. upload every sample through the real file input
    jobs = {}
    for s in manifest:
        page.goto(BASE + "/#/")
        page.wait_for_timeout(500)
        t = time.time()
        page.locator('input[type="file"]').first.set_input_files(os.path.join(ROOT, "samples", s["file"]))
        try:
            page.wait_for_url("**/#/job/**", timeout=60000)
            jid = page.url.split("#/job/")[1].split("?")[0]
            page.wait_for_timeout(1200)
            job = json.loads(get(f"/api/jobs/{jid}")[2])
            jobs[s["id"]] = job
            inc = [q for q in job["parts"] if q["include"]]
            names = " ".join(q["name"].replace(" ", "") for q in inc)
            hit = sum(1 for e in s["expected_parts"] if e.replace(" ", "") in names)
            check(f"upload {s['id']}", len(inc) >= s["min_parts"] and hit >= 0.6 * len(s["expected_parts"]),
                  f"{len(inc)} parts, {hit}/{len(s['expected_parts'])} names, {time.time() - t:.1f}s")
        except Exception as e:  # noqa
            check(f"upload {s['id']}", False, str(e)[:120])
    page.goto(BASE + f"/#/job/{jobs['plant-cell']['id']}")
    page.wait_for_timeout(1500)
    page.screenshot(path=f"{ART}/02_review_plant.png", full_page=True)

    # 3. downloads
    hj = jobs["heart"]
    st, hd, body = get(hj["files"]["pdf"])
    pages = body.count(b"/Type /Page") - body.count(b"/Type /Pages")
    check("PDF download", st == 200 and body[:4] == b"%PDF" and pages == 2, f"{pages} pages, {len(body)//1024} KB")
    st, hd, body = get(hj["files"]["stl"])
    import trimesh
    mesh = trimesh.load(io.BytesIO(body), file_type="stl")
    check("STL download", st == 200 and mesh.is_watertight, f"{len(mesh.faces)} faces, watertight={mesh.is_watertight}")

    # 4. edit -> apply -> reload persists
    page.goto(BASE + f"/#/job/{hj['id']}")
    page.wait_for_timeout(1500)
    first = page.locator("input.part__name").first
    old = first.input_value()
    first.fill("left heart chamber")
    page.get_by_role("button", name="Apply changes").click()
    page.wait_for_timeout(4000)
    page.reload()
    page.wait_for_timeout(1500)
    names = [q["name"] for q in json.loads(get(f"/api/jobs/{hj['id']}")[2])["parts"]]
    check("rename part persists", "left heart chamber" in names, f"was {old!r}")
    page.screenshot(path=f"{ART}/03_review_heart_edited.png", full_page=True)

    # 5. error states
    page.goto(BASE + "/#/job/0000000000")
    page.wait_for_timeout(1500)
    check("unknown job shows friendly page", page.locator("text=/not found|can.t find|couldn.t find/i").count() > 0)
    page.screenshot(path=f"{ART}/04_notfound.png")
    req = urllib.request.Request(BASE + "/api/jobs", method="POST",
                                 data=b"--x\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.txt\"\r\n"
                                      b"Content-Type: text/plain\r\n\r\nhello\r\n--x--\r\n",
                                 headers={"Content-Type": "multipart/form-data; boundary=x"})
    try:
        urllib.request.urlopen(req)
        check("non-image upload rejected", False)
    except urllib.error.HTTPError as e:
        check("non-image upload rejected", e.code == 400, json.loads(e.read())["detail"])
    check("no page errors (desktop)", not errors, "; ".join(errors)[:200])

    # 6. touch mode: every part speaks its own name (phone + desktop hover)
    phone = browser.new_context(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
    for ctx_name, ctx in (("phone", phone), ("desktop", desk)):
        for sid in ("plant-cell", "animal-cell", "bar-chart", "earth-layers"):
            job = json.loads(get(f"/api/jobs/{jobs[sid]['id']}")[2])
            tp = ctx.new_page()
            perr = []
            tp.on("pageerror", lambda e: perr.append(str(e)))
            tp.goto(BASE + f"/#/touch/{job['id']}")
            tp.wait_for_selector("text=Tap anywhere to start", timeout=10000)
            tp.locator("text=Tap anywhere to start").click()
            tp.wait_for_timeout(1500)
            ok, miss = 0, []
            inc = [q for q in job["parts"] if q["include"]]
            for q in inc:
                pt = interior_point(job, q)
                xy = tp.evaluate(f"window.__feelableMap && window.__feelableMap({pt[0]},{pt[1]})") if pt else None
                if not xy:
                    miss.append(q["name"])
                    continue
                tp.mouse.move(10, 10)
                tp.wait_for_timeout(150)
                tp.mouse.move(xy[0], xy[1], steps=4)
                tp.wait_for_timeout(350)
                said = (tp.evaluate("window.__feelableLastSpoken") or "").lower()
                if q["name"].lower() in said:
                    ok += 1
                else:
                    miss.append(f"{q['name']}→{said[:30]!r}")
            tp.screenshot(path=f"{ART}/05_touch_{ctx_name}_{sid}.png")
            check(f"touch {ctx_name} {sid}", ok == len(inc) and not perr, f"{ok}/{len(inc)} " + ", ".join(miss)[:150])
            tp.close()
    browser.close()

passed = sum(r[1] for r in results)
print(f"\n{passed}/{len(results)} checks passed")
with open(os.path.join(ROOT, "qa", "last_run.json"), "w") as f:
    json.dump([{"check": n, "passed": ok, "detail": d} for n, ok, d in results], f, indent=1)
sys.exit(0 if passed == len(results) else 1)
