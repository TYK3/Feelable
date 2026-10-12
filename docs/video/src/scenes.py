import json, sys, urllib.request
from playwright.sync_api import sync_playwright
from rec import Rec, CURSOR_JS, TOUCH_JS

B = "http://127.0.0.1:8765"
OUT = "/tmp/claude-0/vid/rec"
ROOT = "/home/claude/feelable"
only = set(sys.argv[1:])


def api(path, method="GET"):
    req = urllib.request.Request(B + path, method=method)
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def glide(pg, x, y, steps=22):
    pg.mouse.move(x, y, steps=steps)


def click_el(pg, loc, wait=500):
    loc.scroll_into_view_if_needed()
    bb = loc.bounding_box()
    glide(pg, bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
    pg.wait_for_timeout(150)
    pg.mouse.down(); pg.mouse.up()
    pg.wait_for_timeout(wait)


def open_sample(pg, title):
    pg.goto(B)
    pg.wait_for_selector(".gallery .sample", timeout=60000)
    if not pg.locator(".sample", has_text=title).count():
        pg.locator(".gallery__more").click()
    pg.locator(".sample", has_text=title).first.click()
    pg.wait_for_url("**/#/job/**", timeout=120000)
    pg.wait_for_selector(".ready", timeout=120000)
    pg.wait_for_timeout(700)
    return pg.url.split("/job/")[1].split("?")[0]


with sync_playwright() as p:
    br = p.chromium.launch()
    ctx = br.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1)
    ctx.add_init_script(CURSOR_JS)

    def scene(name):
        return not only or name in only

    # ---------------------------------------------------------------- home + gallery
    if scene("home"):
        pg = ctx.new_page(); pg.goto(B); pg.wait_for_selector(".gallery .sample"); pg.mouse.move(800, 450)
        pg.wait_for_timeout(800)
        r = Rec(pg, "home", OUT, 1600, 900); r.start()
        r.cap("Feelable turns a diagram into something a blind student can feel and hear")
        pg.wait_for_timeout(1800)
        for chip in ["Stem", "Leaf", "Centre"]:
            click_el(pg, pg.locator("button", has_text=chip).first, 900)
        r.cap("Ten reviewed lessons: five basic, five complex")
        pg.evaluate("document.querySelector('.gallery').scrollIntoView({behavior:'smooth', block:'start'})")
        pg.wait_for_timeout(1600)
        click_el(pg, pg.locator(".gallery__filter button", has_text="Basic"), 1600)
        click_el(pg, pg.locator(".gallery__filter button", has_text="Complex"), 1800)
        click_el(pg, pg.locator(".gallery__filter button", has_text="All"), 800)
        r.stop(); pg.close()

    # ---------------------------------------------------------------- upload a new photo
    if scene("upload"):
        pg = ctx.new_page(); pg.goto(B); pg.wait_for_selector(".gallery .sample"); pg.mouse.move(800, 450)
        r = Rec(pg, "upload", OUT, 1600, 900); r.start()
        r.cap("A teacher takes a photo of a worksheet", "Water cycle, photographed on a desk")
        glide(pg, 285, 491); pg.wait_for_timeout(500)
        with pg.expect_file_chooser() as fc:
            pg.mouse.down(); pg.mouse.up()
        fc.value.set_files(f"{ROOT}/qa/precision/inputs/water-cycle-fresh.jpg")
        r.cap("Real processing, not sped up", "Read labels → find parts → tactile rules → braille, PDF and 3D")
        pg.wait_for_url("**/#/job/**", timeout=120000)
        pg.wait_for_selector(".ready", timeout=120000)
        pg.wait_for_timeout(600)
        r.cap("Evaporation, condensation, precipitation and collection become arrows you can follow",
              "Every part gets its own texture and a braille label")
        h = pg.locator(".viewer__handle"); bb = h.bounding_box(); cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        glide(pg, cx, cy); pg.mouse.down(); glide(pg, cx - 230, cy, 40); glide(pg, cx + 260, cy, 60); glide(pg, cx + 40, cy, 30); pg.mouse.up()
        pg.wait_for_timeout(1500)
        r.stop(); pg.close()

    # ---------------------------------------------------------------- skull: slider + scan details
    if scene("skull"):
        pg = ctx.new_page(); jid = open_sample(pg, "Skull"); pg.mouse.move(800, 450)
        r = Rec(pg, "skull", OUT, 1600, 900); r.start()
        r.cap("The hard case: a dense textbook skull with 27 labelled parts")
        pg.wait_for_timeout(1200)
        h = pg.locator(".viewer__handle"); bb = h.bounding_box(); cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        glide(pg, cx, cy); pg.mouse.down(); glide(pg, cx - 200, cy, 40); glide(pg, cx + 250, cy, 60); glide(pg, cx - 60, cy, 30); pg.mouse.up()
        r.cap("Original on the left, tactile on the right", "Textures for each bone, sutures as raised lines, braille beside every part")
        pg.wait_for_timeout(1800)
        r.cap("Scan details: what the image AI saw, step by step")
        click_el(pg, pg.locator("button", has_text="Scan details").first, 1500)
        steps = pg.locator(".vision__steps button")
        subs = ["Every label read with neural OCR, blurry words re-read up close",
                "Every pointer line followed from its label to its tip",
                "Bones split along their real edges, even when they share a colour",
                "Sutures traced as the border two bones share"]
        for i in range(min(steps.count(), 4)):
            r.cap("Scan details: what the image AI saw, step by step", subs[i] if i < len(subs) else "")
            click_el(pg, steps.nth(i), 2300)
        r.stop(); pg.close()

    # ---------------------------------------------------------------- fix: teacher stays in control
    if scene("fix"):
        pg = ctx.new_page(); jid = open_sample(pg, "Skull"); pg.mouse.move(800, 450)
        r = Rec(pg, "fix", OUT, 1600, 900); r.start()
        r.cap("The teacher stays in charge", "Anything uncertain is listed first, with a fix for each")
        click_el(pg, pg.locator("#step-fix"), 1800)
        r.cap("Draw a missing part on the photo", "Click inside it and Feelable suggests the outline")
        click_el(pg, pg.get_by_role("button", name="Add missing part"), 900)
        pg.wait_for_selector(".draw__canvas", timeout=10000)
        pg.locator(".edlg input").first.fill("frontal bone")
        pg.wait_for_timeout(400)
        xy = pg.evaluate("""([x, y]) => { const s = document.querySelector('.draw__canvas');
            const pt = s.createSVGPoint(); pt.x = x; pt.y = y; const q = pt.matrixTransform(s.getScreenCTM()); return [q.x, q.y]; }""", [760, 190])
        glide(pg, *xy); pg.mouse.down(); pg.mouse.up(); pg.wait_for_timeout(600)
        click_el(pg, pg.get_by_role("button", name="Find outline"), 300)
        pg.wait_for_function("document.querySelector('.draw__status').textContent.includes('Suggested') || document.querySelector('.draw__status').textContent.includes(\"couldn't\")", timeout=30000)
        pg.wait_for_timeout(1600)
        r.cap("Zoom in to check the edge", "Students only see a lesson after the teacher approves it")
        click_el(pg, pg.get_by_role("button", name="Zoom in"), 900)
        click_el(pg, pg.get_by_role("button", name="Zoom in"), 1500)
        click_el(pg, pg.get_by_role("button", name="Fit"), 900)
        click_el(pg, pg.get_by_role("button", name="Cancel"), 700)
        r.stop(); pg.close()

    # ---------------------------------------------------------------- structure: flowchart + map
    if scene("graphs"):
        pg = ctx.new_page(); jid = open_sample(pg, "Plant watering"); pg.mouse.move(800, 450)
        r = Rec(pg, "graphs", OUT, 1600, 900); r.start()
        r.cap("Not just shapes: structure", "A flowchart becomes boxes and arrows a student can follow")
        pg.wait_for_timeout(1200)
        chip = pg.locator(".glance__chip", has_text="Soil dry")
        if chip.count():
            click_el(pg, chip.first, 600)
            pg.locator(".peek").first.scroll_into_view_if_needed()
        r.cap("“Soil dry? A decision. Yes: open valve. No: log reading.”", "What the student hears when they touch the diamond")
        pg.wait_for_timeout(3200)
        r.stop()
        jid2 = open_sample(pg, "bus routes"); pg.mouse.move(800, 450)
        r = Rec(pg, "graphs2", OUT, 1600, 900); r.start()
        r.cap("A bus map becomes stations and routes", "Dashed and solid routes stay distinct by touch, not by colour")
        pg.wait_for_timeout(1000)
        chip = pg.locator(".glance__chip", has_text="Central")
        if chip.count():
            click_el(pg, chip.first, 600)
            pg.locator(".peek").first.scroll_into_view_if_needed()
        r.cap("“Central: an interchange”", "Feelable works out where routes meet from the picture itself")
        pg.wait_for_timeout(3000)
        r.stop(); pg.close()

    # ---------------------------------------------------------------- share + 3D
    if scene("share"):
        pg = ctx.new_page(); jid = open_sample(pg, "Animal cell"); pg.mouse.move(800, 450)
        r = Rec(pg, "share", OUT, 1600, 900); r.start()
        r.cap("Approve and publish", "Then print it, 3D-print it, or hand the student a QR code")
        btn = pg.get_by_role("button", name="Approve and publish").first
        click_el(pg, btn, 1500)
        click_el(pg, pg.locator("#step-share"), 1600)
        pg.mouse.move(1150, 600, steps=10)
        for _ in range(6):
            pg.mouse.wheel(0, 90); pg.wait_for_timeout(120)
        pg.wait_for_timeout(1800)
        pg.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})"); pg.wait_for_timeout(700)
        r.cap("A 3D-printable relief of the same diagram")
        click_el(pg, pg.locator("#step-preview"), 700)
        click_el(pg, pg.locator("button", has_text="3D").first, 2500)
        cv = pg.locator(".viewer canvas").first
        if cv.count():
            bb = cv.bounding_box(); cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
            glide(pg, cx, cy); pg.mouse.down(); glide(pg, cx + 180, cy + 40, 50); glide(pg, cx - 120, cy - 30, 50); pg.mouse.up()
        pg.wait_for_timeout(1500)
        r.stop(); pg.close()
        open(f"{OUT}/cell_job.txt", "w").write(jid)

    # ---------------------------------------------------------------- phone: touch mode
    if scene("phone"):
        jid = open(f"{OUT}/cell_job.txt").read().strip() if "share" not in only and not (not only) else None
        pc = br.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        pc.add_init_script(TOUCH_JS)
        pg = pc.new_page()
        if not jid:
            jid = open(f"{OUT}/cell_job.txt").read().strip()
        job = api(f"/api/jobs/{jid}")
        pg.goto(f"{B}/#/touch/{jid}")
        pg.wait_for_selector(".ts-t-start, .ts-t-explore-shell", timeout=60000)
        r = Rec(pg, "phone", OUT, 780, 1688); r.start()
        pg.wait_for_timeout(1500)
        if pg.locator(".ts-t-start-btn").count():
            bb = pg.locator(".ts-t-start-btn").bounding_box()
            pg.touchscreen.tap(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        pg.wait_for_selector(".ts-t-explore", timeout=30000)
        pg.wait_for_timeout(1500)
        ok = pg.get_by_role("button", name="OK")
        if ok.count():
            bb = ok.first.bounding_box(); pg.touchscreen.tap(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)
        pg.wait_for_timeout(900)
        for x, y, w in [(122, 405, 2000), (170, 382, 2000), (300, 368, 2000), (272, 466, 2000), (31, 400, 2000), (237, 430, 2200)]:
            pg.touchscreen.tap(x, y); pg.wait_for_timeout(w)
        pg.touchscreen.tap(300, 368); pg.wait_for_timeout(120); pg.touchscreen.tap(300, 368)
        pg.wait_for_timeout(4200)
        r.stop(0.2); pc.close()
    br.close()
