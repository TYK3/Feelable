---
title: Feelable
emoji: 🍌
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

<p align="center">
  <img src="docs/img/banner.png" alt="Feelable: Every diagram, feelable. A child's finger traces a raised plant-cell diagram while a phone says the part's name." width="100%">
</p>

<p align="center">
  <a href="https://feelable.onrender.com"><img alt="Live demo" src="https://img.shields.io/badge/live%20demo-feelable.onrender.com-FFD23F?style=for-the-badge&labelColor=17171C"></a>
  <a href="https://github.com/TYK3/feelable/actions/workflows/ci.yml"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/TYK3/feelable/ci.yml?branch=main&style=for-the-badge&label=tests&labelColor=17171C"></a>
  <img alt="Banana Hacks 2026" src="https://img.shields.io/badge/Banana%20Hacks-2026-FFD23F?style=for-the-badge&labelColor=17171C">
  <img alt="Image AI: OCR, segmentation, Gemini vision" src="https://img.shields.io/badge/image%20AI-OCR%20%2B%20segmentation%20%2B%20Gemini-1E9650?style=for-the-badge&labelColor=17171C">
</p>

# Feelable: *every diagram, feelable*

**A teacher photographs a textbook diagram. Feelable turns it into a picture a blind student can feel and hear.**

From one photo it makes three things:

1. **A swell-paper PDF.** Print it, heat it, and the black lines rise. Every part has its own texture and a braille label, and page 2 is a braille key.
2. **A 3D-printable relief** (a watertight STL) of the same diagram.
3. **A talking touch mode** on the student's phone. Touch a part to hear its name, double-tap to hear more, and feel a buzz at each edge. A "Find the…" game guides a finger to a part with hotter/colder beeps.

The teacher checks every part before a student sees it.

| | |
|---|---|
| **Live demo** | https://feelable.onrender.com (free server: the first visit after a quiet spell takes about 50 s to wake up; prepared lessons open at once; a new photo takes about 15–40 s) |
| **Code** | https://github.com/TYK3/feelable |
| **Write-up** | [`DEVPOST.md`](DEVPOST.md) |
| **Demo script (3 min)** | [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) |
| **Latest fix report** | [`docs/FIX-STATUS-2026-10-12-precision.md`](docs/FIX-STATUS-2026-10-12-precision.md) |

Built for **Banana Hacks 2026**, the generative and image AI hackathon (online, 9–12 October 2026).

---

## Contents

1. [Judges: a 2-minute tour](#1-judges-a-2-minute-tour)
2. [The problem and who it is for](#2-the-problem-and-who-it-is-for)
3. [How it works: image AI from start to finish](#3-how-it-works-image-ai-from-start-to-finish)
4. [Measured results](#4-measured-results)
5. [How Feelable meets each judging criterion](#5-how-feelable-meets-each-judging-criterion)
6. [Award categories we are eligible for](#6-award-categories-we-are-eligible-for)
7. [Hackathon submission checklist](#7-hackathon-submission-checklist)
8. [Run it yourself](#8-run-it-yourself)
9. [Repo map](#9-repo-map)
10. [Known limits](#10-known-limits)
11. [Questions and answers](#11-questions-and-answers)

---

## 1. Judges: a 2-minute tour

1. Open **https://feelable.onrender.com**. If it is asleep, wait about 50 s.
2. The gallery has **ten prepared lessons**. Use the **Basic / Complex** filter:

   | Basic | Complex |
   |---|---|
   | Parts of a flower | Skull: bones and sutures (27 targets) |
   | Layers of the Earth | Animal cell |
   | Favourite fruit (bar chart, with values) | Mixed circuit: series and parallel |
   | Water cycle (with arrows for its four processes) | Plant watering robot (a flowchart with Yes/No branches) |
   | Four flat shapes | Riverside bus routes (a map with stations and routes) |

   An **extended library** holds eight earlier examples. Some of their outlines have not been re-checked, and the page says so.
3. Open **Skull: bones and sutures**. Drag the **Original ↔ Tactile** slider. Tap a bone and its row lights up. Open the **Scan details** tab to see each step the image pipeline took on this picture: the labels it read, the pointer lines it followed, and the shapes it traced.
4. Open **Touch mode** on a phone (scan the QR code on the Share step). Touch a part to hear it. Double-tap for its description.
5. Click **Download PDF** (page 2 is the braille key) or look at the **3D** tab.
6. Try your own photo. Use **Take photo** or **Choose photo**: a textbook page, a worksheet, or a hand-drawn diagram.

---

## 2. The problem and who it is for

Blind and partially sighted students learn from the same textbooks as everyone else. Every diagram has to be **remade by hand** as a tactile graphic.

- That takes **10–15 minutes for a simple diagram and hours for a complex one** ([TactileNet](https://arxiv.org/abs/2504.04722)).
- It adds up to **around 300 hours per textbook** ([ASEE](https://nemo.asee.org/public/conferences/327/papers/38322/download)).
- Students often **wait weeks** for their materials ([Carleton University](https://carleton.ca/news/story/communicating-visual-information-vision-impaired)).

AI image describers can say *what* a diagram shows. A blind learner also needs to know *how it is laid out*: which part is where, how big it is, and what touches what. That needs a graphic they can feel.

**Who uses Feelable:**

- **The teacher, or a qualified teacher of the visually impaired.** They get a print-ready first draft in seconds instead of an hour, and they stay in charge of every part.
- **The blind student.** They can explore the diagram on swell paper, on a 3D print, or with no printer at all on a phone that speaks.

**Why it can be used after the weekend:**

- The core image pipeline runs on our own server **with no paid API**, so a school can use it on day one.
- The output follows tactile-graphics rules and is **measured** against them: 1.5 mm outlines, enlarged tiny parts, distinct textures for neighbouring parts, and UEB braille keys.
- Students only ever see a lesson **after the teacher approves it**.

---

## 3. How it works: image AI from start to finish

<p align="center"><img src="docs/img/how-it-works.png" alt="Three steps: snap a diagram, get a tactile graphic (PDF and 3D relief), touch it and hear it on a phone" width="100%"></p>

Image AI is not an add-on here. **Every output is built from what the vision pipeline finds in the photo.** Take the vision out and there is nothing to print, feel or hear.

| 1 · Read every label | 2 · Follow each pointer | 3 · Trace the shapes | 4 · Make it feelable |
|---|---|---|---|
| <img src="web/public/how/1-read.webp" alt="Every label highlighted"> | <img src="web/public/how/2-follow.webp" alt="Pointer lines traced to the bones they name"> | <img src="web/public/how/3-trace.webp" alt="Bones outlined, carved ones shaded"> | <img src="web/public/how/4-feel.webp" alt="The raised-line page with textures and braille keys"> |

Figure: OpenStax *Anatomy & Physiology*, CC BY 4.0.

### The pipeline

```
Photo
 │
 ├─ 1. Prepare      EXIF rotation, HEIC, page flattening, resize, quarter-turn check
 ├─ 2. Read         neural OCR (PP-OCR detection + recognition, ONNX, on our server)
 │                  unsure words re-read from a 2× zoom, snapped to 880 science terms
 ├─ 3. Classify     diagram · chart · flowchart · map · circuit · object photo · scene · text page
 ├─ 4. Segment      k-means in Lab colour space, or enclosed regions for line art
 ├─ 5. Associate    follow each pointer line from its label to its tip (across crossings)
 │                  zoom-and-carve: seeded watershed on a magnified crop splits
 │                  same-coloured neighbours along their real edges
 ├─ 6. Type         every target is a region, boundary, path, landmark or group
 │                  e.g. a suture = the border two bones share; an arrow = a path with a direction
 ├─ 7. Ground (AI)  Gemini vision: an inside point and a box for structures whose pointer is lost,
 │                  then GrabCut on a magnified crop; names unlabelled parts; writes descriptions
 ├─ 8. Render       tactile rules applied and measured; UEB braille; textures; PDF + watertight STL
 └─ 9. Speak        Gemini text-to-speech in touch mode, with the device voice as an instant fallback
```

### Model choices and why

| Job | What we use | Why this one |
|---|---|---|
| Read labels | **PP-OCR** through RapidOCR (ONNX, CPU) | Runs on a free server with no key. Finds small text in figures well. |
| Fix misread labels | 2× re-read plus a fuzzy match (RapidFuzz) against 880 science terms | Phone screenshots turn "Parietal bone" into "Parktatbone". Common English words are never changed. |
| Split parts | k-means (Lab), marker-controlled watershed, GrabCut (OpenCV) | Fast, explainable, and a cut is accepted only when it sits on a real edge in the picture. |
| Find lost structures, name unlabelled parts, describe parts | **Gemini 3.5 Flash** (vision), with Flash-Lite as a fallback | Visual grounding returns points and boxes that our own segmentation turns into exact outlines. |
| Voice | Gemini text-to-speech | A natural voice. The device voice takes over at once if it fails. |
| 3D relief | trimesh | Every STL is checked to be a clean, watertight solid. |

Steps 1–6 and 8 run **without any API key**. Gemini adds the grounding, the names and the spoken descriptions. Every AI sentence carries a visible **AI** badge, and the teacher can edit it.

### Understanding the structure, not just the pixels

Diagrams are more than coloured blobs, so Feelable gives each label the right kind of target:

- **Regions:** a bone or an organ.
- **Boundaries:** a suture is traced as the border two bones share, not as a dot at the end of a pointer.
- **Paths:** "Evaporation" becomes an arrow from the sea to the cloud.
- **Graphs:** a flowchart becomes boxes and arrows, spoken as "Soil dry? A decision. Yes: open valve. No: log reading." A bus map becomes stations, routes and the places where routes meet.
- **Landmarks:** a point, such as the pterion on a skull.

If Feelable finds where a structure is but not how far it goes, it says **"location found, extent unresolved"**. It offers a **Trace the border** or **Draw** tool rather than pretending it is done.

---

## 4. Measured results

All numbers come from scripts in `qa/`. We report **where outlines land, not just how many labels were read**.

### Hand-annotated targets (`qa/precision`)

We annotated targets by hand from the source images (points read off a pixel grid, never from Feelable's output). `qa/target_score.py` checks every point:

- a **region** must cover all its points and none belonging to another target;
- a **path** or **boundary** must cover its whole course;
- a **critical error** is a name placed on the wrong structure.

| Input | Before | After |
|---|---|---|
| Fresh skull photo | 14/25 valid, 3 critical | **18/25, 2 critical** (sutures traced along their course) |
| Water cycle | 4/8, 2 critical | **8/8, 0 critical** |
| Flowchart | 4/12; arrows 0/11 | **12/12; arrows 11/11** |
| Bus map | 1/15; routes 0/15 | **15/15; routes 15/15** |
| Rotated circuit | 0/4 | **3/4** |
| Notebook organs (handwritten) | 5/10, 1 critical | **6/10, 0 critical** |
| **All 11 inputs** | **67/142, 16 critical** | **104/142, 10 critical** |

**Honest caveats:**

- We wrote the annotations ourselves.
- These images were used while we built the fixes.
- This is **not** a held-out accuracy claim.
- The blurred skull gained one critical error (2 → 3).

The full table and the before/after overlays are in [`docs/FIX-STATUS-2026-10-12-precision.md`](docs/FIX-STATUS-2026-10-12-precision.md) and `docs/evidence/precision-2026-10-12/`.

<p align="center">
  <img src="docs/evidence/precision-2026-10-12/before-skull-fresh-overlay.jpg" alt="Before: skull outlines from the earlier version" width="49%">
  <img src="docs/evidence/precision-2026-10-12/after-skull-fresh-overlay.jpg" alt="After: skull outlines with sutures traced as lines" width="49%">
</p>
<p align="center"><em>Skull, before and after. Sutures are now traced along the border two bones share.</em></p>

### Dense anatomy benchmark (`qa/dense`)

This set has 24 dense figures: skulls, eye, ear, kidney, digestive system, heart, organ paintings, and 420 px low-resolution copies. It was run offline with AI off.

| | Before | Now |
|---|---|---|
| Labels read correctly | 94 % | **99 %** |
| Labels that became a named, feelable shape | 52 % | **71 %** |

### Robustness (`qa/robustness`)

This set has 239 images from the internet: objects, animals, scenes, maps, charts, handwriting, odd file formats and phone-like distortions.

| | Baseline | Now |
|---|---|---|
| Crashes | 0 | 0 |
| Images with junk output | 192 | 114 |
| Runs over 512 MB | 63 | 4 |
| Worst-case memory | 5.3 GB | 962 MB |

### Tests

- **304 backend tests**, including the annotated end-to-end targets: `make test`.
- **Browser checks** with Playwright at 1440 px, 390 px and 320 px: `qa/e2e_precision.py` (27 checks) and `qa/e2e_fixes.py` (16 checks).
- **Gallery gate:** `qa/check_gallery.py` requires each of the ten featured lessons to have no missing, unnamed or wrongly placed parts.

---

## 5. How Feelable meets each judging criterion

Each criterion is scored from 1 to 5, and all five count equally.

| Criterion | What to look at | Where it is |
|---|---|---|
| **Technical execution and difficulty** | A multi-stage vision pipeline, not an API wrapper: neural OCR with zoomed re-reads, pointer tracing across crossings, zoom-and-carve watershed, boundary tracing between regions, flowchart and map graph extraction, Gemini visual grounding refined by GrabCut, tactile rendering measured against rules, UEB braille, and watertight STL. Graded against hand-annotated targets. | `backend/engine/`, `qa/target_score.py`, section 4 |
| **Innovation and creativity** | Image AI aimed at **fingers instead of eyes**. The same photo becomes a raised-line print, a 3D relief and a phone you can touch and hear. Structure-aware targets (sutures as borders, arrows as paths, flowcharts as graphs) are what a blind reader needs and what a plain image describer leaves out. | Sections 3 and 11 |
| **Use of the image AI theme** | The image is both the input and the output. Vision reads, segments and grounds the photo, and the system *generates* a new image (the tactile print), a 3D model and speech from it. Remove the image AI and nothing is left. | Section 3 |
| **Impact and usefulness** | A real, measured bottleneck (hours per diagram, weeks of waiting). A clear user (teachers of blind students), a clear benefit (a draft in seconds), and a path to use after the weekend: free, no key needed for the core, teacher approval built in. | Section 2 |
| **Design, presentation and demo** | A clean teacher flow: Upload → Review → Fix → Share. 48 px touch targets on phones, no sideways scroll at 320 px, a talking touch mode, a Basic/Complex gallery, and a **Scan details** tab that shows the AI's work on your own picture. Demo video of 3 minutes or less. | Live site, [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md), [`DEVPOST.md`](DEVPOST.md) |

---

## 6. Award categories we are eligible for

Every submission is considered for every category, so nothing needs applying for. Here is where we think Feelable fits, as honestly as we can put it.

| Award | Fit | Why |
|---|---|---|
| **Top project** | Strong | Scores across all five criteria. |
| **Best Creative Tool** | **Strong** | Feelable puts image AI in the hands of a teacher **making** something: a tactile graphic. The teacher stays the author. They review, rename, redraw a missing part on the photo, trace a border, write what the student hears, and approve. |
| **Most Surprising Output** | **Strong** | Nobody expects an image AI project to output a picture you **cannot see but can feel**: a sheet of raised lines with braille, a 3D-printed relief, and a phone that speaks a flowchart's Yes/No branches when you touch them. |
| **People's Choice** | Good | Quick to understand and easy to try live. Close your eyes, touch the phone, hear "left ventricle". |
| **Best Fine-tune** | Weak | We did **not** train a model, LoRA or adapter. Our tuning is a hand-built pipeline: the vocabulary repair, the segmentation rules and a scored benchmark. We would rather say that plainly. |

---

## 7. Hackathon submission checklist

| Requirement | Status |
|---|---|
| Built with generative or image AI as the core | ✅ Vision pipeline plus Gemini vision and text-to-speech. See section 3. |
| Team of 1–4 | ✅ |
| Working, hosted demo | ✅ https://feelable.onrender.com |
| Public code repository | ✅ https://github.com/TYK3/feelable |
| Devpost write-up | ✅ [`DEVPOST.md`](DEVPOST.md) |
| Demo video, 3 minutes or less | ✅ Script and shot list in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md); the video is linked on Devpost |
| Third-party content credited | ✅ Skull, heart and digestive figures: OpenStax *Anatomy & Physiology*, CC BY 4.0. Mixed circuit: OpenStax *University Physics Volume 2*, CC BY 4.0. Shapes, flowchart, bus map and the other generated samples: Feelable original artwork, CC0. |
| Secrets kept out of the repo | ✅ `GEMINI_API_KEY` is set only in the hosting dashboard |

---

## 8. Run it yourself

You need Python 3.11+ (built on 3.13) and Node 18+ (built on 22).

```bash
# backend
pip install -r backend/requirements.txt        # add --break-system-packages on some Linux setups
# web app
cd web && npm install && npm run build && cd ..
# start the web app and the API on one port
./run.sh                                       # → http://localhost:8000
```

On Windows, double-click `start-windows.bat`.

To turn on the optional Gemini features (finding lost structures, naming unlabelled parts, descriptions, AI voice):

```bash
export GEMINI_API_KEY=...             # never commit this
export GEMINI_MODEL=gemini-3.5-flash  # optional
```

**Development:**

- `make dev` runs the API on :8000, and `make web` runs Vite on :5173 (it proxies `/api`).
- `make test` runs the backend tests.
- `cd web && npx tsc -b --noEmit` type-checks the web app.

**Quality checks:**

```bash
python qa/bulk_eval.py /tmp/eval qa/precision/inputs/* --truth qa/precision/truth   # hand-annotated targets
python qa/check_gallery.py                                              # the ten featured lessons
python qa/e2e_precision.py http://localhost:8000 /tmp/shots              # browser checks
```

**Deploy:** `render.yaml` is a Render Blueprint (one free Docker web service). Pushes to `main` deploy automatically. If one doesn't, use **Manual Deploy → Deploy latest commit** in the Render dashboard. `/api/health` reports the commit it is running. See [`DEPLOY.md`](DEPLOY.md).

---

## 9. Repo map

| Path | What it does |
|---|---|
| `backend/engine/analyze.py` | Reads the image: classification, OCR, segmentation, pointer tracing, flowcharts, maps |
| `backend/engine/grounding.py` | Label-to-geometry association: zoom-and-carve, boundary tracing, paired parts, cleanup |
| `backend/engine/ocr.py`, `lexicon.py` | Neural OCR, zoomed re-reads, the 880-term science vocabulary |
| `backend/engine/ai.py` | Gemini vision grounding, names, descriptions, text-to-speech |
| `backend/engine/tactile.py`, `textures.py`, `braille.py`, `export.py` | Tactile rules, textures, UEB braille, PDF and STL |
| `backend/engine/pipeline.py` | Jobs, the readiness check and teacher edits |
| `backend/app/main.py` | FastAPI server |
| `web/src/pages`, `web/src/components` | Teacher app: upload, review, fix, share |
| `web/src/touch/` | Talking touch mode and the "Find the…" game |
| `samples/` | Gallery images, `manifest.json` (reviewed corrections), the art generators |
| `qa/` | Benchmarks, the target scorer, the gallery gate, browser checks |
| `docs/` | Fix reports, evidence and images |

---

## 10. Known limits

- **Not a 90 % accuracy claim.** There is no independent, held-out annotated set yet.
- **Pale or blurred figures are still hard.** The low-contrast notebook scores 3/10. The blurred skull has one more critical error than before.
- **Some skull parts are still wrong.** The zygomatic arch is not outlined correctly, and the mastoid parts overlap their neighbours.
- **The mixed circuit misreads R3 as "Rg".** The series/parallel wiring is not extracted. Only the reviewed description says it.
- **Not yet tested with blind students, specialist teachers, screen readers on real phones, or real swell-paper and STL prints.** This is the most important next step.
- **The free server sleeps.** The first request after a quiet spell is slow.

---

## 11. Questions and answers

**Q: Isn't this just an image description API with extra steps?**
No. A describer returns a sentence. Feelable has to know **where every part is and exactly where its edge runs**, because a finger follows that edge. Most of the work is our own computer vision: OCR with zoomed re-reads, pointer tracing, watershed carving, and boundary and graph extraction. Gemini helps where a pointer is lost, and its point and box are refined into an exact outline by our segmentation.

**Q: Where is the *generative* part?**
Feelable generates new media from a photo. It makes a tactile image designed by rule for fingers (textures, enlarged parts, braille), a 3D relief mesh, spoken descriptions written by Gemini, and a Gemini voice. The output is a new picture, just one meant to be felt.

**Q: Why not ask a large vision model to segment everything?**
We do use Gemini for grounding, as points and boxes. A raised line has to sit exactly on the real edge, though, and a hosted model costs money and needs a network on every run. So our own segmentation draws the final outline. Our pipeline runs free on a CPU, is explainable (the **Scan details** tab shows every step), and accepts a cut only when it sits on a real edge in the photo.

**Q: How do you know the outlines are right?**
We score against hand-annotated targets, not against label counts. A region has to cover its own points and nobody else's, and a suture has to cover its whole course. Results went from 67/142 to 104/142 valid targets, and critical errors went from 16 to 10. We also list what is still wrong (section 10).

**Q: What happens when Feelable gets something wrong?**
The readiness panel lists problems first, with a fix button for each: **Draw** a missing part on the photo, **Trace the border**, **Exclude nearby area**, rename or merge parts. A lesson with a blocking issue, such as a title braille cannot spell, cannot be published. Students only see a lesson after the teacher approves it.

**Q: Does a school need a 3D printer or swell paper?**
No. Touch mode works on any phone: touch to hear, double-tap for more, a buzz at edges. Swell paper and 3D prints are there for schools that have them.

**Q: Does it work without an internet AI service?**
Yes. Reading, segmenting, tracing and the PDF and STL all run on our server with no API key. Gemini adds finding lost structures, naming unlabelled parts, descriptions and the natural voice. If it fails, the lesson still works and the page says why.

**Q: What kinds of pictures can it handle?**
- labelled science diagrams (cells, organs, skulls, the Earth's layers)
- charts with values
- process diagrams with arrows
- flowcharts
- maps with routes
- circuits
- geometry
- photos of real objects
- handwritten notebook drawings

Pale, blurred and very crowded figures are the weakest.

**Q: Is braille handled properly?**
Labels and keys use UEB grade 1 braille, placed beside each part with a clear pad, plus a key page. If a title has characters braille can't spell here (for example "H₂O" or Chinese characters), publishing is blocked until the teacher edits it, instead of quietly dropping them.

**Q: Why should we believe someone will use this after the weekend?**
- The problem is measured in hours per diagram.
- The core is free to run.
- The output follows the tactile-graphics rules that specialist teachers already use.
- The teacher stays in control.

The next step is testing with teachers of the visually impaired and blind students, then a shared, approved library so each textbook diagram only has to be converted once.

**Q: What was hardest?**
Dense textbook figures: two dozen small labels over bones of nearly the same colour, with pointer lines that cross. Making an image was never the hard part. Making one that **fingers can read correctly** was.

**Q: Did you fine-tune a model?**
No, and we don't claim Best Fine-tune. The "tuning" is our pipeline, the science vocabulary and a scored benchmark.

**Q: Is the content licensed for use?**
Yes. The OpenStax figures are CC BY 4.0, credited in `samples/manifest.json` and in this README. All other samples are our own artwork, released CC0.
