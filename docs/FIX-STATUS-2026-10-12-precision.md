# Feelable: precision audit fix status (12 October 2026)

This responds to *Feelable precision and demo fix handover* (audit of commit `e356963`). The work started from GitHub `main` at `e356963d0d8ed1836bb335a836a1dee9a5f1cec1`, not from a Downloads backup. History was not rewritten.

**Short version:** the changes fix how labels are matched to shapes. A suture is now traced as a line, a process arrow is now a path with a direction, flowchart boxes are named one-to-one, map stations and routes are separate targets, and the rotated circuit reads correctly again. On the hand-annotated audit set, 104 of 142 targets are now valid (67 before) and there are 10 critical errors (16 before). **None of this is a 90% accuracy claim.** The annotations were written by the same agent that made the fixes, and the images are the audit images that were tuned on, not held-out sources.

## How it was measured

- `qa/precision/inputs/`: the audit's fresh, transformed and diagnostic images.
- `qa/precision/truth/*.json`: target annotations written **by hand from the source image** (points read off a pixel grid), never from Feelable output. Each target has a type:
  - **region**: every point must be inside the named part, and no point of another target may be.
  - **path / boundary**: every point along the course must be covered. A tip spot fails.
  - **landmark**: a point.
  - **mask**: IoU of at least 0.85 against the audit's ground-truth masks.

  Graph truth (flowchart edges, route membership) comes from the audit's `graph-controls-groundtruth.json`.
- `qa/target_score.py` scores an analysis against these annotations, and `qa/bulk_eval.py --truth` runs a whole set. A **critical** error is a name that covers another target's point (wrong anatomy or wrong node) or a forbidden point.
- Disclosure: during the work, three relations were added to the skull annotation (the temporal bone contains the zygomatic arch area; the zygomatic bone contains the temporal process; the mandible contains the mental protuberance). The occipital points were also corrected after checking the source. The "before" column was re-scored with the **final** annotation file and scorer, so both columns use the same rules.

## Before and after (same annotations, same scorer, local CPU)

| Input | Before (`e356963`) | After | Notes |
| --- | --- | --- | --- |
| Fresh notebook organs | 5/10, 1 critical | **6/10, 0 critical** | Kidneys no longer on the colon. Both kidneys are now one plural part. |
| Notebook, low contrast | 2/10, 2 critical | 3/10, 2 critical | Still poor: robustness is open. |
| Fresh skull | 14/25, 3 critical | **18/25, 2 critical** | Coronal, squamous and lambdoid sutures are traced boundaries. Ethmoid is found. |
| Blurred skull | 16/25, 2 critical | 16/25, **3 critical** | **Regression** in critical errors under blur. |
| Rotated skull | 14/25, 3 critical | 16/25, 3 critical | |
| Water cycle | 4/8, 2 critical | **8/8, 0 critical** | Evaporation, condensation, precipitation and collection are arrow paths with a direction. |
| Flowchart control | 4/12, 3 critical; edges 0/11 | **12/12, 0 critical; edges 11/11** | Same-colour boxes are separate. Arrows are edges with Yes/No branches. |
| Transit control | 1/15; routes 0/15 | **15/15; routes 15/15** | Stations are points. Routes are paths. Interchanges are derived from the geometry. |
| Mixed circuit | 3/4 | 3/4 | R3's designator is still misread ("Rg"), so it stays "resistor, 13.00 ohms". |
| Rotated mixed circuit | **0/4** | **3/4** | A quarter turn is accepted when it reads far more notation. |
| Geometry control | 4/4 | 4/4 | |
| **Total** | **67/142, 16 critical** | **104/142, 10 critical** | |

The median local processing time went from 13.1 s to 11.6 s (2 workers, this sandbox's CPU, not Render). The internal heart now completes in 23 s locally. Its 120 s timeout on Render was not re-measured.

Evidence is in `docs/evidence/precision-2026-10-12/`: before and after overlays for the skull and organs, the water cycle and transit overlays, `before-after-targets.json`, the browser-check JSON and phone screenshots.

## What changed

### Typed targets and label-to-geometry association (P0)

- `Part.target_type` can be region, boundary, path, landmark or group. `Part.extent` is full or unresolved. Both appear in every result.
- **Sutures and fissures** are traced as the border shared by the two outlined bones that meet at the pointer tip (`grounding.trace_boundary`). If no two bones meet there, the suture stays *location found, extent unresolved*. It is reported with a **Trace the border** action (`PATCH {"trace_boundary": id}`) instead of being counted as done.
- **Point landmarks** (pterion) stay points. Spots no longer cut a hole in their parent bone.
- **Parent structures keep their nested sub-parts** ("squamous temporal" inside "temporal bone"), but only when the names share a word. The greater wing of the *sphenoid* is never nested into the parietal bone.
- **Ties no longer depend on label order.** When several pointers end in one region, the whole structure owns it, then the pointer that ends deepest inside.
- **Pointers are followed across a crossing organ** (the kidney arrow crosses the colon).
- **Lumped colour clusters are split.** Hairline bridges are broken before judging distance, and pieces with a clearly different colour are taken out. Hairline slivers are removed.
- **Paired organs.** "Kidneys" pointed at one kidney takes its mirror twin (same colour and size, mirror position). A dark rim takes the lighter body it wraps.
- **No more title-noun names.** A leftover piece next to two organs is no longer called "organs".
- **Dot sets.** "Ribosomes" now names every dot in the cytoplasm. The prepared lessons' single drawn "ribosome" spot was removed.
- **Audit code risks:**
  - `_largest` takes the nearest piece within 12 px, never the largest far one.
  - `_support` judges the outline's real edge evidence instead of returning 99.
  - An OCR engine failure is reported (`ocr_failed`, Try again) instead of passing as "no text".

### P1 engine defects

- **Process arrows:** labels such as Evaporation become path targets ("An arrow from the sea to the cloud"). A shape that only had the caption sitting on it is switched off.
- **Flowcharts:** text that fills a box names that box, ahead of a stray pointer. "Yes" and "No" are branch captions. Arrows become edges with direction and branch, and are spoken as "Soil dry? A decision. Yes: open valve. No: log reading."
- **Maps:** ringed stations become point targets named by the nearest text. Coloured routes, including dashed ones, are named by the legend entry printed in their colour. Route stops and interchanges are listed.
- **Rotation:** a quarter turn is accepted for notation-heavy circuits.
- **Title Braille blocker:** `braille_unsupported_title` blocks publication, with an Edit title action. H₂O no longer silently becomes "HO".
- **Health:** `/api/health` now reports `commit` (Render's `RENDER_GIT_COMMIT`, or `FEELABLE_COMMIT`).

### Editor and student view (P2)

- **Drawing tool:**
  - Zoom in, Zoom out, Fit, Focus on marks.
  - Move picture (pan) and Adjust points (drag a vertex).
  - Full screen at phone widths.
  - The status line always describes the current draft. Undo says "Suggested outline removed." and Redo says "restored". Pending requests are aborted and `finding` is reset.
  - A suggestion in several pieces keeps **every** piece (`add_part.more_polygons`), with a Keep largest piece option.
- **Counts:** the header reads "27 included parts · 4 omitted candidates". Excluded candidates are hidden from the student chips. A generic title shows as "Untitled diagram".
- **Warnings:**
  - Missing parts now read "X has no matching outline."
  - Crowding now reads "The full image has N targets. Create detail pages…".
  - Description failure now reads "Descriptions could not be added. Your diagram is saved…", with provider diagnostics in a collapsed **Diagnostics** box.
  - The layout panel is renamed **Print layout checks** and says it is not a source-match review.
- **Phones:** principal controls are 48 px. The view tab strip wraps, so the 3D tab is on screen. Long parts lists get a search box.
- **Student view:** outlines plus "Loading the picture…" (or "didn't load" with Retry) replace the blank white sheet. Teacher preview has a direct **Back to editor** button.
- **Copy:**

  | Old | New |
  | --- | --- |
  | Every pointer line is traced | Follows visible pointer lines and flags uncertain matches |
  | AI names & descriptions | Add spoken descriptions |
  | Describe with AI | Add descriptions |
  | How it read | Scan details |
  | Mark not the part | Exclude nearby area |

### Gallery: five basic and five complex

- **Featured basic:** flower, Earth layers, fruit bar chart, water cycle (now with its four processes), four flat shapes (new).
- **Featured complex:** skull bones and sutures (27 targets), animal cell, mixed circuit, plant-watering flowchart (new), riverside bus routes (new, with a dashed route so identity isn't colour alone).
- The three new lessons are original Feelable artwork (`samples/make_featured.py`, CC0).
- The earlier eight examples stay in an **extended library**, labelled "outlines not re-checked". Digestive system, internal heart and notebook organs still have open checks there.
- The skull's reviewed corrections are recorded in the manifest:
  - remove the gum-line sliver
  - re-outline the ethmoid
  - outline the lacrimal bone and the greater wing of the sphenoid
  - extend the occipital bone
  - trace the lambdoid suture
- Prepared caches are stamped with the source SHA-256 and `annotation_version`. A cache is served only for the matching version.
- `qa/check_gallery.py` gates the featured ten strictly and reports the extended library. Result: **GALLERY PASS**.

## Tests run

- Backend: **305 passed**, including the new `tests/test_precision_audit.py`. That file covers the code-risk regressions, boundary tracing, mirror twins, the title blocker and the trace edit, and runs the annotated end-to-end targets with graph checks.
- Browser, local build: `qa/e2e_precision.py` **27/27**. This covers the gallery, skull counts and chips, sutures as boundaries, Undo/Redo status, zoom, the title blocker, flowchart and map descriptions, 48 px targets at 390 and 320 px, no sideways scroll, full-screen drawing, and a painted student canvas. `qa/e2e_fixes.py` passed **16/16**.

## Still open and unverified

- **Not a 90% claim.** There is no held-out, independently annotated source set yet. The annotations are mine, and the images were used during development.
- **Fresh notebook organs: 6/10.**
  - The lungs are pale and not segmented.
  - The small intestine gets only a small piece.
  - The ascending colon is not joined to the large intestine.
  - The liver's left lobe is split off.
- **The low-contrast notebook is 3/10.**
- **The blurred skull has one more critical error than before.**
- **Skull: the zygomatic arch is still wrong.** Its pointer stops at a bracket and the bracket span is not used. The mastoid portion and mastoid process overlap neighbours.
- **Curated anatomy in the extended library** (digestive, internal heart, notebook) still uses coarse rectangles from the earlier pass and is labelled as not re-checked.
- **R3 in the mixed circuit is read as "Rg".** Its designator is not inferred, and the wire graph (series/parallel) is not extracted. Only the reviewed descriptions say it.
- **Not built:**
  - overview and detail lesson pages
  - a "Not part of this lesson" decision with a recorded reason
  - Locate on the pointer tip for every warning
  - insert and delete vertex in the drawing tool
  - a lighter OCR or segmentation model comparison (PP-OCR, MobileSAM) on Render hardware
  - preview-first, lazy export timing on Render
  - Boundary IoU and clDice metrics
- **Not tested here:** real phones with a keyboard and gestures, VoiceOver and TalkBack, vibration, swell-paper or STL prints, a Braille reader or specialist teacher, and the Render deployment itself until it is checked live.
- **Taste and Impeccable skills:** not available in this session. The UI changes follow the audit's own brief.
