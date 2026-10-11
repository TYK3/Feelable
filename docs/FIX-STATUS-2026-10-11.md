# Feelable fix status — 11 October 2026 (second pass)

Source: "Feelable repair handover" and "live stress-test handoff" (11 Oct 2026), reference commit
`e491670`. This document lists what changed, how it was verified, and what is still open. It does
not certify physical tactile suitability, full UEB compliance, or semantic accuracy on every upload.

## How it was tested

* **Bulk test, 83 images** (`qa/bulk_eval.py`), each run fresh through the real engine (no sample
  cache, automatic descriptions off), before and after the changes:
  * 41 densely labelled OpenStax figures (CC BY 4.0, fetched by `qa/fetch_corpus.sh`): skulls,
    heart (internal, surface, valves, conduction), digestive tract, lungs, larynx, kidney, eye,
    ear, neuron, cells, bones, endocrine, lymphatic, tongue and six resistor circuits;
  * the 30 audit fixtures (blur, glare, rotation, EXIF, charts, other scripts, damaged files);
  * the 12 earlier gallery images.
  For every traced pointer line it checks whether the line ends inside the part that carries its
  label, and it flags parts with a sizeable piece far from the part's main body.
* **Gallery release gate** (`qa/check_gallery.py`): builds all 15 prepared examples exactly as the
  server cache does and fails on any unresolved missing target, unnamed part or Braille block.
  Every selection was also checked by eye on per-part contact sheets (`qa/part_sheet.py`).
* **Browser test** (`qa/e2e_fixes.py`, 17 checks) against a local server: gallery, phone widths,
  Remove part, Add missing part, approval wording, close-up, student touch.
* **Unit tests**: 286 backend tests pass (1 skipped); frontend typecheck and build pass.

### Bulk results (83 images, before → after)

| Measure | Before | After |
|---|---:|---:|
| Named parts | 571 | 639 |
| Pointer lines whose label got no part | 245 | 213 |
| Parts with a sizeable far-away piece (the "nasal bone selects the jaw" failure) | 103 | 60 |
| Pointer ends inside its own part | 493 | 525 |
| Pointer ends outside its own part | 67 | 71 |
| "Possible missing part" suggestions | 237 | 201 |
| Unnamed parts | 164 | 173 |
| Lessons blocked by cut Braille | 11 | 0 |
| Median seconds per image (2 workers) | 9.1 | 8.6 |
| Errors (all 3 are files that are not images) / timeouts | 3 / 0 | 3 / 0 |

These are consistency measures (does a name sit on the shape its pointer reaches), not accuracy
against a human annotation of every figure.

### Audit fixtures replayed (before → after, fresh upload)

| Fixture | Before | After |
|---|---|---|
| Skull, side view | 17 parts; nasal bone covers the jaw; 10 labels missing; Braille blocked | 27 parts for the 27 labels; nasal bone is the nose, mandible the jaw; all 23 traced pointers land on their own part; sutures and landmarks are marked spots |
| Sideways animal cell | "Main object / Middle detail / Right detail" | turned upright, 11 named parts |
| Blurred heart | 3 generic parts, nothing said | same outline, plus "labels couldn't be read reliably … retake" |
| Chinese sign | teachable "Main object" | blocked: "Chinese or Japanese text … Feelable reads English labels" |
| Two-series line chart | Part 1 / Part 2 | named "y=1" / "y=-1" from the legend colours |
| OpenStax mixed circuit | 3 unnamed parts, loop filled in, R2/R3 missing | battery 12.0 V, R1 1.00 Ω, R2 6.00 Ω, R3 flagged to check (OCR read "Rg"); wires raised |
| Audit circuit photo | Part 1, Part 2, "rs", Part 4 | R2 6.00 Ω, resistor 13.0 Ω (flagged), battery 12.0 V, one unnamed |
| Truncated JPEG / animated GIF | accepted silently | review notes "file was incomplete" / "only its first frame was used" |
| Braille probes | `.5` and `"cell"` wrong; 63-cell key rows cut to 44; 80-cell title cut | all nine audit expectations match; long titles wrap; long key lines wrap and add key pages; nothing is cut |

## Repaired

| ID | Change | Evidence |
|---|---|---|
| F01 | Grounding repair after carving: a pointer on a separate piece of another label's region takes that piece; carving no longer refuses a correct split because the new piece is larger; pointer tips that stop on an outline are stepped forward into the region; unexplained far pieces are taken out of a part (repeated copies kept only when the shape matches); a name whose pointer lands nowhere on its region is released; a "part" lying along a pointer line is dropped; long "X of Y" names are kept; landmark labels (suture, fossa, tubercle, pterion …) get a marked spot at the pointer tip when no region can be traced. | skull fixture above; bulk table; `tests/test_label_roles.py`, existing analyze tests |
| F02 | Unsupported characters block publication with the part named; wording "Feelable cannot currently translate". | first pass |
| F03 | Titles wrap onto up to 3 Braille rows; key names wrap onto continuation lines under their name and extra key pages are added (PNG stacked, PDF one page each); a cut title still blocks. | `tests/test_braille_overflow.py` counts raised dots on the rendered pages against the dots the full text needs |
| F04 | "Ready" only when approved with no open checks; approved lessons with open checks say "Approved — N checks still open". | browser test; readiness tests |
| F05 | Per-stage deadline; stuck worker killed; retryable timeout message. | first pass, `tests/test_engine_deadline.py` |
| F06 | Text-less pictures are tried at 90°/270° before falling back to a photo outline; recognisable but unreadable labels say so. | rotated cell and blurred heart above |
| F07 | View captions and equations are not missing parts; group headings ("Small intestine:", "Salivary glands:") are shown as groups of the parts under them; circuit designators become named components with values. | `tests/test_label_roles.py`; digestive example |
| F08 | Close-up starts on the pointer's end, marked dashed, with a hint; when only a stub was traced it says "Choose the structure this label refers to". | `tests/test_assisted_outline.py::test_closeup_target_needs_a_real_pointer`; browser test |
| F09 | `.5` → numeric indicator first; prose quotes → ⠦…⠴; curly quotes supported. | `tests/test_braille_overflow.py::test_translator_matches_independent_ueb_expectations` |
| F10 | Damaged/animated input disclosed; no exception names in user text. | fixtures above |
| F11 | Error and processing copy (first pass). | first pass |
| F12 (partial) | Unnamed chart series are named from legend entries by marker colour. | line-chart fixture |
| F13 | Non-Latin scripts block with a plain explanation (unit signs like Ω, µ excluded). | Chinese fixture |
| Clutter | Ruled-paper dashes and a box round the title are removed from the raised layer. | notebook example |

## New features

* **Remove part / Restore part** on the selected part and in the parts list, with Undo; removed
  parts leave the student view, key, audio and downloads.
* **Add missing part** (assisted): tap inside, optionally mark what is not part of it, **Find
  outline** proposes a locally traced outline (never from the name alone); the teacher can use it,
  edit the points, or **Draw myself**; Undo/Redo/Start over; choose a new part, **Add to** or
  **Remove from** an existing part. If the outline covers other parts the teacher chooses **Cut it
  out of them**, **Keep both** or **Replace** — nothing is erased silently. Late answers for an older
  draft are ignored. Endpoint: `POST /api/jobs/{id}/outline` (read-only).
* **Fix outline** on each part opens the same tool on that part.
* **Focus** zooms to the selected part and keeps it centred.

## Gallery: exactly 15 examples

Ten simple examples are kept; five complex systems: **Skull: bones and sutures**, **Digestive
system: organs and pathway**, **Heart: chambers, valves and vessels** (OpenStax), **Human organs:
handwritten notebook** (team-supplied drawing), **Mixed circuit: series and parallel** (OpenStax).
The home page shows 4 simple examples, the 5 complex systems and "Show all 15"; each card has a
learning objective and says it is a prepared example.

All 15 pass the release gate (zero unresolved missing targets). Where automatic analysis was wrong,
the example carries **reviewed corrections** in `samples/manifest.json` (`review`), made with the
same operations a teacher has (rename, describe, remove, merge, tap-to-outline, draw, cut a piece
out of a part) and replayed when the cache is built. Corrections per example: skull 0, digestive 15,
heart 22, notebook 22, mixed circuit 5, plant cells / animal cell / simple circuit 2 each (a marked
spot for ribosomes or wire). The cache is rebuilt on every deploy; existing lessons are not changed.

## Still open

* **Fresh-upload accuracy on dense figures.** The gallery versions of the heart, digestive system
  and notebook are correct because of reviewed corrections; a fresh upload of those figures still
  leaves unnamed parts and some missing labels (e.g. textbook heart: 19 named of 23 parts). The
  repair tools are the path for teachers; a stronger local segmentation model would be the next step.
* Pointer lines that stop at a bracket (grouped labels) are not followed beyond it.
* Hand-drawn outlines in the curated heart/notebook/digestive examples are approximate; the
  jejunum/ileum boundary is not drawn in the source and the descriptions say so.
* F05: heartbeat recovery after a process restart and job identity across polling failures.
* F12: pH-scale values and an editable data table are not done.
* F13: scripts OCR cannot read at all (Hebrew page) still get the generic "no clear object" message.
* Overview + linked detail pages for crowded lessons (the 25-part heart is flagged as crowded).
* Holes and multi-piece outlines in the draw tool; a magnifier for finger drawing.
* Real-device checks (VoiceOver, TalkBack, haptics), Braille-reader and specialist review of a
  printed or 3D-printed result, and printer-bed tiling for the STL.
