# Feelable fix status — 11 October 2026

Source: "Feelable live stress-test handoff" (11 Oct 2026). Reference audit commit `e491670`.
Scope of this document: what changed in the repository, what is verified by automated tests,
and what is still open. Nothing here certifies physical tactile suitability, Braille accuracy
against a Braille reader, or the live deployment.

## Repaired and verified by automated tests

| ID | Finding | Root cause (source) | Change | Verified by |
|---|---|---|---|---|
| F02 | Unsupported characters vanish from Braille; warning hidden | `braille.py` skips unsupported chars; readiness ignored the warning | `build_readiness` adds a blocking `braille_unsupported` issue naming the characters and part; warning reworded to "Feelable cannot currently translate…" | `test_readiness_rules`, `test_label_roles` (readiness path) |
| F03 | Long Braille title and key rows cut silently | `tactile.py` sliced title to one row and key rows to column width | Title wraps onto up to 3 Braille rows (word-aware, long words split by cell); key lines that must be cut raise a blocking `braille_overflow` issue | `tests/test_braille_overflow.py` (4 tests) |
| F04 | Approval shows "Ready to teach" with open checks | `ready` ignored open warnings | `ready` only when approved and zero open warnings; approval still publishes the reviewed snapshot; blocked lessons never publish | `test_readiness_rules`, `test_published_bundle_is_immutable_until_reapproved` |
| F05 (partial) | Stalled engine jobs, no deadline | `_engine_call` waited on `fut.result()` without limit | Per-stage deadline `FEELABLE_ENGINE_DEADLINE_S` (default 300 s); on timeout the worker process is killed and the job fails with code `timeout` (HTTP 504) and a retryable message | `tests/test_engine_deadline.py` |
| F07 | Captions and equations suggested as missing parts | "view" not a caption word; equation fragments accepted | Caption/view/figure words and equation fragments are rejected as structure names | `tests/test_label_roles.py` (11 tests) |
| F10 (partial) | Truncated JPEG and animated GIF accepted silently | Recovery and first-frame use not reported | Review warnings added: "incomplete at its end" and "only its first frame was used" | `test_robust_intake` (existing) and full suite |
| F11 | Gateway error blames connection; homepage says on-device OCR | Copy | Service-unavailable copy; "timeout" message; onboarding now says Feelable's server reads the text; privacy footer accurate | build + copy review |

Full backend suite at the time of writing: 271 passed, 1 skipped. Frontend `npm run build` and `tsc -b` pass.

## Partially addressed

- **F05 stale jobs:** the deadline and worker kill are in place. Heartbeat-based recovery after a
  process restart, persistence of job identity across polling failures, and cancel-reaches-terminal-state
  checks are not done. Replaying the animal cell, muscles and EXIF circuit fixtures against the live
  server was not possible from this environment.
- **F10 format messaging:** decode and first-frame notices now reach review. The HEIC and
  "format list in two places" consistency check is not done.
- **F02 Braille code:** characters are still not translated; they now block publication until renamed.
  Support for other scripts needs a language/code choice (F13).

## Not done (open)

| ID | Item | Why open |
|---|---|---|
| F01 | Label-to-geometry grounding (skull nasal bone; 10 missing skull targets; lungs fragments; heart endocardium) | Needs a labelled annotation set and CV work; not a code-only change. |
| F03 (full) | Wrap for legend key rows (currently cut and blocked, not wrapped); key pagination | Legend wrapping changes the page layout and needs physical-dimension review. |
| F04 (full) | Targeted acknowledgment per warning; teacher-facing unresolved counts on the approval screen | UI work in `ReadinessPanel.tsx` / `ReviewPage.tsx`. |
| F06 | Degraded diagrams (blurred, rotated) recovery path "Did you mean a labelled diagram?" | Needs orientation preview UI and recovery logic. |
| F08 | Detail close-up seeded from the target box, not the label box | The readiness data does not yet carry leader endpoints or target boxes. |
| F09 | Braille punctuation context (decimal `.5`, quotation signs) | Needs a maintained UEB translator (e.g. Liblouis) and independent expected-cell fixtures. |
| F12 | Chart legends and scale semantics | Requires a chart-specific semantic representation. |
| F13 | Non-English text mistaken for objects | Needs supported-script messaging and a language choice. |
| Assisted drawing of missing parts | Outline proposal, Undo/Redo, save part, conflict safety | Large feature; not started. |
| Gallery to exactly 15 | Five complex systems, ten retained | Requires annotated, reviewed examples and cache rebuild; not started. |
| Curated-example release gate | Zero unresolved missing targets | Depends on F01 and the gallery work. |
| Mobile improvements | Focus selected part, Previous step, prominent preview, view picker | Not started. |
| Regression evidence | Before/after replay against the live site | The sandbox cannot reach feelable.onrender.com (egress policy). Replay must be run from a machine with access. |

## Limits

- No Braille reader, specialist teacher, printer, phone haptics, or screen-reader session was available.
- Automated tests check internal consistency, not semantic accuracy or UEB compliance.
- The three Braille probe JSON files and the skull worklist from the audit folder were not available
  in this environment; the Braille fixtures in the tests are small synthetic cases.
- The live deployment has not been verified after the most recent push.
