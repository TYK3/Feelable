# Precision benchmark (development set)

Inputs from the 11 Oct 2026 precision audit, plus hand annotations in `truth/`. Every target is
written from the SOURCE image (points read off a pixel grid), never from Feelable output.

    cd backend && python ../qa/bulk_eval.py OUT ../qa/precision/inputs/* --truth ../qa/precision/truth

`qa/target_score.py` explains the scoring rules (region / path / landmark / mask, critical errors,
flowchart edges, route membership). This is a DEVELOPMENT set: the same images were used while
fixing the engine and the annotations were written by the same agent, so its numbers are not a
held-out accuracy claim. Add independently annotated, unseen source families before quoting one.
`aliases.json` maps transformed inputs (blur, rotate, contrast) to their upright annotation.
