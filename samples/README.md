# Feelable sample diagrams

These are the textbook-style test images used in the app's one-click sample gallery, in QA and Vision tests, and in the demo video.
`manifest.json`: Extra fields are `description` on every entry and `arrow_labels` on the water cycle.

| id | file | kind | expected parts | min | What it stresses |
|---|---|---|---|---|---|
| `plant-cell` | plant-cell.png | colour | cell wall, cytoplasm, vacuole, nucleus, chloroplast, mitochondrion | 5 | Nesting (wall ⊃ cytoplasm ⊃ organelles), repeated parts (5 chloroplasts, 3 mitochondria), 170 ribosome dots, highlights and a drop shadow to remove as clutter |
| `heart` | heart.png | colour | right/left atrium, right/left ventricle, aorta, pulmonary artery, vena cava, septum | 7 | Adjacent chambers that share walls, similar hues (several blues and several reds), two-word labels, tube-shaped vessels that overlap, an elbowed leader line |
| `water-cycle` | water-cycle.png | colour | sun, cloud, mountain, sea | 3 | A full-frame sky background, a part made of two pieces (two clouds), thin rain strokes and sun rays as clutter, four curved black arrows (linework) with process labels that have no leader lines |
| `plant-cell-photo` | plant-cell-photo.jpg | photo | the plant-cell parts | 5 | Page flattening (the page is photographed at an angle on wood), body text that must be ignored, blur, noise, uneven light and vignetting, JPEG artefacts at quality 85 |
| `animal-cell` | animal-cell.png | colour | cell membrane, cytoplasm, nucleus, nucleolus, mitochondrion, golgi body | 5 | An irregular outline, a part inside a part (nucleolus in nucleus), a thin part made of several pieces (Golgi cisternae), chromatin flecks and ribosomes as clutter |
| `earth-layers` | earth-layers.png | colour | crust, mantle, outer core, inner core | 3 | Concentric sectors, a very thin ring (crust), large parts with no label (ocean and continents), long leader lines that cross other parts |
| `bar-chart` | bar-chart.png | chart | apple, banana, orange, grapes, kiwi | 4 | Chart axes, ticks, gridlines and number labels; labels sit under the bars, not on them; a rotated y-axis title |
| `circuit` | circuit.png | lineart | battery, switch, bulb, resistor | 2 | Black-on-white line art only, so parts are enclosed regions (bulb quadrants, resistor box); the loop is open at the cell and the switch |

The order of the list is the order for the gallery, with the most impressive samples first.

## Regenerate

```bash
python3 samples/make_samples.py            # all images + manifest.json
python3 samples/make_samples.py heart      # only the listed ids (the manifest is always rewritten)
```

The output is deterministic: it uses fixed seeds and no clock or network, so running the script twice gives byte-identical files.
It needs Pillow, numpy, opencv and shapely, plus the Inter fonts at `/usr/share/fonts/opentype/inter/`.
Images are drawn at 2× and downsampled, which gives anti-aliased edges like a real screenshot or scan.

## Style

- Flat fills with a thin, darker outline in the same hue, and a distinct hue for each part.
- Labels in Inter Medium 22 px with 1.5 px leader lines that end in a dot on the part. Multi-word labels have a slightly wider word gap so OCR keeps the words apart.
- Title is "Fig. X.Y" in Inter SemiBold, top left. One grey caption line sits at the bottom.

## OCR check

`rapidocr_onnxruntime` reads 100 % of `expected_parts` (plus the arrow labels) on every sample, including the photo.

## Gallery of 15 (October 2026)

`manifest.json` now also has `group` (`simple` / `complex`), `objective` (what a student learns) and
`review`: reviewed corrections replayed by `backend/precompute_samples.py` when the gallery cache is
built (rename, describe, remove, merge, tap-to-outline, draw, cut a piece out of a part — the same
operations a teacher has in Feelable). **`make_samples.py` rewrites `manifest.json`: re-apply the
`group`, `objective` and `review` fields afterwards**, and run the release gate:

```bash
cd backend && python ../qa/check_gallery.py /tmp/gallery-check   # exactly 15, five complex, zero unresolved targets
```

| id | file | source |
|---|---|---|
| `heart-internal` | heart-internal.jpg | OpenStax *Anatomy and Physiology* (GitHub `openstax/osbooks-anatomy-physiology`, `media/2008_Internal_Anatomy_of_the_HeartN.jpg`), CC BY 4.0 |
| `human-organs-notebook` | human-organs-notebook.png | hand-drawn classroom notebook page supplied by the Feelable team |
| `mixed-circuit` | mixed-circuit.jpg | OpenStax *University Physics* (GitHub `openstax/osbooks-university-physics-bundle`, `media/CNX_UPhysics_27_01_ResistPar2.jpg`), CC BY 4.0 |
