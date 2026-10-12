# Feelable — Every diagram, feelable.

**Try it:** https://feelable.onrender.com · **Code:** https://github.com/TYK3/feelable

## Inspiration
Blind students use the same textbooks as everyone else, but every diagram must be hand-made into a **tactile graphic**: 10–15 minutes for a simple one, hours for a complex one ([TactileNet](https://arxiv.org/abs/2504.04722)), around 300 hours per textbook ([ASEE](https://nemo.asee.org/public/conferences/327/papers/38322/download)). Students wait weeks ([Carleton](https://carleton.ca/news/story/communicating-visual-information-vision-impaired)). AI describers say *what* a diagram is, not *how it's built*, which is what a blind learner needs.

## What it does
A teacher snaps a photo of a textbook diagram. In about 3 seconds Feelable returns:
- a **swell-paper PDF** (print it, heat it, and the black lines rise), with braille labels and a braille key page;
- a **3D-printable relief** (watertight STL) of the same diagram;
- a **talking touch mode** for the student's phone: touch a part to hear its name, double-tap to hear more, and feel a buzz when crossing an edge. A "Find the nucleus" game plays hotter/colder beeps to guide a finger to a part.

The teacher stays in charge:
- rename parts, rewrite what the student hears, or switch parts off;
- **draw a part the scan missed** straight onto the original photo;
- approve the lesson before students can open it. Students only ever see the approved version.

Touch mode reads with a **natural AI voice** (Gemini text-to-speech), and falls back to the device's voice instantly.

A **How it read** tab shows the teacher exactly what the image pipeline saw on their picture: every label it read (and the blurry ones it repaired), every pointer line it followed, and which shapes were traced by colour, carved out up close, or located by the image model.

## How we built it
**Understanding the image: a multi-step image-AI pipeline.**
1. **Read.** Photographed pages are flattened and labels read with on-device neural OCR (PP-OCR detection + recognition, ONNX). Unsure readings are re-read from a padded crop and a sharpened 2× zoom, and each candidate is snapped to a vocabulary of 880 science terms: "Parktatbone" becomes "Parietal bone", "Witroouschamber" becomes "Vitreous chamber". Common English words are never touched.
2. **Follow.** Thin leader lines are traced from each label to the spot it names.
3. **Segment.** Colour diagrams are segmented with k-means in Lab colour space; line drawings by enclosed regions. Clutter is dropped (ribosome dots are too small to feel), nested parts detected, chart values read.
4. **Zoom-and-carve.** Dense figures defeat a global segmentation: neighbouring bones or organs share a colour and come out as one region. So every pointer tip that lands in a shared region becomes a seed for a marker-controlled watershed on a magnified crop, which splits the region along its real colour edges, one piece per label. Tips that land on nothing get a local watershed in a window that widens until the structure closes. A cut is only accepted if its boundary sits on a real edge (gradient well above the region's own texture), so a label is never given an invented shape.
5. **Visual grounding (Gemini).** Where a pointer line is lost in a realistic painting, Gemini is asked where each remaining structure is (an inside point and a box), and GrabCut on a magnified crop traces it. Gemini also names unlabelled parts and writes the spoken descriptions.

**Tactile design rules.** We follow BANA-style guidelines, applied automatically and then *measured on the output*:
- lines at least 1 mm and outlines 1.5 mm;
- tiny parts enlarged to 6 mm or more;
- neighbouring parts always get different textures, with a 2 mm gap around every texture;
- UEB braille keys placed beside each part with a clear pad, plus a key page;
- a warning when a diagram has too many parts to read by touch.

**AI on images.** With Gemini on, Feelable names parts a textbook didn't label and writes spoken descriptions that say *where* each part is ("Top left, above the left ventricle…"). It also handles photos of real objects: Gemini's bounding boxes, combined with our own subject cut-out, turn a phone photo of a computer mouse into a touchable outline with its buttons named. Every AI sentence carries a visible badge, and the teacher approves before anything reaches a student.

**Stack:** Python, OpenCV, RapidOCR (PP-OCR), RapidFuzz, Gemini (vision + TTS), trimesh, FastAPI, React, TypeScript, three.js, and the Web Speech, Vibration and Web Audio APIs. Steps 1–4 run on our own server with no API key; step 5 and the descriptions need Gemini.

## Challenges we ran into
Real diagrams are messy: leader lines that stop short, a membrane broken by an overlapping organelle, OCR merging "Outer core" into "Outercore", tick numbers posing as bar values, phone screenshots of textbook figures at 400 px where "Mandible" comes out as "Mardtle". Dense anatomy figures were the hardest: two dozen labels in small print over bones of nearly the same colour. Each problem got a principled rule, a regression test, and a place in a benchmark. The hardest part wasn't generating an image — it was generating one **fingers** can read.

## Accomplishments we're proud of
- On a 24-image benchmark of dense anatomy figures (skulls, eye, ear, kidney, digestive system, heart, organ paintings, low-res screenshots), labels read correctly rose from 94 % to 99 % and labels that became a named, feelable shape from 52 % to 71 % — offline, before Gemini.
- Ten reviewed lessons (five basic, five complex: skull, cell, circuit, flowchart, bus map) open instantly, plus an extended library.
- Robustness-tested on 239 images from the internet: objects, animals, scenes, maps, charts, handwriting, odd file formats and phone-like distortions. No crashes, and junk output down from 192 to 114 images.
- 304 backend tests and end-to-end browser checks at desktop and phone sizes pass. Every exported 3D model is checked to be a clean, watertight solid. Every part speaks its own name in touch mode on both phone and desktop.

## Credits and AI tools used
- **Inside the product:** Google Gemini 3.5 Flash / Flash-Lite (vision grounding, part names, descriptions) and Gemini text-to-speech. PP-OCR models via RapidOCR on ONNX Runtime.
- **To build it:** Claude (Anthropic), through Claude Code, as an AI coding assistant for code, tests and documentation.
- **Libraries:** OpenCV, NumPy, Pillow, Shapely, trimesh, RapidFuzz, FastAPI, React, three.js, Vite and others. Full list in the README.
- **Images:** OpenStax (CC BY 4.0) figures, our own CC0 artwork, and public test images whose sources are listed in `qa/robustness/manifest.csv`.

## What's next
Testing with teachers of the visually impaired (QTVIs) and blind students; embosser export; a shared, teacher-approved library so each textbook diagram only has to be converted once; and more diagram types (maps, geometry, circuits with symbols). Feelable is free to try and the vision pipeline runs without paid APIs, so a school can use it from day one.
