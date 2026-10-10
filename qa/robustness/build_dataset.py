#!/usr/bin/env python3
"""Build the Feelable robustness dataset: download ~140 images from open-source repos,
render SVGs, synthesise odd files and phone-like variants, and write manifest.csv.

    python qa/robustness/build_dataset.py            # idempotent: skips files already present
    python qa/robustness/build_dataset.py --force    # re-download everything

Only raw.githubusercontent.com is needed. Images land in qa/robustness/images/<category>/.
Every row of manifest.csv records where the bytes came from and a rough licence note.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
GH = "https://raw.githubusercontent.com/"

OSX_BIO = GH + "philschatz/biology-book/master/resources/"
OSX_ANA = GH + "philschatz/anatomy-book/master/resources/"
OSX_PHY = GH + "philschatz/physics-book/master/resources/"
OSX_ECO = GH + "philschatz/economics-book/master/resources/"
OSX_SOC = GH + "philschatz/sociology-book/master/resources/"
OSX_HIS = GH + "philschatz/us-history-book/master/resources/"
CV = GH + "opencv/opencv/4.x/samples/data/"
SK = GH + "scikit-image/scikit-image/v0.22.0/skimage/data/"
DN = GH + "pjreddie/darknet/master/data/"
IN = GH + "EliSchwartz/imagenet-sample-images/master/"
OM = GH + "hfg-gmuend/openmoji/master/color/618x618/"
GHX = GH + "github/explore/main/topics/"
D2L = GH + "d2l-ai/d2l-en/master/img/"
MPL = GH + "matplotlib/matplotlib/main/lib/matplotlib/"
D3G = GH + "d3/d3-geo/main/test/snapshots/"
PIL = GH + "python-pillow/Pillow/11.0.0/Tests/images/"
HTR = GH + "githubharald/"
EOCR = GH + "JaidedAI/EasyOCR/master/examples/"
TESS = GH + "tesseract-ocr/test/main/testing/"

L_OSX = "CC BY 4.0 (OpenStax, via philschatz mirror)"
L_CV = "OpenCV samples (Apache-2.0 repo; third-party photos, test use)"
L_SK = "scikit-image data (CC0/public domain per skimage data README)"
L_DN = "darknet repo (public domain code; third-party photos, test use)"
L_IN = "ImageNet sample (non-commercial research/test use only)"
L_OM = "OpenMoji CC BY-SA 4.0"
L_GHX = "github/explore topic logo (trademark of owner; test use only)"
L_D2L = "d2l.ai CC BY-SA 4.0"
L_MPL = "matplotlib (PSF-style licence)"
L_D3 = "d3-geo ISC; Natural Earth data public domain"
L_PIL = "Pillow test suite (MIT-CMU); hopper photo public domain"
L_IAM = "IAM-style handwriting sample from githubharald repo (research/test use)"
L_EOCR = "EasyOCR examples (Apache-2.0 repo; test use)"
L_TESS = "tesseract-ocr test data (Apache-2.0)"

# (id, category, url, licence, expected)  -- expected = what a teacher would reasonably expect
SOURCES: list[tuple[str, str, str, str, str]] = [
    # ---------------------------------------------------------------- textbook labelled diagrams
    ("tb-animal-cell", "textbook", OSX_BIO + "Figure_04_03_01a.png", L_OSX, "labelled diagram: animal cell; nucleus, mitochondrion, ER, Golgi, cytoplasm, plasma membrane, lysosome"),
    ("tb-plant-cell", "textbook", OSX_BIO + "Figure_04_03_01b.jpg", L_OSX, "labelled diagram: plant cell; cell wall, chloroplast, central vacuole, nucleus, mitochondrion"),
    ("tb-chloroplast", "textbook", OSX_BIO + "Figure_04_03_10.jpg", L_OSX, "labelled diagram: chloroplast; outer/inner membrane, thylakoid, granum, stroma"),
    ("tb-chloroplast-2", "textbook", OSX_BIO + "Figure_08_01_05.png", L_OSX, "labelled diagram: chloroplast in leaf; grana, stroma, thylakoid"),
    ("tb-ribosome", "textbook", OSX_BIO + "Figure_04_03_06.jpg", L_OSX, "labelled diagram: ribosome; large subunit, small subunit, mRNA"),
    ("tb-bird-digestive", "textbook", OSX_BIO + "Figure_34_01_06.jpg", L_OSX, "labelled diagram: avian digestive system; esophagus, crop, proventriculus, gizzard, intestine, cloaca"),
    ("tb-ruminant", "textbook", OSX_BIO + "Figure_34_01_07.jpg", L_OSX, "labelled diagram: ruminant stomach; rumen, reticulum, omasum, abomasum"),
    ("tb-large-intestine", "textbook", OSX_BIO + "Figure_34_01_12.jpg", L_OSX, "labelled diagram: large intestine; ascending/transverse/descending colon, cecum, appendix, rectum"),
    ("tb-stomach", "textbook", OSX_BIO + "Figure_34_01_10f.png", L_OSX, "labelled diagram: stomach and small intestine; esophagus, stomach, duodenum, pancreas"),
    ("tb-heart-internal", "textbook", OSX_ANA + "2008_Internal_Anatomy_of_the_HeartN.jpg", L_OSX, "labelled diagram: heart interior; 4 chambers, aorta, valves, pulmonary trunk"),
    ("tb-heart-wall", "textbook", OSX_ANA + "2004_Heart_Wall.jpg", L_OSX, "labelled diagram: heart wall layers; pericardium, myocardium, endocardium"),
    ("tb-long-bone", "textbook", OSX_ANA + "603_Anatomy_of_a_Long_Bone.jpg", L_OSX, "labelled diagram: long bone; epiphysis, diaphysis, medullary cavity, periosteum"),
    ("tb-axial-skeleton", "textbook", OSX_ANA + "701_Axial_Skeleton-01.jpg", L_OSX, "labelled diagram: skeleton; skull, rib cage, vertebral column, sacrum"),
    ("tb-muscles", "textbook", OSX_ANA + "1102_Fascicle_Muscle_Shapes.jpg", L_OSX, "labelled diagram: body with muscle shapes labelled (parallel, convergent, pennate, circular...)"),
    ("tb-phloem", "textbook", OSX_BIO + "Figure_30_05_06.jpg", L_OSX, "labelled diagram: phloem; sieve-tube element, companion cell, sieve plate"),
    ("tb-pine-water", "textbook", OSX_BIO + "Figure_30_05_04.png", L_OSX, "labelled diagram: water transport in a tree; root, xylem, leaf, evaporation arrows"),
    # ---------------------------------------------------------------- unlabelled diagrams / clip art
    ("ul-apple-emoji", "unlabelled", OM + "1F34E.png", L_OM, "clip art: red apple; body, leaf, stem"),
    ("ul-tree-emoji", "unlabelled", OM + "1F333.png", L_OM, "clip art: tree; canopy, trunk"),
    ("ul-heart-emoji", "unlabelled", OM + "2764.png", L_OM, "clip art: heart shape; one part"),
    ("ul-car-emoji", "unlabelled", OM + "1F697.png", L_OM, "clip art: car; body, wheels, windows"),
    ("ul-house-emoji", "unlabelled", OM + "1F3E0.png", L_OM, "clip art: house; roof, walls, door, window"),
    ("ul-butterfly-emoji", "unlabelled", OM + "1F98B.png", L_OM, "clip art: butterfly; wings, body"),
    ("ul-fish-emoji", "unlabelled", OM + "1F41F.png", L_OM, "clip art: fish; body, fins, eye"),
    ("ul-rocket-emoji", "unlabelled", OM + "1F680.png", L_OM, "clip art: rocket; body, window, fins, flame"),
    ("ul-globe-emoji", "unlabelled", OM + "1F30D.png", L_OM, "clip art: globe; land vs ocean"),
    ("ul-brain-svg", "unlabelled", GH + "hfg-gmuend/openmoji/master/color/svg/1F9E0.svg", L_OM, "clip art: brain (SVG rendered to PNG); one outline with folds"),
    ("ul-shapes-pic1", "unlabelled", CV + "pic1.png", L_CV, "unlabelled shapes: several outlined geometric shapes"),
    ("ul-shapes-pic3", "unlabelled", CV + "pic3.png", L_CV, "unlabelled shapes: nested/concentric shapes"),
    ("ul-horse-silhouette", "unlabelled", SK + "horse.png", L_SK, "silhouette: horse; one solid shape"),
    ("ul-blobs", "unlabelled", CV + "detect_blob.png", L_CV, "unlabelled: grid of coloured blobs"),
    # ---------------------------------------------------------------- charts and graphs
    ("ch-demand-curve", "charts", OSX_ECO + "CNX_Econ_C03_001.jpg", L_OSX, "chart: demand curve + schedule table; axes, one downward line"),
    ("ch-supply-demand", "charts", OSX_ECO + "CNX_Econ_C03_003.jpg", L_OSX, "chart: supply & demand curves crossing at equilibrium"),
    ("ch-gdp-pie", "charts", OSX_ECO + "CNX_Econv1-2_C19_03.jpg", L_OSX, "pie chart: GDP components (consumption, investment, government, trade)"),
    ("ch-gdp-lines", "charts", OSX_ECO + "CNX_Econv1-2_C19_04.jpg", L_OSX, "line chart: GDP demand components over years (two panels)"),
    ("ch-pop-pyramid", "charts", OSX_SOC + "Figure_20_01_03.jpg", L_OSX, "bar chart: population pyramid by age and sex"),
    ("ch-africa-pop", "charts", OSX_SOC + "Figure_20_01_04.png", L_OSX, "chart: projected population growth, Africa"),
    ("ch-em-spectrum", "charts", OSX_BIO + "Figure_08_02_03.jpg", L_OSX, "labelled scale: electromagnetic spectrum bands (gamma ... radio)"),
    ("ch-ph-scale", "charts", OSX_BIO + "Figure_02_02_07.jpg", L_OSX, "labelled scale: pH 0-14 with example substances"),
    ("ch-mpl-bars", "charts", MPL + "tests/baseline_images/test_axes/bar_tick_label_multiple.png", L_MPL, "bar chart: 2 bars with tick labels"),
    ("ch-mpl-pie", "charts", MPL + "tests/baseline_images/test_axes/pie_frame_grid.png", L_MPL, "pie charts: several pies with slices"),
    ("ch-mpl-boxplot", "charts", MPL + "tests/baseline_images/test_axes/boxplot.png", L_MPL, "box plot: 2 boxes"),
    ("ch-mpl-errorbar", "charts", MPL + "tests/baseline_images/test_axes/errorbar_basic.png", L_MPL, "line chart with error bars"),
    ("ch-mpl-stackplot", "charts", MPL + "tests/baseline_images/test_axes/stackplot_test_image.png", L_MPL, "stacked area chart: 3 bands"),
    ("ch-mpl-legend", "charts", MPL + "tests/baseline_images/test_legend/legend_auto1.png", L_MPL, "line chart with legend: 2 series"),
    # ---------------------------------------------------------------- maps
    ("mp-us-confederacy", "maps", OSX_HIS + "CNX_History_15_01_Confed.jpg", L_OSX, "labelled map: US states, Union vs Confederacy regions"),
    ("mp-crittenden", "maps", OSX_HIS + "CNX_History_15_01_Crittenden.jpg", L_OSX, "map: US with free/slave regions split by a line"),
    ("mp-new-netherland", "maps", OSX_HIS + "CNX_History_03_02_Map.jpg", L_OSX, "historic map (1684 engraving): coastline, settlements; hard"),
    ("mp-world-chlorophyll", "maps", OSX_BIO + "Figure_08_00_01.jpg", L_OSX, "world map: colour-graded chlorophyll; continents vs ocean"),
    ("mp-d3-albers-usa", "maps", D3G + "albersUsa.png", L_D3, "outline map: US states (Albers USA)"),
    ("mp-d3-mercator", "maps", D3G + "mercator.png", L_D3, "outline map: world countries (Mercator)"),
    ("mp-d3-orthographic", "maps", D3G + "orthographic.png", L_D3, "outline map: globe (orthographic)"),
    ("mp-d3-equirect", "maps", D3G + "equirectangular.png", L_D3, "outline map: world (equirectangular)"),
    ("mp-natural-earth", "maps", GH + "SciTools/cartopy/main/lib/cartopy/data/raster/natural_earth/50-natural-earth-1-downsampled.png", "cartopy BSD-3; Natural Earth public domain", "shaded-relief world raster; continents vs ocean"),
    ("mp-leaflet-choropleth", "maps", GH + "Leaflet/Leaflet/main/docs/examples/choropleth/thumbnail.png", "Leaflet BSD-2; tiles (c) OpenStreetMap contributors", "web map screenshot: US states choropleth"),
    ("mp-leaflet-street", "maps", GH + "Leaflet/Leaflet/main/docs/examples/quick-start/thumbnail.png", "Leaflet BSD-2; tiles (c) OpenStreetMap contributors", "street map screenshot with marker/circle/polygon"),
    # ---------------------------------------------------------------- flowcharts / circuits
    ("fc-simple-circuit", "flow_circuits", OSX_PHY + "Figure_21_02_01a.jpg", L_OSX, "circuit: battery + resistor loop with current arrow"),
    ("fc-series-circuit", "flow_circuits", OSX_PHY + "Figure_22_01_02.jpg", L_OSX, "circuit schematic: 3 resistors in series with battery (two panels)"),
    ("fc-mixed-circuit", "flow_circuits", OSX_PHY + "Figure_22_01_06.jpg", L_OSX, "circuit schematic: R1 in series with R2||R3, 12 V source"),
    ("fc-switch-circuit", "flow_circuits", OSX_PHY + "Figure_22_01_08.jpg", L_OSX, "circuit schematic: source, internal r, R, switch"),
    ("fc-protein-digestion", "flow_circuits", OSX_BIO + "Figure_34_03_02.jpg", L_OSX, "flowchart: protein digestion steps (boxes + arrows)"),
    ("fc-carb-digestion", "flow_circuits", OSX_BIO + "Figure_34_03_01.jpg", L_OSX, "flowchart: carbohydrate breakdown pathways"),
    ("fc-calcium", "flow_circuits", OSX_ANA + "625_Calcium_Homeostasis.jpg", L_OSX, "feedback-loop flowchart: calcium homeostasis"),
    ("fc-d2l-rnn", "flow_circuits", D2L + "rnn.svg", L_D2L, "block diagram: unrolled RNN (boxes, arrows, labels)"),
    ("fc-d2l-mlloop", "flow_circuits", D2L + "ml-loop.svg", L_D2L, "cycle diagram: ML loop (data, model, deploy...)"),
    ("fc-d2l-transformer", "flow_circuits", D2L + "transformer.svg", L_D2L, "architecture block diagram: transformer encoder/decoder"),
    ("fc-d2l-lenet", "flow_circuits", D2L + "lenet.svg", L_D2L, "pipeline diagram: LeNet layers"),
    ("fc-k8s-components", "flow_circuits", GH + "kubernetes/website/main/static/images/docs/components-of-kubernetes.svg", "Kubernetes docs CC BY 4.0", "architecture diagram: control plane + nodes with icons"),
    # ---------------------------------------------------------------- photos of single objects
    ("ob-mouse", "objects", IN + "n03793489_mouse.JPEG", L_IN, "photo: computer mouse; one object (buttons, wheel at most)"),
    ("ob-mug", "objects", IN + "n03063599_coffee_mug.JPEG", L_IN, "photo: coffee mug; body, handle"),
    ("ob-phone", "objects", IN + "n02992529_cellular_telephone.JPEG", L_IN, "photo: mobile phone; one object"),
    ("ob-shoe", "objects", IN + "n04120489_running_shoe.JPEG", L_IN, "photo: running shoe; one object"),
    ("ob-banana", "objects", IN + "n07753592_banana.JPEG", L_IN, "photo: bananas; one object/bunch"),
    ("ob-granny-smith", "objects", IN + "n07742313_Granny_Smith.JPEG", L_IN, "photo: green apple; one object"),
    ("ob-laptop", "objects", IN + "n03642806_laptop.JPEG", L_IN, "photo: laptop; screen, keyboard"),
    ("ob-wall-clock", "objects", IN + "n04548280_wall_clock.JPEG", L_IN, "photo: wall clock; face, hands"),
    ("ob-keyboard", "objects", IN + "n03085013_computer_keyboard.JPEG", L_IN, "photo: keyboard; one object (keys too fine)"),
    ("ob-water-bottle", "objects", IN + "n04557648_water_bottle.JPEG", L_IN, "photo: water bottle; one object"),
    ("ob-soccer-ball", "objects", IN + "n04254680_soccer_ball.JPEG", L_IN, "photo: soccer ball; one round object"),
    ("ob-teapot", "objects", IN + "n04398044_teapot.JPEG", L_IN, "photo: teapot; body, spout, handle, lid"),
    ("ob-guitar", "objects", IN + "n02676566_acoustic_guitar.JPEG", L_IN, "photo: acoustic guitar; body, neck, sound hole"),
    ("ob-coffee", "objects", SK + "coffee.png", L_SK, "photo: cup of coffee on saucer; cup, saucer, spoon"),
    ("ob-orange", "objects", CV + "orange.jpg", L_CV, "photo: orange fruit; one round object"),
    ("ob-fruits", "objects", CV + "fruits.jpg", L_CV, "photo: several fruits on a table; each fruit an outline"),
    ("ob-rocket", "objects", SK + "rocket.jpg", L_SK, "photo: rocket on launch pad; rocket, tower"),
    ("ob-box", "objects", CV + "box.png", L_CV, "grayscale photo: cereal box; one rectangle (printed text = packaging, not labels)"),
    # ---------------------------------------------------------------- animals / people
    ("ap-chelsea-cat", "animals_people", SK + "chelsea.png", L_SK, "photo: cat face; one animal outline (eyes, ears at most)"),
    ("ap-astronaut", "animals_people", SK + "astronaut.png", L_SK, "photo: astronaut portrait with flag; person outline"),
    ("ap-tabby", "animals_people", IN + "n02123045_tabby.JPEG", L_IN, "photo: tabby cat; one animal"),
    ("ap-zebra", "animals_people", IN + "n02391049_zebra.JPEG", L_IN, "photo: zebra; one animal (stripes are texture, not parts)"),
    ("ap-elephant", "animals_people", IN + "n02504458_African_elephant.JPEG", L_IN, "photo: elephant; one animal"),
    ("ap-dog-bike", "animals_people", DN + "dog.jpg", L_DN, "photo: dog, bicycle, truck; 3 objects"),
    ("ap-horses", "animals_people", DN + "horses.jpg", L_DN, "photo: several horses in a field"),
    ("ap-eagle", "animals_people", DN + "eagle.jpg", L_DN, "photo: eagle in flight; one bird"),
    ("ap-giraffe", "animals_people", DN + "giraffe.jpg", L_DN, "photo: giraffe and zebra"),
    ("ap-person-dog-horse", "animals_people", DN + "person.jpg", L_DN, "photo: person, dog, horse"),
    ("ap-messi", "animals_people", CV + "messi5.jpg", L_CV, "photo: footballer on pitch; person + ball"),
    ("ap-baboon", "animals_people", CV + "baboon.jpg", L_CV, "photo: mandrill face; one animal face"),
    ("ap-zidane", "animals_people", GH + "ultralytics/assets/main/im/zidane.jpg", "ultralytics assets (AGPL-3.0 repo; test use)", "photo: two men in suits; 2 people"),
    ("ap-grace-hopper", "animals_people", MPL + "mpl-data/sample_data/grace_hopper.jpg", L_MPL + "; photo public domain (US Navy)", "photo: portrait of a person"),
    ("ap-butterfly", "animals_people", CV + "butterfly.jpg", L_CV, "photo: butterfly on flower; wings, body"),
    # ---------------------------------------------------------------- scenes
    ("sc-building", "scenes", CV + "building.jpg", L_CV, "photo: building facade with windows; too busy, expect a simplified outline"),
    ("sc-seashore", "scenes", IN + "n09428293_seashore.JPEG", L_IN, "photo: seashore; sea, sand, sky bands"),
    ("sc-valley", "scenes", IN + "n09468604_valley.JPEG", L_IN, "photo: valley landscape; hills, sky"),
    ("sc-library", "scenes", IN + "n03661043_library.JPEG", L_IN, "photo: library interior; shelves"),
    ("sc-restaurant", "scenes", IN + "n04081281_restaurant.JPEG", L_IN, "photo: restaurant interior; tables"),
    ("sc-volcano", "scenes", IN + "n09472597_volcano.JPEG", L_IN, "photo: volcano; mountain cone, sky"),
    ("sc-lakeside", "scenes", IN + "n09332890_lakeside.JPEG", L_IN, "photo: lakeside; water, trees, sky"),
    ("sc-streetcar", "scenes", IN + "n04335435_streetcar.JPEG", L_IN, "photo: street with tram"),
    ("sc-bus-street", "scenes", GH + "ultralytics/assets/main/im/bus.jpg", "ultralytics assets (AGPL-3.0 repo; test use)", "photo: street with bus and people"),
    ("sc-kite-beach", "scenes", DN + "kite.jpg", L_DN, "photo: beach with people flying kites (large file)"),
    ("sc-starry-night", "scenes", CV + "starry_night.jpg", L_CV + "; painting public domain", "painting: Starry Night; sky swirls, village, cypress"),
    ("sc-grocery", "scenes", OSX_BIO + "Figure_08_01_07.jpg", L_OSX, "photo: grocery store aisle with people"),
    ("sc-call-center", "scenes", OSX_SOC + "Figure_10_01_02.jpg", L_OSX, "photo: office call centre, many desks"),
    ("sc-patio", "scenes", IN + "n03899768_patio.JPEG", L_IN, "photo: patio with furniture"),
    # ---------------------------------------------------------------- text pages / screenshots
    ("tx-page", "text_screens", SK + "page.png", L_SK, "scanned text page (unevenly lit); text only, nothing to draw"),
    ("tx-skimage-text", "text_screens", SK + "text.png", L_SK, "photo of text on a surface; text only"),
    ("tx-phototest", "text_screens", TESS + "phototest.tif", L_TESS, "text page (TIFF): paragraph of printed text"),
    ("tx-eurotext", "text_screens", TESS + "eurotext.tif", L_TESS, "text page (TIFF): multilingual paragraphs"),
    ("tx-hebrew", "text_screens", TESS + "hebrew.png", L_TESS, "text page: Hebrew paragraphs (RTL)"),
    ("tx-easyocr-sign", "text_screens", EOCR + "english.png", L_EOCR, "photo of street sign with English text (large PNG)"),
    ("tx-easyocr-chinese", "text_screens", EOCR + "chinese.jpg", L_EOCR, "photo of Chinese sign"),
    ("tx-easyocr-french", "text_screens", EOCR + "french.jpg", L_EOCR, "photo of French sign"),
    ("tx-imagetext", "text_screens", CV + "imageTextN.png", L_CV, "rendered text lines"),
    ("tx-sudoku", "text_screens", CV + "sudoku.png", L_CV, "photo of printed sudoku grid with digits"),
    ("tx-btop-screen", "text_screens", GH + "aristocratos/btop/main/Img/normal.png", "btop Apache-2.0", "UI screenshot: terminal system monitor with graphs"),
    ("tx-website", "text_screens", IN + "n06359193_web_site.JPEG", L_IN, "screenshot of a web page"),
    ("tx-comic", "text_screens", IN + "n06596364_comic_book.JPEG", L_IN, "photo of comic book cover; busy"),
    # ---------------------------------------------------------------- handwriting / whiteboards
    ("hw-digits", "handwriting", CV + "digits.png", L_CV + "; MNIST-style digits", "sheet of 5000 handwritten digits; a grid, nothing meaningful to draw"),
    ("hw-htr-word", "handwriting", HTR + "SimpleHTR/master/data/word.png", L_IAM, "single handwritten word"),
    ("hw-htr-line", "handwriting", HTR + "SimpleHTR/master/data/line.png", L_IAM, "single handwritten line"),
    ("hw-htr-sample1", "handwriting", HTR + "HTRPipeline/master/data/sample_1.png", L_IAM, "handwritten text page/lines"),
    ("hw-htr-sample2", "handwriting", HTR + "HTRPipeline/master/data/sample_2.png", L_IAM, "photo of handwritten notes (large)"),
    ("hw-worddetector-line", "handwriting", HTR + "WordDetector/master/data/line/0.png", L_IAM, "handwritten line of words"),
    ("hw-notebook-page", "handwriting", GH + "Breta01/handwriting-ocr/master/data/pages/page01.jpg", "handwriting-ocr repo (MIT; test use)", "phone photo of handwritten notebook page (2.7 MB)"),
    ("hw-deslant", "handwriting", HTR + "DeslantImg/master/doc/example.png", L_IAM, "slanted handwriting before/after"),
    ("hw-excalidraw", "handwriting", GH + "excalidraw/excalidraw/master/public/og-image-3.png", "excalidraw MIT", "hand-drawn whiteboard-style sketch (shapes + arrows + handwriting font)"),
    ("hw-tldraw", "handwriting", GH + "tldraw/tldraw/main/assets/github-hero-light.png", "tldraw licence (test use)", "whiteboard app screenshot with hand-drawn shapes"),
    # ---------------------------------------------------------------- logos / icons
    ("lg-python", "logos_icons", GHX + "python/python.png", L_GHX, "logo: Python two-snake mark; 2 parts"),
    ("lg-react", "logos_icons", GHX + "react/react.png", L_GHX, "logo: React atom; orbits + nucleus"),
    ("lg-docker", "logos_icons", GHX + "docker/docker.png", L_GHX, "logo: Docker whale with containers"),
    ("lg-rust", "logos_icons", GHX + "rust/rust.png", L_GHX, "logo: Rust gear R; one shape"),
    ("lg-linux", "logos_icons", GHX + "linux/linux.png", L_GHX, "logo: Tux penguin; body, belly, beak, feet"),
    ("lg-android", "logos_icons", GHX + "android/android.png", L_GHX, "logo: Android robot (tiny PNG)"),
    ("lg-opencv", "logos_icons", CV + "opencv-logo.png", L_CV, "logo: 3 coloured rings + wordmark"),
    ("lg-windows", "logos_icons", CV + "WindowsLogo.jpg", L_CV, "logo: Windows flag, 4 panes"),
    ("lg-skimage", "logos_icons", SK + "logo.png", L_SK, "logo: scikit-image snake + text"),
    ("lg-github-svg", "logos_icons", GH + "simple-icons/simple-icons/develop/icons/github.svg", "simple-icons CC0 (trademark of owner)", "icon: GitHub octocat mark (SVG rendered); one shape"),
    ("lg-camera-svg", "logos_icons", GH + "feathericons/feather/main/icons/camera.svg", "Feather icons MIT", "line icon: camera (SVG rendered); body + lens"),
    ("lg-twemoji-snail", "logos_icons", GH + "twitter/twemoji/master/assets/svg/1f40c.svg", "Twemoji CC BY 4.0", "emoji: snail (SVG rendered); shell + body"),
    # ---------------------------------------------------------------- weird files
    ("wf-cmyk-jpeg", "weird", PIL + "pil_sample_cmyk.jpg", L_PIL, "CMYK JPEG: should decode correctly (colours not inverted)"),
    ("wf-cmyk-tiff", "weird", PIL + "tiff_strip_cmyk_jpeg.tif", L_PIL, "CMYK JPEG-in-TIFF: decode or clean error"),
    ("wf-16bit-tiff", "weird", PIL + "16bit.cropped.tif", L_PIL, "16-bit grayscale TIFF: decode, not black"),
    ("wf-16bit-rgb-tiff", "weird", PIL + "tiff_16bit_RGB.tiff", L_PIL, "16-bit RGB TIFF: decode or clean error"),
    ("wf-16bit-pgm", "weird", PIL + "hopper_16bit.pgm", L_PIL, "16-bit PGM: decode or clean error"),
    ("wf-palette-png", "weird", PIL + "pil123p.png", L_PIL, "palette PNG: decode"),
    ("wf-alpha-png", "weird", PIL + "transparent.png", L_PIL, "RGBA PNG with transparency: composite on white"),
    ("wf-alpha-text", "weird", PIL + "transparent_background_text.png", L_PIL, "transparent PNG with black text: text visible on white"),
    ("wf-animated-gif", "weird", PIL + "iss634.gif", L_PIL, "animated GIF: use first frame"),
    ("wf-animated-webp", "weird", PIL + "iss634.webp", L_PIL, "animated WebP: use first frame"),
    ("wf-apng", "weird", PIL + "iss634.apng", L_PIL, "animated PNG: use first frame"),
    ("wf-webp", "weird", PIL + "flower.webp", L_PIL, "WebP photo of a flower"),
    ("wf-tiff", "weird", PIL + "hopper.tif", L_PIL, "TIFF photo (hopper)"),
    ("wf-bmp", "weird", PIL + "hopper.bmp", L_PIL, "BMP photo (hopper)"),
    ("wf-ico", "weird", PIL + "hopper.ico", L_PIL, "ICO icon: decode or clean error"),
    ("wf-psd", "weird", PIL + "hopper.psd", L_PIL, "Photoshop PSD: decode or clean error"),
    ("wf-mpo", "weird", PIL + "sugarshack.mpo", L_PIL, "MPO (multi-picture JPEG from 3D camera): decode first image"),
    ("wf-multipage-tiff", "weird", PIL + "multipage.tiff", L_PIL, "multi-page TIFF: first page or clean error"),
    ("wf-exif-orient6", "weird", PIL + "hopper_orientation_6.jpg", L_PIL, "JPEG with EXIF orientation 6: must be rotated upright"),
    ("wf-gray-jpeg", "weird", PIL + "hopper_gray.jpg", L_PIL, "grayscale JPEG"),
    ("wf-tiny-png", "weird", PIL + "tiny.png", L_PIL, "tiny PNG: clean error or trivial result"),
    ("wf-truncated-png", "weird", PIL + "truncated_image.png", L_PIL, "truncated PNG: clean error"),
    ("wf-truncated-jpeg", "weird", PIL + "truncated_jpeg.jpg", L_PIL, "truncated JPEG: clean error (or partial decode)"),
    ("wf-broken-png", "weird", PIL + "broken.png", L_PIL, "broken PNG: clean error"),
    ("wf-decompression-bomb", "weird", PIL + "decompression_bomb.gif", L_PIL, "GIF claiming huge dimensions: clean error, no OOM"),
    ("wf-raw-svg", "weird", GH + "feathericons/feather/main/icons/heart.svg", "Feather icons MIT", "raw SVG bytes uploaded as-is: clean 'not an image' error"),
]

# Local (non-internet) fixtures from the external audit -- the real failing user photo etc.
AUDIT_DIR = os.path.normpath(os.path.join(HERE, "..", "audit", "reproduce"))
LOCAL = [
    ("fx-user-mouse-photo", "fixtures", "mouse-photo.jpg", "user photo (audit fixture)", "real user phone photo: mouse on red floral mouse pad; expect ONE part 'mouse' (or unnamed), no brand OCR names"),
    ("fx-four-shapes", "fixtures", "four-shapes.png", "audit fixture (generated)", "labelled: circle, square, triangle, ellipse -> 4 named parts"),
    ("fx-four-shapes-sideways", "fixtures", "four-shapes-sideways.png", "audit fixture (generated)", "same 4 shapes rotated 90 deg; 4 named parts"),
    ("fx-four-shapes-gray", "fixtures", "four-shapes-grayscale.png", "audit fixture (generated)", "grayscale 4 shapes; 4 named parts"),
    ("fx-blank-page", "fixtures", "blank-page.png", "audit fixture (generated)", "blank page: 0 parts + clear 'nothing found' warning"),
]


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "feelable-qa/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def ext_for(url: str) -> str:
    base = url.rsplit("/", 1)[-1]
    ext = os.path.splitext(base)[1].lower()
    return ext if ext else ".bin"


def render_svg(data: bytes, width: int = 1200) -> bytes:
    import cairosvg
    from PIL import Image
    png = cairosvg.svg2png(bytestring=data, output_width=width)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    out = io.BytesIO()
    bg.convert("RGB").save(out, "PNG", optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------- synthetic weird files and phone variants
def _pil(path):
    from PIL import Image, ImageOps
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        # flatten onto white like a viewer/phone screenshot would (not black, which hides dark labels)
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        im = bg
    return im.convert("RGB")


def _save_jpeg(im, path, q=88, exif=None):
    kw = {"quality": q}
    if exif is not None:
        kw["exif"] = exif
    im.save(path, "JPEG", **kw)


def make_synthetic(rows: list[dict]) -> None:
    """Odd sizes/contents derived from downloaded images (recorded as variants of their source)."""
    import numpy as np
    from PIL import Image, ImageEnhance
    by_id = {r["id"]: r for r in rows}
    d = os.path.join(IMG, "weird")
    os.makedirs(d, exist_ok=True)

    def add(id_, fname, src_id, expected, builder):
        path = os.path.join(d, fname)
        if not os.path.exists(path):
            builder(path)
        rows.append({"id": id_, "category": "weird", "source_url": f"derived:{src_id} <- {by_id[src_id]['source_url']}",
                     "licence_note": by_id[src_id]["licence_note"], "expected": expected,
                     "file": os.path.relpath(path, HERE), "variant_of": src_id})

    cell = os.path.join(HERE, by_id["tb-animal-cell"]["file"])
    bldg = os.path.join(HERE, by_id["sc-building"]["file"])
    page = os.path.join(HERE, by_id["tx-page"]["file"])
    mug = os.path.join(HERE, by_id["ob-mug"]["file"])

    add("wf-tiny-8x8", "tiny-8x8.png", "tb-animal-cell", "8x8 px image: clean 'too small' error or trivial result",
        lambda p: _pil(cell).resize((8, 8), Image.LANCZOS).save(p))
    add("wf-huge-12000", "huge-12000x9000.jpg", "sc-building", "12000x9000 JPEG (108 MP): must not OOM in 512 MB; downscale early or clean error",
        lambda p: _save_jpeg(_pil(bldg).resize((12000, 9000), Image.BILINEAR), p, q=60))
    add("wf-strip-wide", "strip-6000x80.png", "tb-animal-cell", "extreme aspect 75:1 strip: clean error or sensible result",
        lambda p: _pil(cell).resize((6000, 80), Image.LANCZOS).save(p))
    add("wf-strip-tall", "strip-60x4000.png", "tb-animal-cell", "extreme aspect 1:66 strip: clean error or sensible result",
        lambda p: _pil(cell).resize((60, 4000), Image.LANCZOS).save(p))
    add("wf-near-blank", "near-blank.png", "tx-page", "near-blank page (contrast 3%): 0 parts + 'nothing found' warning",
        lambda p: ImageEnhance.Contrast(_pil(page)).enhance(0.03).save(p))
    add("wf-all-black", "all-black.jpg", "ob-mug", "all-black photo (lens cap): 0 parts + warning",
        lambda p: _save_jpeg(ImageEnhance.Brightness(_pil(mug)).enhance(0.01), p))
    add("wf-noise", "noise.png", "ob-mug", "pure random noise 800x600: 0 parts or 1 junk part + warning",
        lambda p: Image.fromarray(np.random.default_rng(0).integers(0, 256, (600, 800, 3), dtype=np.uint8)).save(p))
    add("wf-zero-bytes", "zero-bytes.jpg", "ob-mug", "empty file: clean error",
        lambda p: open(p, "wb").close())
    add("wf-text-as-jpg", "text-renamed.jpg", "ob-mug", "a text file renamed .jpg: clean error",
        lambda p: open(p, "wb").write(b"This is not an image, it is a note.\n" * 20))
    add("wf-jpeg-head-only", "jpeg-first-2kb.jpg", "ob-mug", "JPEG cut after 2 KB (bad upload): clean error or partial",
        lambda p: open(p, "wb").write(open(mug, "rb").read()[:2048]))


PHONE_BASES = [
    "tb-animal-cell", "tb-heart-internal", "tb-bird-digestive", "fc-mixed-circuit", "ch-demand-curve",
    "mp-us-confederacy", "ob-mouse", "ob-mug", "ul-apple-emoji", "lg-python", "tx-page", "hw-htr-sample1",
    "ap-chelsea-cat", "fx-four-shapes", "fc-protein-digestion", "ch-pop-pyramid", "tb-long-bone",
    "ul-house-emoji", "mp-d3-albers-usa", "sc-volcano", "ob-teapot", "tb-plant-cell",
]

# variant name -> description
VARIANTS = ["rot90", "rot180", "tilt", "blur", "glare", "jpegq30", "lowlight", "phone4032", "exif6"]


def _variant(im, kind: str):
    import numpy as np
    import cv2
    from PIL import Image, ImageFilter, ImageEnhance
    if kind == "rot90":
        return im.rotate(90, expand=True), None
    if kind == "rot180":
        return im.rotate(180, expand=True), None
    if kind == "tilt":
        a = np.asarray(im)
        h, w = a.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * 0.08, h * 0.05], [w * 0.95, 0], [w, h], [0, h * 0.93]])
        M = cv2.getPerspectiveTransform(src, dst)
        # put the "page" on a grey-brown desk
        out = cv2.warpPerspective(a, M, (w, h), borderValue=(120, 105, 90))
        return Image.fromarray(out), None
    if kind == "blur":
        return im.filter(ImageFilter.GaussianBlur(radius=max(2, im.width / 300))), None
    if kind == "glare":
        a = np.asarray(im).astype(np.float32)
        h, w = a.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        g = np.exp(-(((xx - w * 0.65) / (w * 0.22)) ** 2 + ((yy - h * 0.35) / (h * 0.2)) ** 2))
        a = a + (255 - a) * g[..., None] * 0.9
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), None
    if kind == "jpegq30":
        return im, 30
    if kind == "lowlight":
        a = np.asarray(ImageEnhance.Brightness(im).enhance(0.28)).astype(np.float32)
        a += np.random.default_rng(1).normal(0, 6, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)), None
    if kind == "phone4032":
        w, h = im.size
        if w >= h:
            return im.resize((4032, 3024), Image.BICUBIC), None
        return im.resize((3024, 4032), Image.BICUBIC), None
    raise ValueError(kind)


def make_phone_variants(rows: list[dict]) -> None:
    from PIL import Image
    by_id = {r["id"]: r for r in rows}
    d = os.path.join(IMG, "phone_variants")
    os.makedirs(d, exist_ok=True)
    # each base gets 2-3 variants, rotating through the list so every kind is used >= 4 times
    k = 0
    for base in PHONE_BASES:
        src = by_id[base]
        im = _pil(os.path.join(HERE, src["file"]))
        n = 3 if base in ("ob-mouse", "tb-animal-cell", "fc-mixed-circuit", "tx-page", "ob-mug") else 2
        for _ in range(n):
            kind = VARIANTS[k % len(VARIANTS)]
            k += 1
            vid = f"pv-{base}-{kind}"
            path = os.path.join(d, f"{base}-{kind}.jpg")
            if not os.path.exists(path):
                if kind == "exif6":
                    # store pixels rotated 90 deg CCW + EXIF Orientation=6: an upright viewer shows the original
                    exif = Image.Exif()
                    exif[0x0112] = 6
                    _save_jpeg(im.rotate(90, expand=True), path, q=88, exif=exif.tobytes())
                else:
                    out, q = _variant(im, kind)
                    _save_jpeg(out, path, q=q or 85)
            rows.append({"id": vid, "category": "phone_variants",
                         "source_url": f"derived:{base}:{kind} <- {src['source_url']}",
                         "licence_note": src["licence_note"],
                         "expected": f"[variant {kind}] same as {base}: {src['expected']}",
                         "file": os.path.relpath(path, HERE), "variant_of": base})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--audit-dir", default=AUDIT_DIR, help="folder with the audit reproduce/ fixtures")
    args = ap.parse_args()
    if args.force and os.path.isdir(IMG):
        shutil.rmtree(IMG)
    rows: list[dict] = []
    failed = []
    for id_, cat, url, lic, exp in SOURCES:
        d = os.path.join(IMG, cat)
        os.makedirs(d, exist_ok=True)
        is_svg = url.endswith(".svg") and cat != "weird"
        path = os.path.join(d, id_ + (".png" if is_svg else ext_for(url)))
        if not os.path.exists(path):
            try:
                data = fetch(url)
                if is_svg:
                    data = render_svg(data)
                with open(path, "wb") as f:
                    f.write(data)
                print("got", id_, len(data))
            except Exception as e:  # keep going; report at the end
                failed.append((id_, url, str(e)))
                continue
        rows.append({"id": id_, "category": cat, "source_url": url, "licence_note": lic + (" [SVG rendered to PNG]" if is_svg else ""),
                     "expected": exp, "file": os.path.relpath(path, HERE), "variant_of": ""})
    for id_, cat, fname, lic, exp in LOCAL:
        src = os.path.join(args.audit_dir, fname)
        d = os.path.join(IMG, cat)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, id_ + os.path.splitext(fname)[1])
        if not os.path.exists(path):
            if not os.path.exists(src):
                failed.append((id_, src, "missing local fixture"))
                continue
            shutil.copyfile(src, path)
        rows.append({"id": id_, "category": cat, "source_url": "local:qa/audit/reproduce/" + fname, "licence_note": lic,
                     "expected": exp, "file": os.path.relpath(path, HERE), "variant_of": ""})
    make_synthetic(rows)
    make_phone_variants(rows)
    with open(os.path.join(HERE, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "source_url", "licence_note", "expected", "file", "variant_of"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} images in manifest; {len(failed)} failed")
    for x in failed:
        print("FAILED", *x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
