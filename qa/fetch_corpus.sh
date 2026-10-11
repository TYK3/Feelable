#!/usr/bin/env bash
# Fetch the 41 labelled OpenStax figures used by the bulk test (CC BY 4.0, OpenStax / Rice
# University) into $1 (default: ./corpus). Needs git and network access to github.com.
#   bash qa/fetch_corpus.sh /tmp/corpus && cd backend && python ../qa/bulk_eval.py OUT /tmp/corpus/*
set -euo pipefail
OUT="${1:-corpus}"; mkdir -p "$OUT"; TMP="$(mktemp -d)"
get() { # repo files...
  local repo="$1"; shift
  git clone -q --filter=blob:none --no-checkout --depth 1 "https://github.com/openstax/$repo.git" "$TMP/$repo"
  for f in "$@"; do git -C "$TMP/$repo" show "HEAD:media/$f" > "$OUT/os-$(echo "$f" | tr 'A-Z' 'a-z')"; done
}
get osbooks-anatomy-physiology 700_Lateral_View_of_Skull-01.jpg 705_Lateral_View_of_Skull-01.jpg 704_Skull-01.jpg \
  726_Mandible.jpg 713_Bones_Forming_Orbit.jpg 730_Posterior_View_Skull.jpg 2008_Internal_Anatomy_of_the_HeartN.jpg \
  2005_Surface_Anatomy_of_the_Heart.jpg 2011_Heart_Valves.jpg 2018_Conduction_System_of_Heart.jpg \
  2401_Components_of_the_Digestive_System.jpg 2414_Stomach.jpg 2420_Large_Intestine.jpg 2409_Tooth.jpg \
  2301_Major_Respiratory_Organs.jpg 2312_Gross_Anatomy_of_the_Lungs.jpg 2306_The_Larynx.jpg 2610_The_Kidney.jpg \
  1413_Structure_of_the_Eye.jpg 1404_The_Structures_of_the_Ear.jpg 1206_The_Neuron.jpg 0312_Animal_Cell_and_Components.jpg \
  603_Anatomy_of_a_Long_Bone.jpg 1801_The_Endocrine_System.jpg 1820_The_Pancreas.jpg 2201_Anatomy_of_the_Lymphatic_System.jpg \
  718_Vertebra.jpg 807_Pelvis.jpg 812_Bones_of_the_Foot.jpg 1311_Brain_Stem.jpg 2303_Anatomy_of_Nose-Pharynx-Mouth-Larynx.jpg \
  1402_The_Tongue.jpg
get osbooks-biology-bundle eye_diagram.png ear_anatomy.png kidney_section.png
get osbooks-university-physics-bundle CNX_UPhysics_27_01_FourResist.jpg CNX_UPhysics_27_01_ResistPar2.jpg \
  CNX_UPhysics_27_01_SeriesRes.jpg CNX_UPhysics_27_02_Circuit.jpg CNX_UPhysics_27_02_KirchVolt.jpg CNX_UPhysics_25_03_Circuits.jpg
rm -rf "$TMP"; ls "$OUT" | wc -l
