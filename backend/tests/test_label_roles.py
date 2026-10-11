"""F07: captions, view labels and equation fragments must not become missing-part suggestions."""
import pytest
from engine.vision_utils import looks_like_structure


@pytest.mark.parametrize("text", ["Right lateral view", "Sagittal section", "Vp v- v", "I = V/R",
                                  "channels connect", "two plant cells"])
def test_non_structures_are_rejected(text):
    assert not looks_like_structure(text)


@pytest.mark.parametrize("text", ["pancreas", "left atrium", "endoplasmic reticulum", "T cell",
                                  "lacrimal fossa", "coronal suture", "sagittal sinus",
                                  "greater wing of sphenoid bone", "mental protuberance of mandible"])
def test_real_structures_still_pass(text):
    assert looks_like_structure(text)
