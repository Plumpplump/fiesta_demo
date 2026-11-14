"""Glasses decision rules (the model itself is exported separately)."""

import numpy as np

from models import constants as C
from models.glasses import GlassesClassifier, decide_glasses
from models.types import GLASSES, NO_GLASSES, UNKNOWN


def test_positive_margin_means_glasses():
    assert decide_glasses(0.30).label == GLASSES


def test_negative_margin_means_no_glasses():
    assert decide_glasses(-0.10).label == NO_GLASSES


def test_borderline_margin_is_unsure():
    verdict = decide_glasses(C.GLASSES_SIM_THRESHOLD - C.GLASSES_SIM_UNSURE_MARGIN / 2)
    assert verdict.label == "unsure"


def test_no_model_reports_unknown_not_a_guess():
    assert decide_glasses(None).label == UNKNOWN


def test_classifier_without_model_is_unknown_for_a_real_crop():
    classifier = GlassesClassifier()
    crop = np.zeros((40, 80, 3), dtype=np.uint8)
    assert classifier.classify(crop).label == UNKNOWN


def test_tiny_crop_is_rejected_before_any_model_call():
    classifier = GlassesClassifier()
    crop = np.zeros((5, 10, 3), dtype=np.uint8)
    verdict = classifier.classify(crop)
    assert verdict.label == UNKNOWN
    assert "small" in verdict.detail