# Author: David Li, SISHK
"""Glasses decision rules (the model itself is exported separately)."""

import numpy as np

from models import constants as C
from models.glasses import GlassesClassifier, decide_glasses
from models.types import GLASSES, NO_GLASSES, UNKNOWN


def test_confident_probability_means_glasses():
    assert decide_glasses(0.30).label == GLASSES


def test_low_probability_means_no_glasses():
    assert decide_glasses(0.005).label == NO_GLASSES


def test_borderline_probability_is_unsure():
    verdict = decide_glasses(
        C.GLASSES_PROB_THRESHOLD - C.GLASSES_PROB_UNSURE_MARGIN / 2
    )
    assert verdict.label == "unsure"


def test_no_model_reports_unknown_not_a_guess():
    assert decide_glasses(None).label == UNKNOWN


def test_classifier_without_model_is_unknown_for_a_real_crop():
    classifier = GlassesClassifier()
    crop = np.zeros((40, 80, 3), dtype=np.uint8)
    assert classifier.classify(crop).label == UNKNOWN


def test_tiny_crop_cannot_be_judged():
    class FakePrompts:
        available = True

    classifier = GlassesClassifier(encoder=object(), prompts=FakePrompts())
    tiny = np.zeros((10, 20, 3), dtype=np.uint8)
    assert classifier.probability(tiny) is None
    # No usable view -> unknown, never a guess.
    verdict = classifier.classify(tiny, tiny)
    assert verdict.label == UNKNOWN
    assert "small" in verdict.detail


def test_fused_probability_takes_the_confident_view():
    class FakeEncoder:
        def embed_image(self, crop):
            return np.array([1.0, 0.0], dtype=np.float32)

    class FakePrompts:
        available = True

        def label_scores(self, embedding):
            return {"glasses": 0.30, "no glasses": 0.30}

    classifier = GlassesClassifier(
        encoder=FakeEncoder(), prompts=FakePrompts(), min_face_px=1, min_detail_px=1
    )
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    verdict = classifier.classify(crop, crop)
    assert verdict.score is not None
    assert abs(verdict.score - 0.5) < 1e-6


def test_a_weak_view_cannot_veto_a_confident_one():
    """Real webcam case: face 0.42, blurred eye band 0.00 -> must say glasses."""

    class FakeEncoder:
        def embed_image(self, crop):
            if crop.mean() > 127:
                return np.array([1.0, 0.0], dtype=np.float32)  # glasses-like
            return np.array([0.0, 1.0], dtype=np.float32)  # bare-eye-like

    class FakePrompts:
        available = True

        def label_scores(self, embedding):
            return {
                "glasses": float(embedding[0]) * 0.4,
                "no glasses": float(embedding[1]) * 0.4,
            }

    classifier = GlassesClassifier(
        encoder=FakeEncoder(), prompts=FakePrompts(), min_face_px=1, min_detail_px=1
    )
    strong = np.full((64, 64, 3), 255, dtype=np.uint8)
    weak = np.zeros((64, 64, 3), dtype=np.uint8)

    assert classifier.classify(weak, strong).label == GLASSES
    assert classifier.classify(strong, weak).label == GLASSES