# Author: David Li, SISHK
"""Uniform decision rules, with and without a CLIP model present."""

import numpy as np

from models import constants as C
from models.types import IN_UNIFORM, NOT_IN_UNIFORM, UNSURE
from models.uniform import color_profile, decide_uniform, load_references, reference_dir


def _white_shirt(size=64):
    image = np.full((size, size, 3), 245, dtype=np.uint8)  # BGR white
    image[10:20, 10:30] = (0, 0, 190)  # red trim patch
    return image


def _dark_top(size=64):
    return np.full((size, size, 3), (40, 40, 40), dtype=np.uint8)


def test_color_profile_sees_white_and_red():
    white_ratio, red_ratio = color_profile(_white_shirt())
    assert white_ratio > 0.8
    assert red_ratio > 0.01


def test_color_profile_rejects_dark_clothing():
    white_ratio, _red = color_profile(_dark_top())
    assert white_ratio < 0.1


def test_colour_only_fallback_says_uniform_for_a_white_shirt():
    verdict = decide_uniform(None, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == IN_UNIFORM
    assert "colour only" in verdict.detail


def test_colour_only_fallback_rejects_dark_clothing():
    verdict = decide_uniform(None, white_ratio=0.0, red_ratio=0.0)
    assert verdict.label == NOT_IN_UNIFORM


def test_high_probability_is_a_match():
    verdict = decide_uniform(0.90, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == IN_UNIFORM
    assert verdict.score == 0.90


def test_borderline_probability_is_unsure():
    verdict = decide_uniform(
        C.UNIFORM_PROB_THRESHOLD - C.UNIFORM_PROB_UNSURE_MARGIN / 2, 0.8, 0.02
    )
    assert verdict.label == UNSURE


def test_low_probability_is_not_uniform():
    verdict = decide_uniform(0.05, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == NOT_IN_UNIFORM


def test_colour_is_advisory_once_a_model_decides():
    # The school PE top is not white; a confident model verdict must win.
    verdict = decide_uniform(C.UNIFORM_PROB_THRESHOLD + 0.1, 0.0, 0.0)
    assert verdict.label == IN_UNIFORM
    assert "white 0.00" in verdict.detail


def test_slider_threshold_overrides_the_default():
    probability = 0.70
    assert decide_uniform(probability, 0.8, 0.02).label == IN_UNIFORM
    stricter = decide_uniform(
        probability, 0.8, 0.02, similarity_threshold=0.75
    )
    assert stricter.label == UNSURE  # inside the unsure band below 0.75


def test_reference_folder_is_found_with_a_space_in_the_name(tmp_path):
    folder = tmp_path / "Reference Photos"
    folder.mkdir()
    (folder / "PE_Top.png").write_bytes(b"not really a png")
    assert reference_dir(tmp_path) == folder
    assert len(load_references(tmp_path)) == 1

def test_prompt_fusion_separates_the_two_people_from_the_live_log():
    """Real cosines from a webcam run: one casual white top, one PE uniform."""
    from models.uniform import fuse_probabilities, text_probability

    # Person 1: white casual top (prompts prefer "casual" on every sample).
    casual = [
        (0.30, 0.34, 0.56),
        (0.29, 0.36, 0.56),
        (0.28, 0.33, 0.54),
        (0.32, 0.37, 0.52),
        (0.27, 0.32, 0.55),
    ]
    for uniform_cos, casual_cos, reference_cos in casual:
        from models.clip import reference_probability

        p = fuse_probabilities(
            text_probability(uniform_cos, casual_cos),
            reference_probability(reference_cos),
        )
        assert p is not None and p < C.UNIFORM_PROB_THRESHOLD, (uniform_cos, p)

    # Person 2: the real PE uniform.
    for uniform_cos, casual_cos, reference_cos in [
        (0.38, 0.30, 0.78),
        (0.36, 0.29, 0.79),
        (0.36, 0.32, 0.62),
    ]:
        from models.clip import reference_probability

        p = fuse_probabilities(
            text_probability(uniform_cos, casual_cos),
            reference_probability(reference_cos),
        )
        assert p is not None and p > 0.9, (uniform_cos, p)


def test_reference_alone_cannot_rescue_a_casual_crop():
    from models.clip import reference_probability
    from models.uniform import fuse_probabilities, text_probability

    # A casual top scored 0.56 against the product photos (p=0.88) while the
    # prompts said "casual" (text ~0.01). The weighted mean must stay below.
    p = fuse_probabilities(text_probability(0.30, 0.34), reference_probability(0.56))
    assert p is not None and p < C.UNIFORM_PROB_THRESHOLD
