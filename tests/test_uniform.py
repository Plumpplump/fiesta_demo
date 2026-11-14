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
    verdict = decide_uniform(None, None, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == IN_UNIFORM
    assert "colour only" in verdict.detail


def test_colour_only_fallback_rejects_dark_clothing():
    verdict = decide_uniform(None, None, white_ratio=0.0, red_ratio=0.0)
    assert verdict.label == NOT_IN_UNIFORM


def test_high_similarity_is_a_match():
    verdict = decide_uniform(0.40, None, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == IN_UNIFORM
    assert verdict.score == 0.40


def test_borderline_similarity_is_unsure():
    verdict = decide_uniform(
        C.UNIFORM_SIM_THRESHOLD - C.UNIFORM_SIM_UNSURE_MARGIN / 2, None, 0.8, 0.02
    )
    assert verdict.label == UNSURE


def test_low_similarity_is_not_uniform():
    verdict = decide_uniform(0.01, None, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == NOT_IN_UNIFORM


def test_model_match_without_white_shirt_is_unsure():
    verdict = decide_uniform(C.UNIFORM_SIM_THRESHOLD + 0.01, None, 0.0, 0.0)
    assert verdict.label == UNSURE


def test_text_margin_is_used_when_no_image_reference():
    verdict = decide_uniform(None, 0.35, white_ratio=0.8, red_ratio=0.02)
    assert verdict.label == IN_UNIFORM


def test_reference_folder_is_found_with_a_space_in_the_name(tmp_path):
    folder = tmp_path / "Reference Photos"
    folder.mkdir()
    (folder / "PE_Top.png").write_bytes(b"not really a png")
    assert reference_dir(tmp_path) == folder
    assert len(load_references(tmp_path)) == 1