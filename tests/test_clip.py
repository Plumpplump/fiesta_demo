# Author: David Li, SISHK
"""Preprocessing must match the exported checkpoint, not the CLIP defaults."""

import json

import numpy as np

from models.clip import (
    CLIP_MEAN,
    CLIP_STD,
    load_encoder_meta,
    preprocess,
    reference_probability,
    softmax_pair,
)


def _white(size=8):
    return np.full((size, size, 3), 255, dtype=np.uint8)


def test_preprocess_applies_the_given_mean_and_std():
    tensor = preprocess(
        _white(),
        4,
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    assert tensor.shape == (1, 3, 4, 4)
    assert tensor.dtype == np.float32
    assert np.allclose(tensor, 1.0)  # MobileCLIP2: just /255


def test_preprocess_defaults_to_the_clip_constants():
    tensor = preprocess(_white(), 4)
    expected = (1.0 - CLIP_MEAN) / CLIP_STD
    assert np.allclose(tensor[0, :, 0, 0], expected)


def test_preprocess_keeps_aspect_ratio_with_a_centre_crop():
    # A 160x40 crop (4:1) resized to 224x224 by stretching would repeat pixels
    # to fill the square; the real transform resizes the short side first.
    wide = np.zeros((40, 160, 3), dtype=np.uint8)
    wide[:, :80] = 255  # left half white, right half black
    tensor = preprocess(wide, 8, mean=np.zeros(3, np.float32), std=np.ones(3, np.float32))
    # Centre crop of the middle band: the white/black seam survives.
    assert tensor.shape == (1, 3, 8, 8)
    assert tensor[0, 0].min() < tensor[0, 0].max()


def test_softmax_pair_turns_cosine_gaps_into_probabilities():
    assert softmax_pair(0.30, 0.25) > 0.9
    assert softmax_pair(0.25, 0.30) < 0.1
    assert abs(softmax_pair(0.30, 0.30) - 0.5) < 1e-9


def test_reference_probability_is_monotonic_around_its_threshold():
    from models import constants as C

    assert reference_probability(C.REFERENCE_COS_THRESHOLD) == 0.5
    assert reference_probability(C.REFERENCE_COS_THRESHOLD - 0.05) < 0.2
    assert reference_probability(C.REFERENCE_COS_THRESHOLD + 0.05) > 0.8


def test_encoder_meta_is_read_from_the_sidecar_json(tmp_path):
    model = tmp_path / "mobileclip2_b_vision.onnx"
    model.write_bytes(b"not really onnx")
    model.with_suffix(".json").write_text(
        json.dumps({"image_size": 224, "mean": [0, 0, 0], "std": [1, 1, 1]}),
        encoding="utf-8",
    )

    meta = load_encoder_meta(model)

    assert meta["image_size"] == 224
    assert meta["std"] == [1, 1, 1]


def test_missing_or_broken_meta_falls_back_to_empty(tmp_path):
    model = tmp_path / "mobileclip2_b_vision.onnx"
    model.write_bytes(b"x")
    assert load_encoder_meta(model) == {}

    model.with_suffix(".json").write_text("{not json", encoding="utf-8")
    assert load_encoder_meta(model) == {}


def test_prompt_bank_scores_the_best_prompt_not_the_average():
    from models.clip import PromptBank

    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32
    )
    bank = PromptBank(
        labels=["in uniform", "in uniform", "not in uniform"],
        texts=["white polo", "school shirt", "casual tee"],
        vectors=vectors,
    )

    scores = bank.label_scores(np.array([0.0, 1.0], dtype=np.float32))

    assert scores["in uniform"] == 1.0  # the second prompt matched
    assert scores["not in uniform"] == 0.0
    best = bank.best_match(np.array([0.0, 1.0], dtype=np.float32), "in uniform")
    assert best is not None
    assert best[0] == 1.0
    assert best[1] == "school shirt"