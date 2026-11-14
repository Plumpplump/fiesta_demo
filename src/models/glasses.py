"""Glasses verdict from the eye strip.

Uses the same CLIP encoder as the uniform check, against text prompts exported
once by ``scripts/export_models.py`` (``glasses_prompts.npy``). Without that
model the honest answer is "unknown" - a tiny eye strip is not something to
guess about.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .clip import ClipEncoder, PromptBank
from .types import GLASSES, NO_GLASSES, UNKNOWN
from .uniform import Verdict


def decide_glasses(
    margin: float | None,
    threshold: float = C.GLASSES_SIM_THRESHOLD,
    unsure_margin: float = C.GLASSES_SIM_UNSURE_MARGIN,
) -> Verdict:
    """Pure decision rule: positive margin means "glasses look more likely"."""
    if margin is None:
        return Verdict(UNKNOWN, None, "no glasses model")
    if margin >= threshold:
        return Verdict(GLASSES, margin, f"prompt match ({margin:.2f})")
    if margin >= threshold - unsure_margin:
        return Verdict("unsure", margin, f"borderline ({margin:.2f})")
    return Verdict(NO_GLASSES, margin, f"no glasses ({margin:.2f})")


class GlassesClassifier:
    def __init__(
        self,
        encoder: ClipEncoder | None = None,
        prompts: PromptBank | None = None,
        min_crop_px: int = C.MIN_EYE_CROP_PX,
    ):
        self.encoder = encoder
        self.prompts = prompts
        self.min_crop_px = min_crop_px

    @property
    def ready(self) -> bool:
        return (
            self.encoder is not None
            and self.prompts is not None
            and self.prompts.available
        )

    def classify(self, eye_crop_bgr: np.ndarray) -> Verdict:
        if eye_crop_bgr is None or eye_crop_bgr.size == 0:
            return Verdict(UNKNOWN, None, "no face found")
        height, width = eye_crop_bgr.shape[:2]
        if min(height, width) < self.min_crop_px:
            return Verdict(UNKNOWN, None, "eye area too small")
        if not self.ready:
            return Verdict(UNKNOWN, None, "glasses model not exported")

        embedding = self.encoder.embed_image(eye_crop_bgr)
        scores = self.prompts.label_scores(embedding)
        if GLASSES not in scores or NO_GLASSES not in scores:
            return Verdict(UNKNOWN, None, "prompt bank has no glasses labels")
        return decide_glasses(scores[GLASSES] - scores[NO_GLASSES])