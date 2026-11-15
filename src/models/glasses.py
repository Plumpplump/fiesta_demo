# Author: David Li, SISHK
"""Glasses verdict from a face crop plus the eye band.

Uses the same CLIP encoder as the uniform check, against text prompts exported
once by ``scripts/export_models.py`` (``glasses_prompts.npy``). Without that
model the honest answer is "unknown" - glasses are not something to guess about.

Two views are scored and the **higher** one wins:

* the tight eye strip alone separates glasses from bare eyes no better than
  chance (Youden J 0.00) - too little context;
* the face crop carries the context, but blurs at distance;
* taking the max means a confident view is never vetoed by a blurred one. A
  geometric mean looked good on stills and failed on a real webcam log:
  face 0.42 x eye band 0.00 fused to 0.04 "no glasses".

Measured on live webcam logs: a wearer's face crop scores 0.28-0.56 while the
worst false positive is 0.21, so the default threshold sits at 0.25.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .clip import ClipEncoder, PromptBank, softmax_pair
from .types import GLASSES, NO_GLASSES, UNKNOWN
from .uniform import Verdict


def decide_glasses(
    probability: float | None,
    threshold: float = C.GLASSES_PROB_THRESHOLD,
    unsure_margin: float = C.GLASSES_PROB_UNSURE_MARGIN,
) -> Verdict:
    """Pure decision rule on P(glasses).

    The probability is the better of the face and eye-band views. The measured
    scale is not intuitive, so always calibrate with ``scripts/calibrate.py``
    before changing the threshold.
    """
    if probability is None:
        return Verdict(UNKNOWN, None, "no glasses model")
    if probability >= threshold:
        return Verdict(GLASSES, probability, f"prompt match ({probability:.2f})")
    if probability >= threshold - unsure_margin:
        return Verdict("unsure", probability, f"borderline ({probability:.2f})")
    return Verdict(NO_GLASSES, probability, f"no glasses ({probability:.2f})")


class GlassesClassifier:
    def __init__(
        self,
        encoder: ClipEncoder | None = None,
        prompts: PromptBank | None = None,
        min_face_px: int = C.MIN_FACE_CROP_PX,
        min_detail_px: int = C.MIN_DETAIL_CROP_PX,
    ):
        self.encoder = encoder
        self.prompts = prompts
        self.min_face_px = min_face_px
        self.min_detail_px = min_detail_px

    @property
    def ready(self) -> bool:
        return (
            self.encoder is not None
            and self.prompts is not None
            and self.prompts.available
        )

    def probability(
        self, crop_bgr: np.ndarray | None, min_px: int | None = None
    ) -> float | None:
        """P(glasses) for one crop, or ``None`` when it cannot be judged."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        limit = self.min_face_px if min_px is None else min_px
        height, width = crop_bgr.shape[:2]
        if min(height, width) < limit:
            return None
        embedding = self.encoder.embed_image(crop_bgr)
        scores = self.prompts.label_scores(embedding)
        if GLASSES not in scores or NO_GLASSES not in scores:
            return None
        return softmax_pair(scores[GLASSES], scores[NO_GLASSES])

    def classify(
        self,
        face_crop: np.ndarray | None,
        detail_crop: np.ndarray | None = None,
        threshold: float | None = None,
    ) -> Verdict:
        """Verdict from the face view and (optionally) the eye-band view."""
        if not self.ready:
            return Verdict(UNKNOWN, None, "glasses model not exported")

        face_probability = self.probability(face_crop, self.min_face_px)
        probabilities = [face_probability] if face_probability is not None else []
        # A clearly positive face view needs no second CLIP pass: the label is
        # already "glasses" and the extra pass is what made the verdict slow.
        # A *negative* view is not shortcut: the eye band can still catch a
        # frame the wide face crop missed, so it always gets its say.
        if not (face_probability is not None and face_probability >= C.GLASSES_SHORTCUT_HIGH):
            detail_probability = self.probability(detail_crop, self.min_detail_px)
            if detail_probability is not None:
                probabilities.append(detail_probability)
        if not probabilities:
            return Verdict(UNKNOWN, None, "eye area too small")

        # Max, not mean: one confident view is enough evidence. The geometric
        # mean looked tidy in the portrait spot check, but on a real webcam the
        # eye band is often blurred at distance and vetoed correct face calls
        # (face 0.42 x eye 0.00 -> 0.04 "no glasses").
        probability = max(probabilities)
        detail = f"p={probability:.2f} (" + ", ".join(
            f"{value:.2f}" for value in probabilities
        ) + ")"
        verdict = decide_glasses(
            probability, C.GLASSES_PROB_THRESHOLD if threshold is None else threshold
        )
        return Verdict(verdict.label, verdict.score, f"{detail} {verdict.detail}")