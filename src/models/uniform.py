# Author: David Li, SISHK
"""School-uniform verdict from a torso crop.

Two independent signals:
  * CLIP image similarity against the reference tops in ``assets/reference*``
  * a colour sanity check (the uniform is a white shirt with red trim)

The thresholds are deliberately loose (see plan.txt): the demo prefers a
"looks right" answer over a strict one, and the UI shows the score so a wrong
call is visible rather than mysterious.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import constants as C
from .clip import (
    ClipEncoder,
    PromptBank,
    level_probability,
    load_bgr,
    normalise,
    reference_probability,
    softmax_pair,
)
from .types import IN_UNIFORM, NOT_IN_UNIFORM, UNSURE


@dataclass(frozen=True)
class Verdict:
    label: str
    score: float | None = None
    detail: str = ""
    # Colour evidence from the same crop, so a smoothed verdict can be
    # re-decided later without re-reading the frame.
    white_ratio: float | None = None
    red_ratio: float | None = None


@dataclass(frozen=True)
class Reference:
    name: str
    path: Path


def color_profile(crop_bgr: np.ndarray) -> tuple[float, float]:
    """Return ``(white_ratio, red_ratio)`` for a torso crop."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    white = (val >= 140) & (sat <= 75)
    red = ((hue <= 12) | (hue >= 168)) & (sat >= 90) & (val >= 55)
    return float(white.mean()), float(red.mean())


def decide_uniform(
    probability: float | None,
    white_ratio: float,
    red_ratio: float,
    similarity_threshold: float = C.UNIFORM_PROB_THRESHOLD,
    unsure_margin: float = C.UNIFORM_PROB_UNSURE_MARGIN,
    white_min: float = C.UNIFORM_WHITE_RATIO_MIN,
    red_min: float = C.UNIFORM_RED_RATIO_MIN,
) -> Verdict:
    """Pure decision rule, so the thresholds can be tested without a model.

    ``probability`` is P(uniform) in 0..1 from the prompt level fused with the
    reference-photo cosine. ``None`` means "no model": fall back to colour.

    Colour is *advisory* once a model is present. Blocking a confident verdict
    because the torso is not white made the red/PE uniform read "unsure"; only
    people are classified, so the colour guard has little left to protect.
    """
    has_white = white_ratio >= white_min
    has_trim = red_ratio >= red_min
    colour = f"white {white_ratio:.2f}, red {red_ratio:.2f}"

    if probability is None:
        if has_white:
            return Verdict(IN_UNIFORM, None, f"colour only: white shirt ({colour})")
        return Verdict(NOT_IN_UNIFORM, None, f"colour only: no white shirt ({colour})")

    if probability >= similarity_threshold:
        trim = " with red trim" if has_trim else ""
        return Verdict(
            IN_UNIFORM,
            probability,
            f"matched a uniform prompt{trim} ({probability:.2f}; {colour})",
        )

    if probability >= similarity_threshold - unsure_margin:
        return Verdict(UNSURE, probability, f"borderline match ({probability:.2f}; {colour})")

    return Verdict(NOT_IN_UNIFORM, probability, f"no uniform match ({probability:.2f}; {colour})")


def text_probability(
    uniform_cos: float, casual_cos: float | None = None
) -> float:
    """P(uniform) from the prompts: level x gap.

    Each alone failed on live data: the level alone let a white non-uniform top
    through (u=0.30, same as a real uniform), the gap alone let an ordinary
    portrait through (u=0.22 vs c=0.21 -> gap probability 0.73). Requiring both
    - "looks like a uniform" AND "looks more uniform than casual" - separates
    the two people in the webcam log cleanly.
    """
    level = level_probability(
        uniform_cos, C.UNIFORM_TEXT_COS_MIDPOINT, C.UNIFORM_TEXT_COS_SLOPE
    )
    gap = softmax_pair(uniform_cos, casual_cos) if casual_cos is not None else 1.0
    return level * gap


def fuse_probabilities(
    text: float | None, reference: float | None
) -> float | None:
    """Weighted mean, text-heavy; the reference only confirms.

    Reference photos are product cut-outs, so a casual white top reached 0.56
    cosine (p=0.88). A max() let that outvote prompts that said "casual"; the
    weighted mean keeps the reference as confirmation instead of an override.
    """
    if text is None:
        return reference
    if reference is None:
        return text
    return C.UNIFORM_TEXT_WEIGHT * text + (1.0 - C.UNIFORM_TEXT_WEIGHT) * reference


class UniformClassifier:
    def __init__(
        self,
        references: list[Reference],
        encoder: ClipEncoder | None = None,
        prompts: PromptBank | None = None,
    ):
        self.references = references
        self.encoder = encoder
        self.prompts = prompts
        self._embeddings: list[tuple[str, np.ndarray]] = []
        if encoder is not None:
            self._embeddings = self._embed_references()

    def _embed_references(self) -> list[tuple[str, np.ndarray]]:
        embedded: list[tuple[str, np.ndarray]] = []
        for reference in self.references:
            try:
                image = load_bgr(reference.path)
                if image is None:
                    continue
                embedded.append((reference.name, self.encoder.embed_image(image)))
            except Exception:  # unreadable asset must not kill the demo
                continue
        return embedded

    @property
    def ready(self) -> bool:
        return self.encoder is not None and bool(self._embeddings)

    def classify(self, torso_bgr: np.ndarray, threshold: float | None = None) -> Verdict:
        if torso_bgr is None or torso_bgr.size == 0 or min(torso_bgr.shape[:2]) < 32:
            return Verdict(UNSURE, None, "torso crop too small")

        white_ratio, red_ratio = color_profile(torso_bgr)

        embedding: np.ndarray | None = None
        if self.encoder is not None:
            embedding = self.encoder.embed_image(torso_bgr)

        # Two signals: the prompt level x gap (dominant) and the reference-photo
        # cosine (confirmation). A weighted mean rather than max: a casual white
        # top reached 0.56 against the product cut-outs, and max let that
        # reference outvote prompts that clearly said "casual".
        probability: float | None = None
        signals: list[str] = []

        if embedding is not None and self.prompts is not None and self.prompts.available:
            uniform_match = self.prompts.best_match(embedding, IN_UNIFORM)
            casual_match = self.prompts.best_match(embedding, NOT_IN_UNIFORM)
            if uniform_match is not None:
                # Level x gap: it must look like a uniform AND unlike casual
                # clothes. Each alone failed on live data (see text_probability).
                text = text_probability(
                    uniform_match[0],
                    casual_match[0] if casual_match is not None else None,
                )
                probability = text
                casual_note = (
                    f" vs '{casual_match[1]}' {casual_match[0]:.2f}"
                    if casual_match is not None
                    else ""
                )
                signals.append(
                    f"text {text:.2f} "
                    f"('{uniform_match[1]}' {uniform_match[0]:.2f}{casual_note})"
                )

        if self.ready and embedding is not None:
            best_similarity: float | None = None
            best_name = ""
            for name, reference_embedding in self._embeddings:
                similarity = float(np.dot(normalise(embedding), reference_embedding))
                if best_similarity is None or similarity > best_similarity:
                    best_similarity, best_name = similarity, name
            if best_similarity is not None:
                reference_p = reference_probability(best_similarity)
                probability = fuse_probabilities(probability, reference_p)
                signals.append(
                    f"reference {best_name} {best_similarity:.2f} -> {reference_p:.2f}"
                )

        effective_threshold = (
            C.UNIFORM_PROB_THRESHOLD if threshold is None else threshold
        )
        verdict = decide_uniform(
            probability,
            white_ratio,
            red_ratio,
            similarity_threshold=effective_threshold,
        )
        detail = f"{'; '.join(signals)} | {verdict.detail}" if signals else verdict.detail
        return Verdict(
            verdict.label,
            verdict.score,
            detail,
            white_ratio=white_ratio,
            red_ratio=red_ratio,
        )


def reference_dir(assets_root: Path) -> Path | None:
    """Reference tops live in ``assets/reference*`` (the folder name has a space)."""
    for candidate in sorted(assets_root.iterdir()):
        if not candidate.is_dir():
            continue
        if not candidate.name.lower().startswith("reference"):
            continue
        suffixes = {p.suffix.lower() for p in candidate.rglob("*") if p.is_file()}
        if suffixes & {".png", ".jpg", ".jpeg"}:
            return candidate
    return None


def load_references(assets_root: Path) -> list[Reference]:
    folder = reference_dir(assets_root)
    if folder is None:
        return []
    paths = sorted(
        [*folder.rglob("*.png"), *folder.rglob("*.jpg"), *folder.rglob("*.jpeg")]
    )
    return [Reference(name=p.stem.replace("_", " "), path=p) for p in paths]