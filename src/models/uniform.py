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
from .clip import ClipEncoder, PromptBank, load_bgr, normalise
from .types import IN_UNIFORM, NOT_IN_UNIFORM, UNSURE


@dataclass(frozen=True)
class Verdict:
    label: str
    score: float | None = None
    detail: str = ""


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
    best_similarity: float | None,
    text_margin: float | None,
    white_ratio: float,
    red_ratio: float,
    similarity_threshold: float = C.UNIFORM_SIM_THRESHOLD,
    unsure_margin: float = C.UNIFORM_SIM_UNSURE_MARGIN,
    white_min: float = C.UNIFORM_WHITE_RATIO_MIN,
    red_min: float = C.UNIFORM_RED_RATIO_MIN,
) -> Verdict:
    """Pure decision rule, so the thresholds can be tested without a model."""
    has_white = white_ratio >= white_min
    has_trim = red_ratio >= red_min

    if best_similarity is None and text_margin is None:
        if has_white:
            return Verdict(IN_UNIFORM, None, "colour only: white shirt")
        return Verdict(NOT_IN_UNIFORM, None, "colour only: no white shirt")

    score = best_similarity if best_similarity is not None else text_margin
    assert score is not None

    if score >= similarity_threshold:
        if not has_white and score < similarity_threshold + 0.15:
            return Verdict(
                UNSURE,
                score,
                f"model says uniform ({score:.2f}) but the torso is not white",
            )
        trim = " with red trim" if has_trim else ""
        return Verdict(IN_UNIFORM, score, f"matched reference{trim} ({score:.2f})")

    if score >= similarity_threshold - unsure_margin:
        return Verdict(UNSURE, score, f"borderline match ({score:.2f})")

    return Verdict(NOT_IN_UNIFORM, score, f"no reference match ({score:.2f})")


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

        best_similarity: float | None = None
        best_name = ""
        if self.ready and embedding is not None:
            for name, reference_embedding in self._embeddings:
                similarity = float(np.dot(normalise(embedding), reference_embedding))
                if best_similarity is None or similarity > best_similarity:
                    best_similarity, best_name = similarity, name

        text_margin: float | None = None
        if self.prompts is not None and self.prompts.available and embedding is not None:
            scores = self.prompts.label_scores(embedding)
            if IN_UNIFORM in scores and NOT_IN_UNIFORM in scores:
                text_margin = scores[IN_UNIFORM] - scores[NOT_IN_UNIFORM]

        effective_threshold = (
            C.UNIFORM_SIM_THRESHOLD if threshold is None else threshold
        )
        verdict = decide_uniform(
            best_similarity,
            text_margin,
            white_ratio,
            red_ratio,
            similarity_threshold=effective_threshold,
        )
        if best_name:
            return Verdict(verdict.label, verdict.score, f"{best_name}: {verdict.detail}")
        return verdict


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