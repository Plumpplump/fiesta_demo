# Author: David Li, SISHK
"""CLIP-style image encoder plus pre-computed text prompt vectors.

Runtime only needs onnxruntime: the tokenizer and the text tower live in
``scripts/export_models.py``, which saves ``*_prompts.npy`` + ``*_prompts.json``
once. That keeps the app free of torch/open_clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import constants as C
from .person_detector import ModelUnavailable, create_session

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def softmax_pair(first: float, second: float) -> float:
    """P(first) for two cosine scores pushed through a temperature-scaled softmax.

    Cosines differ in the third decimal, so a plain softmax says 50.5% for
    almost everything; multiplying by ``TEXT_SOFTMAX_TEMPERATURE`` (from the
    measured sibling project) turns a 0.01 gap into a usable probability.
    """
    peak = max(first, second)
    a = float(np.exp((first - peak) * C.TEXT_SOFTMAX_TEMPERATURE))
    b = float(np.exp((second - peak) * C.TEXT_SOFTMAX_TEMPERATURE))
    return a / (a + b)


def level_probability(
    cosine: float,
    midpoint: float,
    slope: float,
) -> float:
    """Map an absolute prompt cosine to P(match) around a measured midpoint.

    The *level* of the best uniform prompt cosine separates cleanly (measured
    on this Mac: real uniform crops 0.32-0.37, ordinary clothing 0.14-0.27),
    while the *gap* between uniform and casual prompts is noise. A softmax on
    that gap turns +0.008 into 99.9%, which is exactly how a casual suit got
    labelled "in uniform".
    """
    value = (cosine - midpoint) / slope
    return float(1.0 / (1.0 + np.exp(-value)))


def reference_probability(
    cosine: float,
    threshold: float = C.REFERENCE_COS_THRESHOLD,
    slope: float = C.REFERENCE_COS_SLOPE,
) -> float:
    """Map an image-image cosine to P(uniform) around a calibrated threshold."""
    return level_probability(cosine, threshold, slope)


def load_bgr(path: Path) -> np.ndarray | None:
    """Read an image file as BGR, compositing transparency onto white."""
    import cv2

    path = Path(path)
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        white = np.full_like(rgb, 255.0)
        composited = rgb * alpha + white * (1.0 - alpha)
        return composited.astype(np.uint8)
    return image[:, :, :3]


def normalise(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def preprocess(
    bgr: np.ndarray,
    image_size: int,
    mean: np.ndarray = CLIP_MEAN,
    std: np.ndarray = CLIP_STD,
) -> np.ndarray:
    """BGR uint8 crop -> normalised encoder tensor (1, 3, S, S).

    ``mean``/``std`` come from the export sidecar: MobileCLIP2 ships
    ``0``/``1``, while older CLIP checkpoints use the CLIP constants above.

    The crop is resized by its shortest side and centre-cropped rather than
    squashed to a square. Squashing stretches a wide torso crop into a
    distorted shirt; the checkpoint was trained on aspect-preserving crops
    (measured: crop quality beats model size).
    """
    import cv2

    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    height, width = bgr.shape[:2]
    scale = image_size / float(min(height, width))
    resized = cv2.resize(
        bgr,
        (max(image_size, int(round(width * scale))), max(image_size, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    top = (resized.shape[0] - image_size) // 2
    left = (resized.shape[1] - image_size) // 2
    square = resized[top : top + image_size, left : left + image_size]
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    array = (rgb - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    return np.ascontiguousarray(np.transpose(array, (2, 0, 1)))[None]


def load_encoder_meta(model_path: str | Path) -> dict:
    """Read ``<model>.json`` (mean/std/image_size) written by the export script."""
    path = Path(model_path).with_suffix(".json")
    if not path.is_file():
        return {}
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


class ClipEncoder:
    def __init__(self, model_path: str | Path, image_size: int | None = None):
        path = Path(model_path)
        # Measured on the demo MacBook Air M4: the MobileCLIP2 vision graph runs
        # ~170 ms/frame on CPU but ~400 ms when CoreML has to partition this
        # transformer, so CPU is the fast path for this model (YOLO keeps CoreML).
        self.session = create_session(path, prefer_cpu=True)
        meta = load_encoder_meta(path)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = list(input_meta.shape)
        detected = shape[-1] if shape and isinstance(shape[-1], int) else 224
        meta_size = meta.get("image_size")
        self.image_size = int(image_size or meta_size or detected)
        self.mean = np.asarray(meta.get("mean", CLIP_MEAN), dtype=np.float32)
        self.std = np.asarray(meta.get("std", CLIP_STD), dtype=np.float32)
        if not np.all(self.std > 0):
            self.mean, self.std = CLIP_MEAN, CLIP_STD

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        tensor = preprocess(bgr, self.image_size, self.mean, self.std)
        outputs = self.session.run(None, {self.input_name: tensor})
        return normalise(np.asarray(outputs[0]))


class PromptBank:
    """Text embeddings grouped by label, exported ahead of time."""

    def __init__(self, labels: list[str], texts: list[str], vectors: np.ndarray):
        self.labels = labels
        self.texts = texts
        self.vectors = np.stack([normalise(v) for v in vectors]) if len(vectors) else np.zeros((0, 0))

    @classmethod
    def load(cls, vectors_path: str | Path) -> "PromptBank":
        path = Path(vectors_path)
        if not path.is_file():
            raise ModelUnavailable(f"prompt bank not found: {path}")
        vectors = np.load(path)
        meta_path = path.with_suffix(".json")
        meta: dict = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = list(meta.get("labels") or [f"prompt{i}" for i in range(len(vectors))])
        texts = list(meta.get("texts") or labels)
        return cls(labels, texts, vectors)

    @property
    def available(self) -> bool:
        return len(self.vectors) > 0

    def label_scores(self, embedding: np.ndarray) -> dict[str, float]:
        """Best (max) cosine per label; the strongest prompt wins (measured).

        Averaging every prompt of a label dilutes one good match with
        near-misses, which flattens the uniform-vs-casual gap.
        """
        if not self.available:
            return {}
        scores = self._sims_by_label(embedding)
        return {label: max(values) for label, values in scores.items()}

    def best_match(
        self, embedding: np.ndarray, label: str
    ) -> tuple[float, str] | None:
        """``(cosine, prompt text)`` of the best prompt for ``label``."""
        if not self.available:
            return None
        scores = self._sims_by_label(embedding).get(label)
        if not scores:
            return None
        best_index = int(np.argmax(scores))
        texts = [
            text
            for text, text_label in zip(self.texts, self.labels)
            if text_label == label
        ]
        text = texts[best_index] if best_index < len(texts) else ""
        return float(scores[best_index]), text

    def _sims_by_label(self, embedding: np.ndarray) -> dict[str, list[float]]:
        sims = self.vectors @ normalise(embedding)
        grouped: dict[str, list[float]] = {}
        for label, sim in zip(self.labels, sims):
            grouped.setdefault(label, []).append(float(sim))
        return grouped