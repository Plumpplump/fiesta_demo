"""CLIP-style image encoder plus pre-computed text prompt vectors.

Runtime only needs onnxruntime: the tokenizer and the text tower live in
``scripts/export_models.py``, which saves ``*_prompts.npy`` + ``*_prompts.json``
once. That keeps the app free of torch/open_clip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .person_detector import ModelUnavailable, create_session

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


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


def preprocess(bgr: np.ndarray, image_size: int) -> np.ndarray:
    """BGR uint8 crop -> normalised CLIP tensor (1, 3, S, S)."""
    import cv2

    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    array = rgb.astype(np.float32) / 255.0
    array = (array - CLIP_MEAN) / CLIP_STD
    return np.transpose(array, (2, 0, 1))[None]


class ClipEncoder:
    def __init__(self, model_path: str | Path, image_size: int | None = None):
        self.session = create_session(model_path)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        shape = list(input_meta.shape)
        detected = shape[-1] if shape and isinstance(shape[-1], int) else 224
        self.image_size = int(image_size or detected)

    def embed_image(self, bgr: np.ndarray) -> np.ndarray:
        tensor = preprocess(bgr, self.image_size)
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
        """Mean cosine similarity per label group."""
        if not self.available:
            return {}
        sims = self.vectors @ normalise(embedding)
        scores: dict[str, list[float]] = {}
        for label, sim in zip(self.labels, sims):
            scores.setdefault(label, []).append(float(sim))
        return {label: sum(values) / len(values) for label, values in scores.items()}