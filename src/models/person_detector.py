"""YOLO11n person detector running on onnxruntime.

The .onnx file is produced once by ``scripts/export_models.py``; nothing here
imports torch or ultralytics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from . import constants as C
from .geometry import clip_box, letterbox, nms, unletterbox_box
from .types import Detection


class ModelUnavailable(RuntimeError):
    """Raised when an ONNX model file is missing or cannot be loaded."""


def providers() -> list[str]:
    import onnxruntime as ort

    if os.environ.get("FIESTA_FORCE_CPU") == "1":
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    preferred = (
        ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        if sys.platform == "darwin"
        else ["CPUExecutionProvider"]
    )
    chosen = [p for p in preferred if p in available]
    return chosen or ["CPUExecutionProvider"]


def create_session(model_path: str | Path):
    import onnxruntime as ort

    path = Path(model_path)
    if not path.is_file():
        raise ModelUnavailable(f"model file not found: {path}")
    try:
        return ort.InferenceSession(str(path), providers=providers())
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise ModelUnavailable(f"could not load {path.name}: {exc}") from exc


def decode_yolo(
    output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    original_size: tuple[int, int],
    confidence: float = C.DETECT_CONFIDENCE,
    iou_threshold: float = C.DETECT_IOU,
    max_detections: int = C.MAX_DETECTIONS,
    class_id: int = C.PERSON_CLASS_ID,
) -> list[Detection]:
    """Turn the raw YOLO output into person boxes in original-frame pixels.

    Ultralytics exports give ``(1, 4 + nc, anchors)`` with ``x, y, w, h`` centre
    form; some tools transpose that, so both layouts are accepted.
    """
    pred = np.asarray(output, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim == 1:
        pred = pred[None, :]
    if pred.ndim != 2:
        raise ValueError(f"unexpected YOLO output shape: {np.shape(output)}")

    # Find the attribute axis: it holds 4 box values + one score per class, so a
    # sane length is 5..200 while the other axis is the (usually much larger)
    # anchor count. Ambiguous shapes fall back to ultralytics' (attrs, anchors).
    rows, cols = pred.shape
    min_attrs = 5
    max_attrs = 200
    if min_attrs <= rows <= max_attrs < cols:
        pred = pred.T
    elif min_attrs <= cols <= max_attrs:
        pass
    elif min_attrs <= rows <= max_attrs:
        pred = pred.T

    if pred.shape[1] <= 4 + class_id:
        raise ValueError(f"YOLO output has no class column {class_id}: {pred.shape}")

    scores = pred[:, 4 + class_id]
    mask = scores >= confidence
    if not np.any(mask):
        return []

    centres = pred[mask, :4]
    scores = scores[mask]

    boxes: list[tuple[float, float, float, float]] = []
    for cx, cy, w, h in centres:
        corner_box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
        boxes.append(unletterbox_box(corner_box, scale, pad_x, pad_y))

    width, height = original_size
    boxes = [clip_box(b, width, height) for b in boxes]
    boxes = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]

    keep = nms(boxes, [float(s) for s in scores], iou_threshold)
    return [Detection(box=boxes[i], score=float(scores[i])) for i in keep][:max_detections]


class YoloOnnxDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: int = C.DETECT_INPUT_SIZE,
        confidence: float = C.DETECT_CONFIDENCE,
        iou_threshold: float = C.DETECT_IOU,
        max_detections: int = C.MAX_DETECTIONS,
    ):
        self.session = create_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = input_size
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        padded, scale, pad_x, pad_y = letterbox(frame_bgr, self.input_size)
        blob = padded[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        blob = np.transpose(blob, (2, 0, 1))[None]
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        height, width = frame_bgr.shape[:2]
        return decode_yolo(
            outputs[0],
            scale,
            pad_x,
            pad_y,
            (width, height),
            self.confidence,
            self.iou_threshold,
            self.max_detections,
        )