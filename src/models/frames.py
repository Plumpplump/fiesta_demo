"""Frame helper functions: preview encoding and drawing overlays."""

from __future__ import annotations

import numpy as np

from . import constants as C
from .types import Track


def to_preview_jpeg(
    frame_bgr: np.ndarray,
    max_width: int = C.PREVIEW_MAX_WIDTH,
    quality: int = C.PREVIEW_JPEG_QUALITY,
) -> bytes:
    """Downscale + JPEG-encode a frame for the browser/desktop preview."""
    import cv2

    height, width = frame_bgr.shape[:2]
    if width > max_width:
        scale = max_width / float(width)
        frame_bgr = cv2.resize(
            frame_bgr,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("could not encode the camera frame as JPEG")
    return buffer.tobytes()


def draw_overlay(frame_bgr: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Person / torso / eye boxes plus a Person N label, drawn in place."""
    import cv2

    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
        tx1, ty1, tx2, ty2 = (int(v) for v in track.torso_box)
        cv2.rectangle(frame_bgr, (tx1, ty1), (tx2, ty2), (60, 180, 75), 2)
        if track.eye_box is not None:
            ex1, ey1, ex2, ey2 = (int(v) for v in track.eye_box)
            cv2.rectangle(frame_bgr, (ex1, ey1), (ex2, ey2), (255, 120, 0), 2)
        label = f"{C.PERSON_LABEL_PREFIX}{track.label_index}"
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 4, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
    return frame_bgr