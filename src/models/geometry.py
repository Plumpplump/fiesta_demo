# Author: David Li, SISHK
"""Pure geometry helpers: letterboxing, box maths, NMS, crop derivation.

Deliberately Flet-free and side-effect free so every rule here can be unit
tested without a camera or a running app.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def xywh_to_xyxy(box: Box) -> Box:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def clip_box(box: Box, width: float, height: float) -> Box:
    """Clamp a box into the image and guarantee x1<x2 / y1<y2."""
    x1, y1, x2, y2 = box
    x1 = clamp(min(x1, x2), 0.0, width)
    y1 = clamp(min(y1, y2), 0.0, height)
    x2 = clamp(max(x1, x2), 0.0, width)
    y2 = clamp(max(y1, y2), 0.0, height)
    return (x1, y1, x2, y2)


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area_ratio(box: Box, width: float, height: float) -> float:
    """Fraction of the frame covered by the box (plan.txt's 15% rule)."""
    frame_area = float(width) * float(height)
    if frame_area <= 0:
        return 0.0
    return box_area(box) / frame_area


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list[Box], scores: list[float], threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first."""
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [i for i in order if iou(boxes[best], boxes[i]) <= threshold]
    return keep


def letterbox(
    image: np.ndarray,
    size: int = 640,
    pad_value: int = 114,
) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping aspect ratio and pad to a square.

    Returns ``(padded, scale, pad_x, pad_y)`` where a letterboxed coordinate
    maps back to the original image with ``(v - pad) / scale``.
    """
    import cv2

    height, width = image.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("empty image")
    scale = min(size / width, size / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0
    padded = np.full((size, size, 3), pad_value, dtype=image.dtype)
    top, left = int(round(pad_y)), int(round(pad_x))
    padded[top : top + new_h, left : left + new_w] = resized
    return padded, scale, float(left), float(top)


def unletterbox_box(box: Box, scale: float, pad_x: float, pad_y: float) -> Box:
    x1, y1, x2, y2 = box
    return (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )


def crop(image: np.ndarray, box: Box) -> np.ndarray:
    """Crop with integer rounding, clipped to the image bounds."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box(box, width, height)
    xi1, yi1 = int(round(x1)), int(round(y1))
    xi2, yi2 = int(round(x2)), int(round(y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return image[0:0, 0:0]
    return image[yi1:yi2, xi1:xi2]


def sub_box(
    box: Box,
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> Box:
    """Slice a sub-rectangle out of a box using relative coordinates."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 + left * w, y1 + top * h, x1 + right * w, y1 + bottom * h)


def eye_box_from_face(
    face: np.ndarray,
    padding: float = 0.45,
    height_ratio: float = 0.50,
) -> Box | None:
    """Eye strip built from the YuNet landmarks, always inside the face box.

    YuNet returns ``[x, y, w, h, re_x, re_y, le_x, le_y, nose..., mouth...,
    score]`` in absolute pixels; the right/left eye points are items 4..7.

    ``padding`` is the half-width added on each side and ``height_ratio`` the
    half-height, both in units of the inter-eye distance. Sizing everything by
    the eye distance keeps the strip on the eyes. The previous version added
    0.9 eye distances per side and let the *tilt* between the eyes
    (``abs(eye_dy)``) inflate the height, so the box wandered onto ears/hair
    and off the frame. The box is intersected with the face box, so it can
    never leave the face.
    """
    if face is None or len(face) < 8:
        return None
    fx, fy, fw, fh = (float(v) for v in face[:4])
    right_eye = (float(face[4]), float(face[5]))
    left_eye = (float(face[6]), float(face[7]))
    eye_dx = abs(left_eye[0] - right_eye[0])
    if eye_dx <= 1.0:
        return None
    eye_y = (right_eye[1] + left_eye[1]) / 2.0
    half_w = padding * eye_dx
    half_h = height_ratio * eye_dx
    x1 = max(min(right_eye[0], left_eye[0]) - half_w, fx)
    y1 = max(eye_y - half_h, fy)
    x2 = min(max(right_eye[0], left_eye[0]) + half_w, fx + fw)
    y2 = min(eye_y + half_h, fy + fh)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

def face_anchored_torso_box(
    person_box: Box,
    face_box: Box,
    face_heights: float = 2.0,
    min_height: float = 32.0,
) -> Box | None:
    """Shirt band measured from the chin instead of from the person box.

    Guessing the shirt from the person box fails whenever the pose changes
    (close-up, arms out, leaning). A YuNet face box gives the head, so the
    band starts just below the chin and reaches ``face_heights`` face heights
    down: collar, badge and chest in one crop. Returns ``None`` when the
    resulting band is too small to classify honestly.
    """
    px1, _py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = face_box
    face_height = max(1.0, fy2 - fy1)
    top = fy2 + 0.05 * face_height
    bottom = min(py2, fy2 + face_heights * face_height)
    if bottom - top < min_height or px2 <= px1:
        return None
    margin = 0.08 * (px2 - px1)
    return (px1 - margin, top, px2 + margin, bottom)


def face_crop_box(face_box: Box, margin: float = 0.0) -> Box:
    """Face box grown by a relative margin; the glasses classifier's input."""
    x1, y1, x2, y2 = face_box
    pad = margin * max(x2 - x1, y2 - y1)
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad)
