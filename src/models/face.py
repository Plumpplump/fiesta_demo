# Author: David Li, SISHK
"""Face detection (OpenCV YuNet) used to locate the eye area.

YOLO gives us a person box, not eyes. YuNet returns the right/left eye points
alongside the face box, which is exactly what the glasses step needs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import Box
from .person_detector import ModelUnavailable


class FaceDetector:
    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 200,
    ):
        import cv2

        path = Path(model_path)
        if not path.is_file():
            raise ModelUnavailable(f"face model not found: {path}")
        self._cv2 = cv2
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), score_threshold, nms_threshold, top_k
        )

    def detect(self, frame_bgr: np.ndarray) -> list[np.ndarray]:
        """Return YuNet rows in frame coordinates (may be empty)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        height, width = frame_bgr.shape[:2]
        self._detector.setInputSize((width, height))
        _retval, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return []
        return [faces[index] for index in range(len(faces))]


def face_box(face: np.ndarray) -> Box:
    """YuNet row -> corner box ``(x1, y1, x2, y2)``.

    YuNet returns ``[x, y, w, h, ...]``; treating ``w``/``h`` as ``x2``/``y2``
    crops a sliver of the forehead and anchors the shirt band above the face.
    """
    x, y, w, h = (float(v) for v in face[:4])
    return (x, y, x + w, y + h)


def pick_face(
    faces: list[np.ndarray], person_box: Box, margin: float = 0.10
) -> np.ndarray | None:
    """Largest face whose centre lies inside the (slightly expanded) person box.

    The sibling project's rule, and it is more robust than "nearest the head
    area" when two people stand close together: a big neighbouring face cannot
    be borrowed by the wrong person.
    """
    x1, y1, x2, y2 = person_box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = width * margin
    pad_y = height * margin

    best: np.ndarray | None = None
    best_area = 0.0
    for face in faces:
        fx, fy, fw, fh = (float(v) for v in face[:4])
        centre_x, centre_y = fx + fw / 2.0, fy + fh / 2.0
        if not (
            x1 - pad_x <= centre_x <= x2 + pad_x
            and y1 - pad_y <= centre_y <= y2 + pad_y
        ):
            continue
        area = fw * fh
        if area > best_area:
            best, best_area = face, area
    return best