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


def pick_face(faces: list[np.ndarray], person_box: Box) -> np.ndarray | None:
    """Choose the face inside a person box, preferring the head area."""
    x1, y1, x2, y2 = person_box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    target_x = x1 + width / 2.0
    target_y = y1 + 0.15 * height

    best: np.ndarray | None = None
    best_distance: float | None = None
    for face in faces:
        fx, fy, fw, fh = (float(v) for v in face[:4])
        centre_x, centre_y = fx + fw / 2.0, fy + fh / 2.0
        if not (x1 <= centre_x <= x2 and y1 <= centre_y <= y2):
            continue
        distance = ((centre_x - target_x) / width) ** 2 + ((centre_y - target_y) / height) ** 2
        if best_distance is None or distance < best_distance:
            best, best_distance = face, distance
    return best