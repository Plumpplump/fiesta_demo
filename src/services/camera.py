"""OpenCV camera capture for every platform.

``flet_camera`` refuses to run on desktop (its ``before_update`` raises
``FletUnsupportedPlatformException`` outside web/Android/iOS), so the demo uses
OpenCV everywhere. Under ``flet run --web`` the Python process still runs on
the same MacBook, so it reads the laptop webcam and pushes frames to the
browser. This does NOT work if the browser is on another machine.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque

import numpy as np

from models import constants as C


class OpenCvCamera:
    def __init__(
        self,
        index: int = C.CAMERA_INDEX,
        width: int = C.CAMERA_WIDTH,
        height: int = C.CAMERA_HEIGHT,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.error: str | None = None

        self._capture = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._stamps: deque[float] = deque(maxlen=30)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> bool:
        if self._thread is not None and self._thread.is_alive():
            return True
        import cv2

        backend = 0
        if sys.platform == "win32":
            # Media Foundation is much faster than DirectShow for laptop webcams
            # (measured 21 fps vs 2.4 fps on the dev machine).
            backend = cv2.CAP_MSMF
        elif sys.platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION

        capture = cv2.VideoCapture(self.index, backend)
        if not capture.isOpened() and backend != 0:
            capture.release()
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            capture.release()
            self.error = (
                f"camera {self.index} could not be opened. On macOS grant camera "
                "access to your terminal in System Settings > Privacy & Security."
            )
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._capture = capture
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        with self._lock:
            self._frame = None
            self._stamps.clear()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- frames ------------------------------------------------------------
    def read_latest(self) -> np.ndarray | None:
        """Return the newest BGR frame. Never blocks on the camera."""
        with self._lock:
            return self._frame

    @property
    def frame_id(self) -> int:
        with self._lock:
            return self._frame_id

    @property
    def fps(self) -> float:
        with self._lock:
            if len(self._stamps) < 2:
                return 0.0
            span = self._stamps[-1] - self._stamps[0]
            return (len(self._stamps) - 1) / span if span > 0 else 0.0

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            capture = self._capture
            if capture is None:
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                failures += 1
                if failures > 100:
                    self.error = "camera stopped returning frames"
                    break
                time.sleep(0.02)
                continue
            failures = 0
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._stamps.append(time.monotonic())