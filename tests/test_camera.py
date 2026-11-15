# Author: David Li, SISHK
"""Camera threading: one reader per capture, no stale thread publishing frames.

No real webcam is touched here - a fake capture object is enough to prove the
lifecycle, which is what the flicker fix is about.
"""

import threading
import time

import numpy as np

from services.camera import OpenCvCamera


class FakeCapture:
    def __init__(self, frame=None):
        self.frame = frame if frame is not None else np.zeros((4, 4, 3), dtype=np.uint8)
        self.reads = 0
        self.released = False

    def read(self):
        self.reads += 1
        time.sleep(0.001)
        return True, self.frame

    def release(self):
        self.released = True


def _wait_for(predicate, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _run(camera: OpenCvCamera, capture: FakeCapture, stop: threading.Event):
    thread = threading.Thread(
        target=camera._run, args=(capture, stop), name="camera-test", daemon=True
    )
    thread.start()
    return thread


def test_reader_publishes_frames_then_releases_its_capture():
    camera = OpenCvCamera()
    capture = FakeCapture()
    stop = threading.Event()
    camera._stop = stop

    thread = _run(camera, capture, stop)

    assert _wait_for(lambda: camera.frame_id > 0)
    assert camera.read_latest() is not None
    assert camera.fps > 0

    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert capture.released is True


def test_stale_reader_never_touches_the_next_run():
    camera = OpenCvCamera()
    old_stop = threading.Event()
    old_capture = FakeCapture()
    camera._stop = old_stop

    thread = _run(camera, old_capture, old_stop)
    assert _wait_for(lambda: camera.frame_id > 0)

    # Simulate start() taking over while the old read() is still blocked.
    camera._stop = threading.Event()

    thread.join(timeout=2.0)
    assert not thread.is_alive(), "old reader must exit when a new run starts"
    assert old_capture.released is True

    published = camera.frame_id
    time.sleep(0.05)
    assert camera.frame_id == published, "stale reader published a frame"


def test_failed_reads_report_an_error_instead_of_spinning_forever():
    camera = OpenCvCamera()
    stop = threading.Event()
    camera._stop = stop

    class BrokenCapture(FakeCapture):
        def read(self):
            return False, None

    capture = BrokenCapture()
    thread = _run(camera, capture, stop)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert camera.error == "camera stopped returning frames"
    assert capture.released is True