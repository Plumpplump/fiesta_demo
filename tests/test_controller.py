"""Controller wiring, exercised with fakes: no camera, no models, no Flutter."""

import asyncio

import numpy as np

from controllers.app_controller import AppController
from models import constants as C
from models.types import IN_UNIFORM, Detection
from models.uniform import Verdict


class FakePage:
    def __init__(self):
        self.controls = []
        self.updates = 0

    def add(self, *controls):
        self.controls.extend(controls)

    def update(self):
        self.updates += 1

    def run_task(self, handler):
        return asyncio.ensure_future(handler())


class FakeCamera:
    def __init__(self, frame, error=None):
        self.frame = frame
        self.error = error
        self.fps = 12.0
        self.running = True
        self.frame_id = 0

    def start(self):
        return self.error is None

    def stop(self):
        self.running = False

    def read_latest(self):
        self.frame_id += 1
        return self.frame


class FakeDetector:
    def __init__(self, box):
        self.box = box
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [Detection(box=self.box, score=0.9)]


class FakeUniform:
    def __init__(self, verdict=None):
        self.verdict = verdict or Verdict(IN_UNIFORM, 0.42, "fake match")
        self.calls = 0

    def classify(self, crop_bgr, threshold=None):
        self.calls += 1
        return self.verdict


def _controller():
    page = FakePage()
    controller = AppController(page)
    controller.mount()
    return page, controller


def _frame():
    return np.full((720, 1280, 3), 180, dtype=np.uint8)


async def test_new_person_is_classified_once_per_track():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())
    controller.detector = FakeDetector((100.0, 100.0, 700.0, 600.0))
    controller.uniform = FakeUniform()

    controller._running = True
    task = asyncio.create_task(controller._detect_loop())
    await asyncio.sleep(0.45)
    controller._running = False
    task.cancel()

    assert controller.detector.calls >= 2
    assert controller.uniform.calls == 1  # reclassify interval has not elapsed
    assert page.updates >= 1
    assert controller.tracks.tracks[0].uniform == IN_UNIFORM


async def test_small_person_is_filtered_by_the_area_rule():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())
    # 64x36 px in a 1280x720 frame is ~0.25% - well under the 15% rule.
    controller.detector = FakeDetector((10.0, 10.0, 74.0, 46.0))
    controller.uniform = FakeUniform()

    controller._running = True
    task = asyncio.create_task(controller._detect_loop())
    await asyncio.sleep(0.35)
    controller._running = False
    task.cancel()

    assert controller.detector.calls >= 1
    assert controller.uniform.calls == 0
    assert controller.tracks.tracks == []


async def test_preview_loop_pushes_frames_to_the_image():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())

    controller._running = True
    task = asyncio.create_task(controller._preview_loop())
    await asyncio.sleep(0.35)
    controller._running = False
    task.cancel()

    assert isinstance(controller.view.preview.src, bytes)
    assert controller.view.fps_text.value.startswith("12.0 FPS")
    assert page.updates >= 1


async def test_start_without_camera_shows_a_banner_instead_of_crashing():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame(), error="camera 0 could not be opened")

    await asyncio.wait_for(controller.start(), timeout=30)

    assert controller.view.banner.visible is True
    assert "camera 0" in controller.view.banner_text.value
    assert controller.view.placeholder.value == "No camera available"


def test_threshold_change_rechecks_everyone():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())
    controller.detector = FakeDetector((100.0, 100.0, 700.0, 600.0))
    controller.uniform = FakeUniform()
    controller.tracks.update([(100.0, 100.0, 700.0, 600.0)], now=1.0)

    class FakeEvent:
        class control:
            value = 0.45

    controller.on_threshold_change(FakeEvent())

    assert controller.uniform_threshold == 0.45
    assert all(t.last_classified == 0.0 for t in controller.tracks.tracks)
    assert "0.45" in controller.view.slider_label.value