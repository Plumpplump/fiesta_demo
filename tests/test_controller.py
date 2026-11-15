# Author: David Li, SISHK
"""Controller wiring, exercised with fakes: no camera, no models, no Flutter."""

import asyncio

import flet as ft
import numpy as np

from controllers.app_controller import AppController
from models import constants as C
from models.types import CHECKING, IN_UNIFORM, Detection
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
        self.running = False
        self.frame_id = 0
        self.start_calls = 0
        self.stop_calls = 0

    def start(self):
        self.start_calls += 1
        if self.error is not None:
            return False
        self.running = True
        return True

    def stop(self):
        self.stop_calls += 1
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
        self.verdict = verdict or Verdict(IN_UNIFORM, 0.62, "fake match")
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


class FakeFaceDetector:
    """Returns one YuNet-style row: [x, y, w, h, re_x, re_y, le_x, le_y, ...]."""

    def __init__(self, row):
        self.row = row
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return [self.row]


def _face_row(x, y, width=80, height=90):
    return np.array(
        [x, y, width, height, x + 20, y + 30, x + 55, y + 30, 0, 0, 0, 0, 0, 0, 0.9],
        dtype=np.float32,
    )


def test_eye_box_follows_the_person_without_reclassifying():
    page, controller = _controller()
    controller.uniform = FakeUniform()
    controller.face_detector = FakeFaceDetector(_face_row(100, 100))
    controller.tracks.update([(50.0, 50.0, 330.0, 630.0)], now=1.0)
    controller.tracks.update([(50.0, 50.0, 330.0, 630.0)], now=1.1)
    track = controller.tracks.tracks[0]
    frame = _frame()

    controller._refresh_face_boxes(frame, [track])
    first_eye = track.eye_box
    assert first_eye is not None
    assert first_eye[0] >= track.face_box[0]  # stays on the face
    assert first_eye[2] <= track.face_box[2]

    # The person moves: the box must move with them, with no CLIP pass.
    controller.face_detector.row = _face_row(160, 120)
    controller._refresh_face_boxes(frame, [track])

    assert track.eye_box != first_eye
    assert controller.uniform.calls == 0


def test_missing_face_clears_the_eye_box():
    page, controller = _controller()
    controller.face_detector = FakeFaceDetector(_face_row(100, 100))
    controller.tracks.update([(50.0, 50.0, 330.0, 630.0)], now=1.0)
    controller.tracks.update([(50.0, 50.0, 330.0, 630.0)], now=1.1)
    track = controller.tracks.tracks[0]
    controller._refresh_face_boxes(_frame(), [track])
    assert track.eye_box is not None

    controller.face_detector.row = _face_row(2000, 2000)  # far outside the box
    controller._refresh_face_boxes(_frame(), [track])

    assert track.eye_box is None
    assert track.face_box is None


async def test_new_person_is_published_after_two_checks(monkeypatch):
    monkeypatch.setattr(C, "TRACK_SETTLE_SECONDS", 0.15)
    monkeypatch.setattr(C, "WARMUP_INTERVAL", 0.05)
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())
    controller.detector = FakeDetector((100.0, 100.0, 700.0, 600.0))
    controller.uniform = FakeUniform()

    controller._running = True
    task = asyncio.create_task(controller._detect_loop())
    await asyncio.sleep(0.9)
    controller._running = False
    task.cancel()

    assert controller.detector.calls >= 3
    # Two agreeing checks are needed before a verdict is shown, and the cache
    # then stops the loop from re-classifying every frame.
    assert controller.uniform.calls == 2
    assert page.updates >= 1
    assert controller.tracks.tracks[0].uniform == IN_UNIFORM


def _confirmed_track(controller):
    manager = controller.tracks
    manager.update([(100.0, 100.0, 700.0, 600.0)], now=1.0)
    manager.update([(100.0, 100.0, 700.0, 600.0)], now=1.1)
    manager.update([(100.0, 100.0, 700.0, 600.0)], now=1.2)
    return manager.tracks[0]


def test_first_check_is_held_as_checking():
    page, controller = _controller()
    controller.uniform = FakeUniform()
    track = _confirmed_track(controller)

    controller._classify(track, _frame())
    assert track.uniform == CHECKING  # one sample is not a verdict
    assert track.uniform_score is None
    assert track.uniform_scores == [0.62]

    controller._classify(track, _frame())
    assert track.uniform == IN_UNIFORM
    assert track.uniform_score == 0.62


def test_live_wobble_0_15_0_59_0_30_settles_as_not_in_uniform():
    """The exact sequence from the webcam log must not flip or hang."""

    page, controller = _controller()

    class SequencedUniform:
        def __init__(self, scores):
            self.scores = list(scores)
            self.calls = 0

        def classify(self, crop_bgr, threshold=None):
            score = self.scores[min(self.calls, len(self.scores) - 1)]
            self.calls += 1
            label = IN_UNIFORM if score >= 0.5 else "not in uniform"
            return Verdict(label, score, f"sample {score:.2f}")

    controller.uniform = SequencedUniform([0.15, 0.59, 0.30])
    track = _confirmed_track(controller)

    controller._classify(track, _frame())
    assert track.uniform == CHECKING
    controller._classify(track, _frame())
    assert track.uniform == "not in uniform"  # median(0.15, 0.59) = 0.37
    controller._classify(track, _frame())
    assert track.uniform == "not in uniform"  # median(0.15, 0.59, 0.30)
    assert track.uniform_score == 0.30


def test_one_wild_sample_does_not_flip_a_steady_verdict():
    page, controller = _controller()

    class SequencedUniform:
        def __init__(self, scores):
            self.scores = list(scores)
            self.calls = 0

        def classify(self, crop_bgr, threshold=None):
            score = self.scores[min(self.calls, len(self.scores) - 1)]
            self.calls += 1
            return Verdict("in uniform" if score >= 0.5 else "not in uniform", score, "")

    controller.uniform = SequencedUniform([0.90, 0.10, 0.92, 0.94])
    track = _confirmed_track(controller)

    for _ in range(4):
        controller._classify(track, _frame())

    assert track.uniform == IN_UNIFORM
    assert track.uniform_score is not None and track.uniform_score > 0.85


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
    # Identity is kept even when somebody is under the area gate; they are just
    # not judged or shown (that is what stopped the Person1/3 churn).
    assert len(controller.tracks.tracks) == 1
    assert controller.view.summary.value == "No one in view"
    assert controller._judged_ids == set()


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


async def test_preview_loop_skips_frames_it_has_already_shown():
    page, controller = _controller()

    class StaticCamera(FakeCamera):
        """Hands back the same frame id forever, like a stalled webcam."""

        def read_latest(self):
            return self.frame

    controller.camera = StaticCamera(_frame())

    controller._running = True
    task = asyncio.create_task(controller._preview_loop())
    await asyncio.sleep(0.3)
    controller._running = False
    task.cancel()

    # First frame is shown, the identical repeat is dropped: re-pushing it
    # would restart the image provider and flicker the preview.
    assert page.updates == 1
    assert controller.view.preview.visible is True


async def test_start_waits_for_the_user_to_press_the_button():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())
    controller._load_pipeline = lambda: None  # no models needed for this test

    await asyncio.wait_for(controller.start(), timeout=30)

    assert controller.camera.start_calls == 0  # the camera must not open by itself
    assert controller.camera.running is False
    assert controller._running is False  # loops are not running yet
    assert controller.view.camera_button.content == "Start camera"
    assert "Press 'Start camera'" in controller.view.placeholder.value
    assert controller.view.banner.visible is False


async def test_pressing_start_with_a_broken_camera_shows_a_banner():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame(), error="camera 0 could not be opened")

    await controller.on_toggle_camera(None)

    assert controller._running is False
    assert controller.view.banner.visible is True
    assert "camera 0" in controller.view.banner_text.value
    assert controller.view.camera_button.content == "Start camera"
    assert controller.camera.start_calls == 2  # opens once, retries once
    assert controller.view.camera_button.disabled is False


async def test_pressing_start_launches_the_loops():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())

    await controller.on_toggle_camera(None)

    assert controller._running is True
    assert controller.view.camera_button.content == "Stop camera"
    assert len(controller._tasks) == 2
    for task in controller._tasks:
        task.cancel()


async def test_pressing_stop_stops_the_camera_and_toggles_the_label():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())

    await controller.on_toggle_camera(None)   # start
    for task in controller._tasks:
        task.cancel()
    await controller.on_toggle_camera(None)   # stop

    assert controller.camera.running is False
    assert controller._running is False
    assert controller.view.camera_button.content == "Start camera"
    assert controller.view.placeholder.value == "Camera stopped"


def test_settings_button_shows_the_panel_under_the_camera():
    page, controller = _controller()
    assert controller.view.settings_panel.visible is False

    controller.on_toggle_settings(None)

    assert controller.view.settings_panel.visible is True
    assert controller.view.settings_button.icon == ft.Icons.CLOSE

    controller.on_toggle_settings(None)

    assert controller.view.settings_panel.visible is False
    assert controller.view.settings_button.icon == ft.Icons.SETTINGS


def test_mount_installs_the_app_bar():
    page, controller = _controller()
    assert page.appbar is controller.view.appbar


def test_glasses_slider_rechecks_everyone():
    page, controller = _controller()
    controller.camera = FakeCamera(_frame())
    controller.tracks.update([(100.0, 100.0, 700.0, 600.0)], now=1.0)
    controller.tracks.update([(100.0, 100.0, 700.0, 600.0)], now=1.1)
    for track in controller.tracks.tracks:
        track.last_classified = 99.0

    class FakeEvent:
        class control:
            value = 0.06

    controller.on_glasses_threshold_change(FakeEvent())

    assert controller.glasses_threshold == 0.06
    assert "0.06" in controller.view.glasses_slider_label.value
    assert all(t.last_classified == 0.0 for t in controller.tracks.tracks)


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

class _StubTrack:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box


def test_area_gate_hysteresis_keeps_a_borderline_person_stable():
    page, controller = _controller()
    width, height = 1280, 720  # 15% of the frame is 138_240 px

    shown = _StubTrack(1, (0.0, 0.0, 400.0, 400.0))  # 17.4%
    too_far = _StubTrack(2, (0.0, 0.0, 365.0, 365.0))  # 14.5%

    judged = controller._apply_area_gate([shown, too_far], width, height)
    assert [t.track_id for t in judged] == [1]

    # Person 1 drifts to 14.2%: already shown, so they stay (no blink).
    drifting = _StubTrack(1, (0.0, 0.0, 362.0, 362.0))
    assert [t.track_id for t in controller._apply_area_gate([drifting], width, height)] == [1]

    # A newcomer at that same size is not shown: entry still needs 15%.
    newcomer = _StubTrack(3, (0.0, 0.0, 362.0, 362.0))
    assert controller._apply_area_gate([newcomer], width, height) == []


def test_closing_the_window_mid_loop_does_not_raise():
    page, controller = _controller()

    class DestroyedPage(FakePage):
        def update(self):
            raise RuntimeError("An attempt to fetch destroyed session.")

    controller.page = DestroyedPage()
    controller._running = True

    controller._safe_update()  # must swallow the closing-session error

    assert controller._running is False


def test_uniform_slider_top_is_clamped_to_the_documented_max():
    page, controller = _controller()

    class FakeEvent:
        class control:
            value = 0.8500000000000001  # what Flutter sends at the far right

    controller.on_threshold_change(FakeEvent())

    assert controller.uniform_threshold == 0.85
    assert "0.85" in controller.view.slider_label.value
    assert controller.view.threshold_slider.value == 0.85


def test_glasses_slider_top_is_clamped_too():
    page, controller = _controller()

    class FakeEvent:
        class control:
            value = 0.6000000000000001

    controller.on_glasses_threshold_change(FakeEvent())

    assert controller.glasses_threshold == 0.6
    assert "0.60" in controller.view.glasses_slider_label.value
