# Author: David Li, SISHK
"""Wiring: camera -> detector -> tracks -> classifiers -> view.

Two loops keep the UI responsive on a MacBook:
  * preview loop, ~12 fps, encodes the newest frame and updates the image
  * detection loop, ~6 fps, runs YOLO (in a worker thread) and classifies
    only people that are new or whose verdict is stale
"""

from __future__ import annotations

import asyncio
import time

import flet as ft

from models import constants as C
from models.clip import ClipEncoder, PromptBank
from models.face import FaceDetector, face_box, pick_face
from models.frames import draw_overlay, to_preview_jpeg
from models.geometry import (
    area_ratio,
    clip_box,
    crop,
    eye_box_from_face,
    face_anchored_torso_box,
)
from models.glasses import GlassesClassifier, decide_glasses
from models.person_detector import ModelUnavailable, YoloOnnxDetector
from models.status import build_status_rows, needs_classification, smoothed_score
from models.tracks import TrackManager
from models.types import CHECKING
from models.uniform import UniformClassifier, decide_uniform, load_references
from services import assets
from services.camera import OpenCvCamera
from views.app_view import AppView


class AppController:
    def __init__(self, page: ft.Page):
        self.page = page
        self.view = AppView()
        self.camera = OpenCvCamera()
        self.tracks = TrackManager()
        self.detector: YoloOnnxDetector | None = None
        self.face_detector: FaceDetector | None = None
        self.uniform: UniformClassifier | None = None
        self.glasses: GlassesClassifier | None = None
        self.uniform_threshold = C.UNIFORM_PROB_THRESHOLD
        self.glasses_threshold = C.GLASSES_PROB_THRESHOLD

        self._running = False
        self._tasks: list[asyncio.Future] = []
        self._notes: list[str] = []
        self._last_detected_frame = -1
        self._last_person_count = -1
        self._judged_ids: set[int] = set()
        self._toggling = False

    # -- setup -------------------------------------------------------------
    @property
    def _has_classifier(self) -> bool:
        return self.uniform is not None or self.glasses is not None

    def mount(self) -> None:
        self.view.camera_button.on_click = self.on_toggle_camera
        self.view.settings_button.on_click = self.on_toggle_settings
        self.view.threshold_slider.on_change = self.on_threshold_change
        self.view.threshold_slider.value = self.uniform_threshold
        self.view.set_threshold_label(self.uniform_threshold)
        self.view.glasses_slider.on_change = self.on_glasses_threshold_change
        self.view.glasses_slider.value = self.glasses_threshold
        self.view.set_glasses_threshold_label(self.glasses_threshold)
        self.page.appbar = self.view.appbar
        self.page.add(self.view.root)

    def _load_prompts(self, filename: str) -> PromptBank | None:
        try:
            return PromptBank.load(assets.model_file(filename))
        except ModelUnavailable:
            return None

    def _load_pipeline(self) -> None:
        """Blocking model loading; call from a worker thread."""
        try:
            self.detector = YoloOnnxDetector(assets.model_file(C.YOLO_MODEL_FILE))
        except ModelUnavailable as exc:
            self.detector = None
            self._notes.append(
                f"Person detection off: {C.YOLO_MODEL_FILE} missing - run "
                "scripts/export_models.py"
            )

        encoder: ClipEncoder | None = None
        try:
            encoder = ClipEncoder(assets.model_file(C.CLIP_VISION_MODEL_FILE))
        except ModelUnavailable:
            self._notes.append(
                "AI uniform comparison off: export the CLIP vision model "
                "(scripts/export_models.py); colour check still runs"
            )
        except Exception as exc:  # pragma: no cover - local runtime specific
            self._notes.append(f"CLIP encoder failed: {exc}")

        try:
            self.face_detector = FaceDetector(assets.model_file(C.YUNET_MODEL_FILE))
        except ModelUnavailable:
            self.face_detector = None
            self._notes.append("Eye/glasses check off: YuNet model missing")

        references = load_references(assets.assets_dir())
        if not references:
            self._notes.append("No uniform reference photos in src/assets/reference*")

        self.uniform = UniformClassifier(
            references, encoder, self._load_prompts("uniform_prompts.npy")
        )
        self.glasses = GlassesClassifier(encoder, self._load_prompts("glasses_prompts.npy"))

    async def start(self) -> None:
        """Load the models and wait. The camera starts only on the button."""
        try:
            await asyncio.to_thread(self._load_pipeline)
        except Exception as exc:  # pragma: no cover - defensive
            self._notes.append(f"pipeline error: {exc}")

        if self._notes:
            self.view.show_banner(" | ".join(self._notes))

        self.view.camera_button.content = "Start camera"
        self.view.show_placeholder("Press 'Start camera' to begin")
        self.page.update()

    def _start_loops(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            self.page.run_task(self._preview_loop),
            self.page.run_task(self._detect_loop),
        ]

    def _stop_loops(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        # A stopped camera starts a fresh session: stale tracks (and their
        # cached verdicts) must not be re-matched to whoever stands there next.
        self.tracks.reset()
        self._judged_ids = set()
        self._last_detected_frame = -1
        self._last_person_count = -1
        self.view.render_people([])

    # -- loops -------------------------------------------------------------
    def _safe_update(self) -> None:
        """Update the page, tolerating a session that is closing.

        Without this, closing the window mid-loop prints
        "RuntimeError: An attempt to fetch destroyed session" from the preview
        loop's final update.
        """
        try:
            self.page.update()
        except RuntimeError:  # session destroyed while closing
            self._running = False

    async def _preview_loop(self) -> None:
        interval = 1.0 / C.PREVIEW_MAX_FPS
        last_frame_id = -1
        while self._running:
            started = time.monotonic()
            frame = self.camera.read_latest()
            frame_id = self.camera.frame_id
            # A stalled camera hands back the same frame forever. Re-pushing it
            # would restart the image provider each tick and flicker the
            # preview, so only encode genuinely new frames.
            if frame is None:
                await asyncio.sleep(0.05)
                continue
            if frame_id == last_frame_id:
                await asyncio.sleep(interval)
                continue
            last_frame_id = frame_id

            annotated = draw_overlay(
                frame.copy(),
                [t for t in self.tracks.visible if t.track_id in self._judged_ids],
            )
            jpeg = await asyncio.to_thread(to_preview_jpeg, annotated)
            self.view.set_frame(jpeg)
            height, width = annotated.shape[:2]
            self.view.set_fps(self.camera.fps, width, height)
            self._safe_update()
            if not self._running:
                return
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, interval - elapsed))

    async def _detect_loop(self) -> None:
        interval = 1.0 / C.DETECT_FPS
        while self._running:
            await asyncio.sleep(interval)
            if self.detector is None:
                continue

            frame = self.camera.read_latest()
            frame_id = self.camera.frame_id
            if frame is None or frame_id == self._last_detected_frame:
                continue
            self._last_detected_frame = frame_id

            detections = await asyncio.to_thread(self.detector.detect, frame)
            height, width = frame.shape[:2]
            # Every detection goes to the tracker: identity must survive a
            # person drifting below the 15% area gate. Filtering boxes here is
            # what recycled Person1/Person3 and threw away their verdicts.
            boxes = [d.box for d in detections]

            now = time.monotonic()
            active, _new = self.tracks.update(boxes, now, frame_height=height)
            # Keep the face / eye / torso boxes glued to the person at the
            # detection rate. Before this, they were only refreshed when a
            # verdict was re-run (every few seconds), so the blue eye box drifted off
            # the eyes as soon as somebody moved.
            self._refresh_face_boxes(frame, active)

            judged = self._apply_area_gate(active, width, height)

            for track in judged:
                if needs_classification(track, now) and self._has_classifier:
                    await asyncio.to_thread(self._classify, track, frame)
                    # One person per iteration keeps the loop responsive: the
                    # CLIP pass costs ~0.2 s on this M4 (two crops), ~1 s on
                    # weak CPUs.
                    break

            self.view.render_people(build_status_rows(judged))
            if len(judged) != self._last_person_count:
                self._last_person_count = len(judged)
                scores = ", ".join(f"{d.score:.2f}" for d in detections)
                gate = int(C.MIN_PERSON_AREA_RATIO * 100)
                print(
                    f"[detect] {len(judged)} judged of {len(detections)} detected "
                    f"({gate}% area gate) | YOLO scores: {scores or 'none'}",
                    flush=True,
                )
            self._safe_update()
            if not self._running:
                return

    def _apply_area_gate(self, active, width: int, height: int) -> list:
        """The plan.txt 15% rule, with hysteresis and no effect on identity.

        A track that was visible stays visible until it drops below 85% of the
        limit, so somebody standing at the boundary does not blink in and out
        (or, worse, get a new Person number and lose their verdict).
        """
        judged = []
        for track in active:
            ratio = area_ratio(track.box, width, height)
            limit = C.MIN_PERSON_AREA_RATIO
            if track.track_id in self._judged_ids:
                limit *= C.AREA_GATE_HYSTERESIS
            if ratio >= limit:
                judged.append(track)
        self._judged_ids = {t.track_id for t in judged}
        return judged

    def _refresh_face_boxes(self, frame, tracks) -> None:
        """Update face/eye/torso boxes for the preview, one YuNet pass per frame."""
        if self.face_detector is None or not tracks:
            return
        try:
            faces = self.face_detector.detect(frame)
        except Exception:  # pragma: no cover - camera/model runtime hiccup
            return
        height, width = frame.shape[:2]
        for track in tracks:
            face = pick_face(faces, track.box)
            if face is None:
                track.face_box = None
                track.eye_box = None
                continue
            track.face_box = clip_box(face_box(face), float(width), float(height))
            anchored = face_anchored_torso_box(track.box, track.face_box)
            if anchored is not None:
                track.torso_box = anchored
            eye_box = eye_box_from_face(face)
            track.eye_box = (
                clip_box(eye_box, float(width), float(height))
                if eye_box is not None
                else None
            )

    def _classify(self, track, frame) -> None:
        """Run face/glasses + uniform for one person. Blocking; worker thread.

        The face is found first: a face-anchored shirt crop is more reliable
        than guessing the band from the person box (this is the enhancement the
        sibling project listed as not-yet-wired), and the same YuNet pass
        provides the eye box for glasses.
        """
        face = None
        uniform_sample: float | None = None
        glasses_sample: float | None = None
        frame_height, frame_width = frame.shape[:2]
        if self.face_detector is not None:
            try:
                face = pick_face(self.face_detector.detect(frame), track.box)
            except Exception:  # pragma: no cover - camera/model runtime hiccup
                face = None
        if face is not None:
            # YuNet returns x/y/w/h; store corners, clipped to the frame so the
            # drawn box can never leave the picture.
            track.face_box = clip_box(
                face_box(face), float(frame_width), float(frame_height)
            )
            anchored = face_anchored_torso_box(track.box, track.face_box)
            if anchored is not None:
                # Show the band that was actually classified.
                track.torso_box = anchored

        if self.uniform is not None:
            verdict = self.uniform.classify(
                crop(frame, track.torso_box), threshold=self.uniform_threshold
            )
            uniform_sample = verdict.score
            if verdict.score is not None:
                track.uniform_scores.append(verdict.score)
                track.white_ratio = verdict.white_ratio or 0.0
                track.red_ratio = verdict.red_ratio or 0.0
            # Publish the median of the recent checks, not the newest sample:
            # one crop must not flip an otherwise steady verdict.
            smoothed = smoothed_score(track.uniform_scores)
            track.uniform = (
                CHECKING
                if smoothed is None
                else decide_uniform(
                    smoothed,
                    track.white_ratio,
                    track.red_ratio,
                    similarity_threshold=self.uniform_threshold,
                ).label
            )
            track.uniform_score = smoothed
            track.uniform_detail = verdict.detail

        track.last_classified = time.monotonic()
        track.classified_box = track.box
        track.checks += 1

        if self.glasses is not None and face is not None:
            eye_box = eye_box_from_face(face)
            if eye_box is None:
                track.eye_box = None
            else:
                eye_box = clip_box(eye_box, float(frame_width), float(frame_height))
                track.eye_box = tuple(float(v) for v in eye_box)
                # Two views: the whole face (context: frame shape, temples)
                # and the eye band (detail: the frame itself); the higher one
                # wins so a blurred view cannot veto a clear one.
                verdict = self.glasses.classify(
                    crop(frame, track.face_box),
                    crop(frame, eye_box),
                    threshold=self.glasses_threshold,
                )
                glasses_sample = verdict.score
                decisive_first = False
                if verdict.score is not None:
                    track.glasses_scores.append(verdict.score)
                    # A clear first sample ("definitely glasses" / "definitely
                    # none") is published at once instead of waiting for the
                    # median, so glasses do not lag behind the uniform verdict.
                    decisive_first = (
                        verdict.score >= C.GLASSES_SHORTCUT_HIGH
                        or verdict.score <= C.GLASSES_SHORTCUT_LOW
                    )
                smoothed_glasses = smoothed_score(track.glasses_scores)
                if smoothed_glasses is None and decisive_first:
                    published = glasses_sample
                else:
                    published = smoothed_glasses
                track.glasses = decide_glasses(
                    published, self.glasses_threshold
                ).label
                track.glasses_score = published
                track.glasses_detail = verdict.detail

        self._log_verdict(track, uniform_sample, glasses_sample)

    @staticmethod
    def _log_verdict(
        track,
        uniform_sample: float | None = None,
        glasses_sample: float | None = None,
    ) -> None:
        """Print published + sample scores: the terminal is the demo's debug view."""

        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        print(
            f"[detect] {C.PERSON_LABEL_PREFIX}{track.label_index}: "
            f"uniform={track.uniform} p={fmt(track.uniform_score)} "
            f"(sample {fmt(uniform_sample)}) | "
            f"glasses={track.glasses} p={fmt(track.glasses_score)} "
            f"(sample {fmt(glasses_sample)})",
            flush=True,
        )
        for detail in (track.uniform_detail, track.glasses_detail):
            if detail:
                print(f"           {detail}", flush=True)

    # -- events ------------------------------------------------------------
    async def on_toggle_camera(self, e) -> None:
        """Start/stop the camera. Async so the button never freezes mid-open."""
        if self._toggling:
            return
        self._toggling = True
        self.view.camera_button.disabled = True
        try:
            if self.camera.running:
                self.view.camera_button.content = "Stopping..."
                self.page.update()
                self._stop_loops()
                await asyncio.to_thread(self.camera.stop)
                self.view.camera_button.content = "Start camera"
                self.view.show_placeholder("Camera stopped")
                self.view.show_banner("Camera stopped. Click 'Start camera' to resume.")
            else:
                self.view.camera_button.content = "Starting..."
                self.page.update()
                started = await asyncio.to_thread(self.camera.start)
                if not started:
                    # macOS often needs a moment after a release before the
                    # device can be opened again; one retry fixes it.
                    await asyncio.sleep(0.4)
                    started = await asyncio.to_thread(self.camera.start)
                if started:
                    self.view.hide_banner()
                    self.view.show_placeholder("Waiting for camera frames...")
                    self.view.camera_button.content = "Stop camera"
                    self._start_loops()
                else:
                    self.view.show_banner(self.camera.error or "camera unavailable")
                    self.view.camera_button.content = "Start camera"
        finally:
            self.view.camera_button.disabled = False
            self._toggling = False
            self.page.update()

    def on_toggle_settings(self, e) -> None:
        """Show/hide the settings under the camera (app-bar gear button)."""
        panel = self.view.settings_panel
        panel.visible = not panel.visible
        button = self.view.settings_button
        button.icon = ft.Icons.CLOSE if panel.visible else ft.Icons.SETTINGS
        button.tooltip = "Hide settings" if panel.visible else "Show settings"
        self.page.update()

    def on_threshold_change(self, e) -> None:
        try:
            value = float(e.control.value)
        except (TypeError, ValueError):
            return
        # The slider's control max carries a float margin (see _SLIDER_MARGIN);
        # the actual threshold never leaves the documented range.
        value = min(max(value, C.UNIFORM_THRESHOLD_MIN), C.UNIFORM_THRESHOLD_MAX)
        self.uniform_threshold = round(value, 2)
        self.view.threshold_slider.value = self.uniform_threshold
        self.view.set_threshold_label(self.uniform_threshold)
        for track in self.tracks.tracks:
            track.last_classified = 0.0  # re-check everyone with the new threshold
        self.page.update()

    def on_glasses_threshold_change(self, e) -> None:
        try:
            value = float(e.control.value)
        except (TypeError, ValueError):
            return
        value = min(max(value, C.GLASSES_THRESHOLD_MIN), C.GLASSES_THRESHOLD_MAX)
        self.glasses_threshold = round(value, 2)
        self.view.glasses_slider.value = self.glasses_threshold
        self.view.set_glasses_threshold_label(self.glasses_threshold)
        for track in self.tracks.tracks:
            track.last_classified = 0.0
        self.page.update()

    def on_disconnect(self, e) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self._stop_loops()
        self.camera.stop()