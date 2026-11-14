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
from models.face import FaceDetector, pick_face
from models.frames import draw_overlay, to_preview_jpeg
from models.geometry import area_ratio, crop, eye_box_from_face
from models.glasses import GlassesClassifier
from models.person_detector import ModelUnavailable, YoloOnnxDetector
from models.status import build_status_rows, needs_classification
from models.tracks import TrackManager
from models.uniform import UniformClassifier, load_references
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
        self.uniform_threshold = C.UNIFORM_SIM_THRESHOLD

        self._running = False
        self._tasks: list[asyncio.Future] = []
        self._notes: list[str] = []
        self._last_detected_frame = -1

    # -- setup -------------------------------------------------------------
    @property
    def _has_classifier(self) -> bool:
        return self.uniform is not None or self.glasses is not None

    def mount(self) -> None:
        self.view.camera_button.on_click = self.on_toggle_camera
        self.view.threshold_slider.on_change = self.on_threshold_change
        self.view.threshold_slider.value = self.uniform_threshold
        self.view.set_threshold_label(self.uniform_threshold)
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
        try:
            await asyncio.to_thread(self._load_pipeline)
        except Exception as exc:  # pragma: no cover - defensive
            self._notes.append(f"pipeline error: {exc}")

        if self._notes:
            self.view.show_banner(" | ".join(self._notes))

        if not self.camera.start():
            self.view.show_placeholder("No camera available")
            self.view.show_banner(self.camera.error or "camera unavailable")
            self.page.update()
            return

        self.view.show_placeholder("Waiting for camera frames...")
        self.view.camera_button.content = "Stop camera"
        self.page.update()

        self._running = True
        self._tasks = [
            self.page.run_task(self._preview_loop),
            self.page.run_task(self._detect_loop),
        ]

    # -- loops -------------------------------------------------------------
    async def _preview_loop(self) -> None:
        interval = 1.0 / C.PREVIEW_MAX_FPS
        while self._running:
            started = time.monotonic()
            frame = self.camera.read_latest()
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            annotated = draw_overlay(
                frame.copy(),
                [track for track in self.tracks.tracks if track.misses == 0],
            )
            jpeg = await asyncio.to_thread(to_preview_jpeg, annotated)
            self.view.set_frame(jpeg)
            height, width = annotated.shape[:2]
            self.view.set_fps(self.camera.fps, width, height)
            self.page.update()
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
            boxes = [
                d.box
                for d in detections
                if area_ratio(d.box, width, height) >= C.MIN_PERSON_AREA_RATIO
            ]

            now = time.monotonic()
            active, _new = self.tracks.update(boxes, now)

            for track in active:
                if needs_classification(track, now) and self._has_classifier:
                    await asyncio.to_thread(self._classify, track, frame)
                    # One person per iteration keeps the loop responsive: the
                    # CLIP pass costs ~0.1 s on Apple silicon, ~1 s on weak CPUs.
                    break

            self.view.render_people(build_status_rows(active))
            self.page.update()

    def _classify(self, track, frame) -> None:
        """Run uniform + face/glasses for one person. Blocking; worker thread."""
        if self.uniform is not None:
            verdict = self.uniform.classify(
                crop(frame, track.torso_box), threshold=self.uniform_threshold
            )
            track.uniform = verdict.label
            track.uniform_score = verdict.score
            track.uniform_detail = verdict.detail

        track.last_classified = time.monotonic()

        if self.face_detector is None or self.glasses is None:
            return

        try:
            faces = self.face_detector.detect(frame)
        except Exception:  # pragma: no cover - camera/model runtime hiccup
            track.eye_box = None
            return
        face = pick_face(faces, track.box)
        if face is None:
            track.eye_box = None
            return

        track.face_box = tuple(float(v) for v in face[:4])
        eye_box = eye_box_from_face(face)
        if eye_box is None:
            track.eye_box = None
            return

        track.eye_box = tuple(float(v) for v in eye_box)
        verdict = self.glasses.classify(crop(frame, eye_box))
        track.glasses = verdict.label
        track.glasses_score = verdict.score

    # -- events ------------------------------------------------------------
    def on_toggle_camera(self, e) -> None:
        if self.camera.running:
            self.camera.stop()
            self.view.camera_button.content = "Start camera"
            self.view.show_placeholder("Camera stopped")
            self.view.show_banner("Camera stopped. Click 'Start camera' to resume.")
        else:
            if self.camera.start():
                self.view.hide_banner()
                self.view.show_placeholder("Waiting for camera frames...")
                self.view.camera_button.content = "Stop camera"
            else:
                self.view.show_banner(self.camera.error or "camera unavailable")
        self.page.update()

    def on_threshold_change(self, e) -> None:
        try:
            self.uniform_threshold = float(e.control.value)
        except (TypeError, ValueError):
            return
        self.view.set_threshold_label(self.uniform_threshold)
        for track in self.tracks.tracks:
            track.last_classified = 0.0  # re-check everyone with the new threshold
        self.page.update()

    def on_disconnect(self, e) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        self.camera.stop()