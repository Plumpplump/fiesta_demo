# Author: David Li, SISHK
"""Controls only: the 60/40 camera + status layout. No detection logic here."""

from __future__ import annotations

import base64

import flet as ft

from models import constants as C
from models.types import CHECKING, GLASSES, IN_UNIFORM, NOT_IN_UNIFORM, UNSURE

# 1x1 dark PNG: ft.Image rejects an empty src ("A valid src value must be
# specified"), so the preview starts with this and stays hidden until a real
# camera frame arrives.
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMQFBQEAABqADS54vDNAAAAAElFTkSuQmCC"
)
_UNIFORM_COLORS = {
    IN_UNIFORM: "#1B5E20",
    NOT_IN_UNIFORM: "#B71C1C",
    UNSURE: "#EF6C00",
    CHECKING: "#546E7A",
}
_GLASSES_COLORS = {
    GLASSES: "#0D47A1",
    "no glasses": "#37474F",
    "unknown": "#78909C",
    UNSURE: "#EF6C00",
}

# One vivid colour per person, picked from the track id so it stays stable
# across redraws. Deliberately no white or light grey: the card background is
# light and the label must stay readable.
_PERSON_COLORS = (
    "#D32F2F",  # red
    "#7B1FA2",  # purple
    "#1976D2",  # blue
    "#00897B",  # teal
    "#F57C00",  # orange
    "#5D4037",  # brown
    "#C2185B",  # pink
    "#388E3C",  # green
    "#512DA8",  # deep purple
    "#0288D1",  # light blue
    "#E64A19",  # deep orange
    "#00796B",  # dark teal
)


def person_color(track_id: int) -> str:
    """Stable, vivid colour for one person (never white/light grey)."""
    return _PERSON_COLORS[int(track_id) % len(_PERSON_COLORS)]


# Sliders are validated by Flet against max, and Flutter's endpoint arithmetic
# can overshoot by one float ulp (0.8500000000000001). A hair of margin keeps
# the far-right position legal; the handlers clamp to the real upper bound.
_SLIDER_MARGIN = 0.001


class AppView:
    def __init__(self):
        self.banner_text = ft.Text("", key="banner_text", size=13, color="#4E342E")
        self.banner = ft.Container(
            key="banner",
            visible=False,
            bgcolor="#FFE0B2",
            padding=10,
            border_radius=6,
            content=self.banner_text,
        )

        self.placeholder = ft.Text(
            "Camera starting...", key="placeholder", color="#9E9E9E", size=14
        )
        # Every camera frame is a new image provider. With gapless_playback off
        # Flutter paints *nothing* while the next JPEG decodes, which at 12 fps
        # looks like the video is flickering; gapless_playback keeps the
        # previous frame on screen until the new one is ready.
        self.preview = ft.Image(
            _PLACEHOLDER_PNG,
            key="preview",
            fit=ft.BoxFit.CONTAIN,
            filter_quality=ft.FilterQuality.MEDIUM,
            gapless_playback=True,
            expand=True,
            visible=False,
        )
        self._frame_shown = False
        self.preview_box = ft.Container(
            key="preview_box",
            expand=True,
            bgcolor="#111111",
            border_radius=8,
            alignment=ft.Alignment.CENTER,
            content=ft.Stack([self.placeholder, self.preview], expand=True),
        )

        self.fps_text = ft.Text("", key="fps", size=12, color="#9E9E9E")
        self.camera_button = ft.FilledButton(
            "Start camera", key="camera_button", on_click=None
        )
        self.slider_label = ft.Text(
            "", key="threshold_label", size=13, color="#616161"
        )
        self.threshold_slider = ft.Slider(
            key="uniform_threshold",
            min=C.UNIFORM_THRESHOLD_MIN,
            max=C.UNIFORM_THRESHOLD_MAX + _SLIDER_MARGIN,
            divisions=55,
            value=0.50,
            on_change=None,
            expand=True,
        )
        self.glasses_slider_label = ft.Text(
            "", key="glasses_threshold_label", size=13, color="#616161"
        )
        self.glasses_slider = ft.Slider(
            key="glasses_threshold",
            min=C.GLASSES_THRESHOLD_MIN,
            max=C.GLASSES_THRESHOLD_MAX + _SLIDER_MARGIN,
            divisions=58,
            value=0.15,
            on_change=None,
            expand=True,
        )

        self.summary = ft.Text("No one in view", key="summary", size=16, color="#616161")
        self.status_list = ft.Column(
            key="status_list", spacing=12, scroll=ft.ScrollMode.AUTO, expand=True
        )

        # --- settings live under the camera, behind the app-bar gear ---------
        self.settings_button = ft.IconButton(
            icon=ft.Icons.SETTINGS,
            key="settings_button",
            tooltip="Show settings",
        )
        self.settings_panel = ft.Container(
            key="settings_panel",
            visible=False,
            bgcolor="#F5F5F5",
            border_radius=8,
            padding=12,
            content=ft.Column(
                [
                    ft.Text("Settings", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text("Uniform match threshold", size=14, color="#616161"),
                    self.slider_label,
                    self.threshold_slider,
                    ft.Text("Glasses sensitivity", size=14, color="#616161"),
                    self.glasses_slider_label,
                    self.glasses_slider,
                ],
                spacing=6,
            ),
        )

        self.appbar = ft.AppBar(
            key="appbar",
            title=ft.Text(
                "Fiesta AI Vision Demo-Detection of Uniform and Eyewear",
                size=20,
                weight=ft.FontWeight.BOLD,
                color="#000000",
                max_lines=1,
                overflow=ft.TextOverflow.ELLIPSIS,
            ),
            actions=[self.settings_button],
            bgcolor="#E8EAF6",
        )

        camera_pane = ft.Container(
            expand=3,  # 60% of the width, per plan.txt
            content=ft.Column(
                [
                    self.preview_box,
                    ft.Row(
                        [self.fps_text, ft.Container(expand=True), self.camera_button],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    self.settings_panel,
                ],
                spacing=8,
                expand=True,
            ),
        )
        self.camera_pane = camera_pane
        status_pane = ft.Container(
            expand=2,  # 40% of the width
            content=ft.Column(
                [
                    ft.Text("Who is in view", size=22, weight=ft.FontWeight.BOLD),
                    self.summary,
                    ft.Divider(height=1),
                    # The list claims every pixel left in the pane.
                    self.status_list,
                ],
                spacing=10,
                expand=True,
            ),
        )
        self.status_pane = status_pane

        self.root = ft.Column(
            [
                self.banner,
                ft.Row(
                    [camera_pane, status_pane],
                    spacing=16,
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                ),
            ],
            spacing=12,
            expand=True,
        )

        self.render_people([])

    # -- updates -----------------------------------------------------------
    def set_frame(self, jpeg: bytes) -> None:
        # Visibility only flips on the first frame: touching it on every frame
        # makes the widget rebuild (and visibly blink) 12 times a second.
        if not self._frame_shown:
            self.placeholder.visible = False
            self.preview.visible = True
            self._frame_shown = True
        self.preview.src = jpeg

    def show_placeholder(self, message: str) -> None:
        """Hide the live image (no valid frame yet / camera stopped)."""
        self.placeholder.value = message
        self.placeholder.visible = True
        self.preview.visible = False
        self._frame_shown = False

    def set_fps(self, fps: float, width: int, height: int) -> None:
        self.fps_text.value = f"{fps:.1f} FPS  |  {width}x{height}"

    def show_banner(self, message: str) -> None:
        self.banner_text.value = message
        self.banner.visible = True

    def hide_banner(self) -> None:
        self.banner.visible = False

    def set_threshold_label(self, threshold: float) -> None:
        self.slider_label.value = f"{threshold:.2f} (higher = stricter, less 'in uniform')"

    def set_glasses_threshold_label(self, threshold: float) -> None:
        self.glasses_slider_label.value = (
            f"{threshold:.2f} (lower = finds more glasses)"
        )

    def render_people(self, rows: list[dict]) -> None:
        if not rows:
            self.summary.value = "No one in view"
            self.status_list.controls = [
                ft.Text(
                    "Walk into the camera view to be checked.",
                    key="empty_state",
                    size=15,
                    color="#9E9E9E",
                )
            ]
            return

        self.summary.value = f"{len(rows)} person(s) in view"
        self.status_list.controls = [self._person_card(row) for row in rows]

    def _person_card(self, row: dict) -> ft.Control:
        uniform = row.get("uniform") or CHECKING
        glasses = row.get("glasses") or "unknown"
        colour = person_color(row.get("track_id") or 0)
        scores = []
        if row.get("uniform_score") is not None:
            scores.append(f"uniform {row['uniform_score']:.2f}")
        if row.get("glasses_score") is not None:
            scores.append(f"glasses {row['glasses_score']:.2f}")

        return ft.Container(
            key=f"person_{row['label']}",
            padding=14,
            border_radius=10,
            bgcolor="#F5F5F5",
            border=ft.Border(left=ft.BorderSide(6, colour)),
            content=ft.Column(
                [
                    ft.Text(
                        row["label"],
                        size=24,
                        weight=ft.FontWeight.BOLD,
                        color=colour,
                    ),
                    ft.Row(
                        [
                            self._badge(uniform, _UNIFORM_COLORS.get(uniform, "#546E7A")),
                            self._badge(glasses, _GLASSES_COLORS.get(glasses, "#546E7A")),
                        ],
                        spacing=10,
                        wrap=True,
                    ),
                    ft.Text(
                        "  ".join(scores) or "checking...",
                        size=13,
                        color="#757575",
                    ),
                    ft.Text(
                        row.get("detail") or "",
                        size=12,
                        color="#9E9E9E",
                        max_lines=3,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                spacing=8,
            ),
        )

    @staticmethod
    def _badge(text: str, color: str) -> ft.Control:
        return ft.Container(
            bgcolor=color,
            border_radius=8,
            padding=ft.Padding(left=14, top=7, right=14, bottom=7),
            content=ft.Text(
                text,
                size=18,
                weight=ft.FontWeight.BOLD,
                color="#FFFFFF",
            ),
        )