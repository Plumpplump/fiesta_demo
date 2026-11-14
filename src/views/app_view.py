"""Controls only: the 60/40 camera + status layout. No detection logic here."""

from __future__ import annotations

import base64

import flet as ft

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
        self.preview = ft.Image(
            _PLACEHOLDER_PNG,
            key="preview",
            fit=ft.BoxFit.CONTAIN,
            expand=True,
            visible=False,
        )
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
            "Stop camera", key="camera_button", on_click=None
        )
        self.slider_label = ft.Text(
            "", key="threshold_label", size=12, color="#616161"
        )
        self.threshold_slider = ft.Slider(
            key="uniform_threshold",
            min=0.05,
            max=0.60,
            divisions=55,
            value=0.22,
            on_change=None,
            expand=True,
        )

        self.summary = ft.Text("No one in view", key="summary", size=14, color="#616161")
        self.status_list = ft.Column(
            key="status_list", spacing=8, scroll=ft.ScrollMode.AUTO, expand=True
        )

        camera_pane = ft.Container(
            col={"sm": 12, "md": 7},
            content=ft.Column(
                [
                    self.preview_box,
                    ft.Row(
                        [self.fps_text, ft.Container(expand=True), self.camera_button],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                ],
                spacing=8,
                expand=True,
            ),
        )
        status_pane = ft.Container(
            col={"sm": 12, "md": 5},
            content=ft.Column(
                [
                    ft.Text("Who is in view", size=18, weight=ft.FontWeight.BOLD),
                    self.summary,
                    ft.Divider(height=1),
                    self.status_list,
                    ft.Divider(height=1),
                    ft.Text("Uniform match threshold", size=12, color="#616161"),
                    self.slider_label,
                    self.threshold_slider,
                ],
                spacing=8,
                expand=True,
            ),
        )

        self.root = ft.Column(
            [
                ft.Text(
                    "Fiesta Demo - School Uniform Check",
                    size=20,
                    weight=ft.FontWeight.BOLD,
                ),
                self.banner,
                ft.ResponsiveRow([camera_pane, status_pane], spacing=16, expand=True),
            ],
            spacing=12,
            expand=True,
        )

        self.render_people([])

    # -- updates -----------------------------------------------------------
    def set_frame(self, jpeg: bytes) -> None:
        self.placeholder.visible = False
        self.preview.src = jpeg
        self.preview.visible = True

    def show_placeholder(self, message: str) -> None:
        """Hide the live image (no valid frame yet / camera stopped)."""
        self.placeholder.value = message
        self.placeholder.visible = True
        self.preview.visible = False

    def set_fps(self, fps: float, width: int, height: int) -> None:
        self.fps_text.value = f"{fps:.1f} FPS  |  {width}x{height}"

    def show_banner(self, message: str) -> None:
        self.banner_text.value = message
        self.banner.visible = True

    def hide_banner(self) -> None:
        self.banner.visible = False

    def set_threshold_label(self, threshold: float) -> None:
        self.slider_label.value = f"{threshold:.2f} (loose = more 'in uniform')"

    def render_people(self, rows: list[dict]) -> None:
        if not rows:
            self.summary.value = "No one in view"
            self.status_list.controls = [
                ft.Text(
                    "Walk into the camera view to be checked.",
                    key="empty_state",
                    size=13,
                    color="#9E9E9E",
                )
            ]
            return

        self.summary.value = f"{len(rows)} person(s) in view"
        self.status_list.controls = [self._person_card(row) for row in rows]

    def _person_card(self, row: dict) -> ft.Control:
        uniform = row.get("uniform") or CHECKING
        glasses = row.get("glasses") or "unknown"
        scores = []
        if row.get("uniform_score") is not None:
            scores.append(f"uniform {row['uniform_score']:.2f}")
        if row.get("glasses_score") is not None:
            scores.append(f"glasses {row['glasses_score']:.2f}")

        return ft.Container(
            key=f"person_{row['label']}",
            padding=10,
            border_radius=8,
            bgcolor="#F5F5F5",
            content=ft.Column(
                [
                    ft.Text(row["label"], size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        [
                            self._badge(uniform, _UNIFORM_COLORS.get(uniform, "#546E7A")),
                            self._badge(glasses, _GLASSES_COLORS.get(glasses, "#546E7A")),
                        ],
                        spacing=8,
                        wrap=True,
                    ),
                    ft.Text(
                        "  ".join(scores) or "checking...",
                        size=11,
                        color="#757575",
                    ),
                ],
                spacing=6,
            ),
        )

    @staticmethod
    def _badge(text: str, color: str) -> ft.Control:
        return ft.Container(
            bgcolor=color,
            border_radius=4,
            padding=ft.Padding(left=8, top=3, right=8, bottom=3),
            content=ft.Text(text, size=12, color="#FFFFFF"),
        )