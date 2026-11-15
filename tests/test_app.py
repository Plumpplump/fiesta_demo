# Author: David Li, SISHK
"""The app builds its controls without a page, a camera or any model file."""

from models import constants as C


def test_assets_folder_is_found():
    from services import assets

    assert assets.assets_dir().is_dir()
    assert assets.models_dir().is_dir()


def test_yunet_model_ships_with_the_repo():
    from services import assets

    assert assets.has_model(C.YUNET_MODEL_FILE)


def test_view_can_be_constructed_headless():
    from views.app_view import AppView

    view = AppView()
    assert view.preview.key == "preview"
    # An empty src makes Flutter complain ("A valid src value must be
    # specified"), so the preview ships a real 1x1 PNG and stays hidden.
    assert isinstance(view.preview.src, bytes) and view.preview.src.startswith(b"\x89PNG")
    assert view.preview.visible is False
    # Without gapless playback Flutter blanks the preview while each new JPEG
    # decodes, which at 12 fps reads as flicker.
    assert view.preview.gapless_playback is True
    # The camera is user-started: the button must say so before anything runs.
    assert view.camera_button.content == "Start camera"
    assert view.glasses_slider.key == "glasses_threshold"
    # Settings live under the camera and only appear when the gear is pressed.
    assert view.settings_panel.key == "settings_panel"
    assert view.settings_panel.visible is False
    assert view.appbar.actions == [view.settings_button]
    assert view.status_list.controls  # empty state is rendered immediately
    view.set_frame(b"\xff\xd8")  # bytes go straight into ft.Image.src
    assert view.preview.visible is True
    assert view.placeholder.visible is False
    view.set_frame(b"\xff\xd9")
    assert view.preview.src == b"\xff\xd9"
    assert view.preview.visible is True
    view.show_placeholder("Camera stopped")
    assert view.preview.visible is False
    view.render_people(
        [
            {
                "label": "Person1",
                "uniform": "in uniform",
                "uniform_score": 0.31,
                "glasses": "glasses",
                "glasses_score": 0.24,
            }
        ]
    )
    assert len(view.status_list.controls) == 1


def test_layout_settings_under_camera_and_full_height_status():
    import flet as ft

    from views.app_view import AppView

    view = AppView()
    row = view.root.controls[-1]
    assert isinstance(row, ft.Row)
    assert row.vertical_alignment == ft.CrossAxisAlignment.STRETCH
    assert [view.camera_pane, view.status_pane] == row.controls
    assert view.camera_pane.expand == 3 and view.status_pane.expand == 2  # 60/40

    camera_children = view.camera_pane.content.controls
    assert camera_children[0] is view.preview_box
    assert camera_children[-1] is view.settings_panel  # under the camera

    status_children = view.status_pane.content.controls
    assert status_children[-1] is view.status_list
    assert view.status_list.expand is True  # fills the whole right pane
    assert view.status_pane.content.expand is True


def test_person_cards_use_vivid_colours_and_big_badges():
    from views import app_view

    view = app_view.AppView()
    view.render_people(
        [
            {
                "label": "Person1",
                "track_id": 7,
                "uniform": "in uniform",
                "uniform_score": 0.9,
                "glasses": "glasses",
                "glasses_score": 0.5,
                "detail": "",
            }
        ]
    )
    card = view.status_list.controls[0]
    label = card.content.controls[0]
    assert label.color == app_view.person_color(7)
    assert label.size >= 22  # names are easy to read from across the room
    badges = card.content.controls[1].controls
    assert all(badge.content.size >= 18 for badge in badges)

    # Different people get different, vivid colours - never white/light grey.
    assert len({app_view.person_color(i) for i in range(6)}) == 6
    banned = {"#FFFFFF", "#F5F5F5", "#EEEEEE", "#E0E0E0"}
    assert not set(app_view._PERSON_COLORS) & banned


def test_reference_photos_are_picked_up():
    from models.uniform import load_references
    from services import assets

    references = load_references(assets.assets_dir())
    assert references, "expected the school uniform reference images in src/assets"


def test_main_wires_the_page_without_a_flutter_host():
    import main as app_main

    class StubPage:
        title = None
        padding = None
        appbar = None
        on_disconnect = None

        def __init__(self):
            self.added = []

        def add(self, *controls):
            self.added.extend(controls)

        def update(self):
            pass

        def run_task(self, handler):
            # `handler` is a coroutine function in main.py; do not start it.
            if hasattr(handler, "close"):
                handler.close()

    page = StubPage()
    app_main.main(page)

    assert page.title
    assert page.appbar is not None  # the gear lives here
    assert page.added
    assert callable(page.on_disconnect)


def test_app_module_imports():
    import main

    assert callable(main.main)

def test_app_bar_title_is_bold_and_black():
    from views.app_view import AppView

    title = AppView().appbar.title
    assert title.value == "Fiesta AI Vision Demo-Detection of Uniform and Eyewear"
    assert title.weight == "bold" or str(title.weight).lower().endswith("bold")
    assert title.color == "#000000"


def test_slider_endpoints_tolerate_the_client_rounding_error():
    """Flutter computes the far-right value as min + step*divisions, which can
    land one float ulp ABOVE max (0.8500000000000001 > 0.85). Flet rejects that
    and the app dies, so the control max carries a margin."""
    from views.app_view import AppView

    view = AppView()
    for slider, intended_top in (
        (view.threshold_slider, 0.85),
        (view.glasses_slider, 0.60),
    ):
        assert slider.max > intended_top
        assert intended_top + 1e-16 <= slider.max  # the observed overshoot
        assert slider.min < intended_top


def test_slider_values_start_inside_their_bounds():
    from views.app_view import AppView

    view = AppView()
    for slider in (view.threshold_slider, view.glasses_slider):
        assert slider.min <= slider.value <= slider.max
