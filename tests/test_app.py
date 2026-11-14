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
    assert view.status_list.controls  # empty state is rendered immediately
    view.set_frame(b"\xff\xd8")  # bytes go straight into ft.Image.src
    assert view.preview.visible is True
    assert view.placeholder.visible is False
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


def test_reference_photos_are_picked_up():
    from models.uniform import load_references
    from services import assets

    references = load_references(assets.assets_dir())
    assert references, "expected the school uniform reference images in src/assets"


def test_app_module_imports():
    import main

    assert callable(main.main)