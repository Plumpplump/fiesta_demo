"""Status rows and re-classification timing."""

from models import constants as C
from models.status import build_status_rows, needs_classification
from models.tracks import TrackManager


def _track(label_index: int, x: float):
    from models.types import Track

    return Track(
        track_id=label_index,
        label_index=label_index,
        box=(x, 0.0, x + 100.0, 200.0),
        torso_box=(x + 18.0, 36.0, x + 82.0, 144.0),
    )


def test_rows_are_numbered_not_named():
    rows = build_status_rows([_track(1, 0.0), _track(2, 300.0)])
    assert [row["label"] for row in rows] == ["Person1", "Person2"]
    assert all("name" not in row for row in rows)


def test_rows_sort_by_assigned_number():
    rows = build_status_rows([_track(2, 10.0), _track(1, 500.0)])
    assert [row["label"] for row in rows] == ["Person1", "Person2"]


def test_new_track_needs_a_check():
    manager = TrackManager()
    _active, new = manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.0)
    assert needs_classification(new[0], now=1.0) is True


def test_fresh_verdict_is_not_rechecked_immediately():
    track = _track(1, 0.0)
    track.uniform = "in uniform"
    track.last_classified = 10.0
    assert needs_classification(track, now=10.5) is False
    assert needs_classification(track, now=10.0 + C.RECLASSIFY_INTERVAL + 0.1) is True