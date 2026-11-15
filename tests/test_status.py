# Author: David Li, SISHK
"""Status rows, the warm-up hold and the verdict cache (re-check timing)."""

from models import constants as C
from models.status import (
    build_status_rows,
    moved_a_lot,
    needs_classification,
    smoothed_score,
)
from models.tracks import TrackManager
from models.types import CHECKING, IN_UNIFORM


def _track(label_index: int, x: float):
    from models.types import Track

    return Track(
        track_id=label_index,
        label_index=label_index,
        box=(x, 0.0, x + 100.0, 200.0),
        torso_box=(x + 8.0, 24.0, x + 92.0, 124.0),
        confirmed=True,
        uniform=IN_UNIFORM,
        uniform_scores=[0.9, 0.9],
        checks=2,
    )


def test_rows_are_numbered_not_named():
    rows = build_status_rows([_track(1, 0.0), _track(2, 300.0)])
    assert [row["label"] for row in rows] == ["Person1", "Person2"]
    assert all("name" not in row for row in rows)


def test_rows_sort_by_assigned_number():
    rows = build_status_rows([_track(2, 10.0), _track(1, 500.0)])
    assert [row["label"] for row in rows] == ["Person1", "Person2"]


def test_checking_rows_do_not_show_a_score():
    track = _track(1, 0.0)
    track.uniform = CHECKING
    track.uniform_score = 0.87
    row = build_status_rows([track])[0]
    assert row["uniform_score"] is None


def test_unconfirmed_track_is_not_judged():
    manager = TrackManager()
    _active, new = manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.0)
    assert needs_classification(new[0], now=1.0) is False


def test_no_check_before_the_track_settles():
    manager = TrackManager()
    manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.0)
    manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.1)
    active, _new = manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.2)

    early = active[0].first_seen + C.TRACK_SETTLE_SECONDS / 2
    settled = active[0].first_seen + C.TRACK_SETTLE_SECONDS + 0.01

    assert needs_classification(active[0], now=early) is False
    assert needs_classification(active[0], now=settled) is True


def test_first_two_checks_are_taken_close_together():
    track = _track(1, 0.0)
    track.checks = 0  # nothing measured yet
    track.last_classified = 10.0

    assert needs_classification(track, now=10.0 + C.WARMUP_INTERVAL / 2) is False
    assert needs_classification(track, now=10.0 + C.WARMUP_INTERVAL + 0.01) is True


def test_published_score_waits_for_two_samples():
    assert smoothed_score([]) is None
    assert smoothed_score([0.59]) is None
    assert smoothed_score([0.10, 0.30]) == 0.20


def test_median_ignores_one_wild_sample():
    # Exactly the live log: 0.15, 0.59, 0.30 must settle at 0.30, not flip.
    assert smoothed_score([0.15, 0.59, 0.30]) == 0.30
    assert smoothed_score([0.90, 0.10, 0.92]) == 0.90
    # Window is 5: four samples keep all four, median = mean of the middle two.
    assert smoothed_score([0.1, 0.2, 0.3, 0.95]) == 0.25
    assert smoothed_score([0.1, 0.2, 0.3, 0.5, 0.95, 0.99]) == 0.5  # last 5


def test_fresh_verdict_is_not_rechecked_immediately():
    track = _track(1, 0.0)
    track.last_classified = 10.0
    track.classified_box = track.box
    assert needs_classification(track, now=10.5) is False
    assert needs_classification(track, now=10.0 + C.RECLASSIFY_INTERVAL + 0.1) is True


def test_small_movement_keeps_the_cached_verdict():
    track = _track(1, 0.0)
    track.last_classified = 10.0
    track.classified_box = track.box
    track.box = (20.0, 20.0, 120.0, 220.0)  # 20 px of 200 px height = 10%
    assert moved_a_lot(track) is False
    assert needs_classification(track, now=10.5) is False


def test_large_movement_triggers_a_recheck():
    track = _track(1, 0.0)
    track.last_classified = 10.0
    track.classified_box = track.box
    track.box = (0.0, 60.0, 100.0, 260.0)  # centre moved 60 px of 200 px = 30%
    assert moved_a_lot(track) is True
    assert needs_classification(track, now=10.5) is True

def test_growing_box_triggers_a_recheck():
    track = _track(1, 0.0)
    track.last_classified = 10.0
    track.classified_box = track.box  # 200 px tall
    track.box = (0.0, 0.0, 100.0, 260.0)  # same centre, 30% taller
    assert moved_a_lot(track) is True
    assert needs_classification(track, now=10.5) is True
