"""Tracking decides when a person is 'new', which is what triggers a check."""

from models import constants as C
from models.tracks import TrackManager, torso_of


def test_first_detection_creates_a_new_track():
    manager = TrackManager()
    active, new = manager.update([(10.0, 10.0, 110.0, 210.0)], now=1.0)
    assert len(new) == 1
    assert len(active) == 1
    assert new[0].label_index == 1
    assert new[0].torso_box == torso_of((10.0, 10.0, 110.0, 210.0))


def test_same_person_is_matched_not_recreated():
    manager = TrackManager()
    manager.update([(10.0, 10.0, 110.0, 210.0)], now=1.0)
    active, new = manager.update([(12.0, 12.0, 112.0, 212.0)], now=1.1)
    assert new == []
    assert len(active) == 1
    assert active[0].track_id == 1


def test_second_person_gets_the_next_number():
    manager = TrackManager()
    manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.0)
    active, new = manager.update(
        [(0.0, 0.0, 100.0, 200.0), (400.0, 0.0, 500.0, 200.0)], now=1.1
    )
    assert len(active) == 2
    assert len(new) == 1
    assert new[0].label_index == 2


def test_label_number_is_reused_after_a_person_leaves():
    manager = TrackManager()
    manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.0)
    for step in range(C.TRACK_MAX_MISSES + 1):
        manager.update([], now=1.1 + step)
    assert manager.tracks == []

    _active, new = manager.update([(300.0, 0.0, 400.0, 200.0)], now=9.0)
    assert [t.label_index for t in new] == [1]
    assert new[0].track_id != 1  # internal id keeps growing, the label does not


def test_track_disappears_after_max_misses():
    manager = TrackManager(max_misses=2)
    manager.update([(0.0, 0.0, 100.0, 200.0)], now=1.0)
    manager.update([], now=1.1)
    manager.update([], now=1.2)
    assert len(manager.tracks) == 1
    manager.update([], now=1.3)
    assert manager.tracks == []


def test_two_people_do_not_swap_boxes():
    manager = TrackManager()
    manager.update([(0.0, 0.0, 100.0, 200.0), (400.0, 0.0, 500.0, 200.0)], now=1.0)
    active, _new = manager.update(
        [(400.0, 0.0, 500.0, 200.0), (0.0, 0.0, 100.0, 200.0)], now=1.1
    )
    by_id = {t.track_id: t for t in active}
    assert by_id[1].box[0] == 0.0
    assert by_id[2].box[0] == 400.0