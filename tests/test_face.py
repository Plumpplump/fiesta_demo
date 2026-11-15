# Author: David Li, SISHK


"""Face detection helpers: YuNet row handling and face/person matching."""

import numpy as np


def test_yunet_row_is_converted_to_a_corner_box():
    from models.face import face_box

    row = np.array([100, 50, 60, 80, 120, 90, 150, 90, 0, 0, 0, 0, 0, 0, 0.9])
    assert face_box(row) == (100.0, 50.0, 160.0, 130.0)


def _row(x, y, w, h):
    return np.array([x, y, w, h, x + w / 3, y + h / 3, x + 2 * w / 3, y + h / 3, 0, 0, 0, 0, 0, 0, 0.9])


def test_pick_face_takes_the_largest_face_inside_the_person_box():
    from models.face import pick_face

    person = (0.0, 0.0, 400.0, 700.0)
    small = _row(40.0, 60.0, 40.0, 40.0)
    large = _row(230.0, 80.0, 120.0, 120.0)
    assert pick_face([small, large], person) is large


def test_pick_face_ignores_a_face_outside_the_person_box():
    from models.face import pick_face

    person = (0.0, 0.0, 200.0, 600.0)
    neighbour = _row(500.0, 50.0, 120.0, 120.0)  # another person's head
    assert pick_face([neighbour], person) is None


def test_pick_face_allows_a_small_margin_at_the_edge():
    from models.face import pick_face

    person = (100.0, 0.0, 300.0, 600.0)
    edge = _row(95.0, 40.0, 100.0, 100.0)  # centre just left of the box (2.5%)
    assert pick_face([edge], person) is edge
