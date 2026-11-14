"""Geometry is the most reused logic in the project, so it carries the tests."""

import numpy as np

from models.geometry import (
    area_ratio,
    clip_box,
    crop,
    eye_box_from_face,
    iou,
    letterbox,
    nms,
    sub_box,
    unletterbox_box,
)


def test_iou_identical_and_disjoint():
    box = (0.0, 0.0, 10.0, 10.0)
    assert iou(box, box) == 1.0
    assert iou(box, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_half_overlap():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    assert abs(iou(a, b) - 1 / 3) < 1e-6


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
    kept = nms(boxes, [0.9, 0.8, 0.7], threshold=0.5)
    assert kept == [0, 2]


def test_clip_box_clamps_and_orders():
    assert clip_box((-5, -5, 20, 20), 10, 10) == (0.0, 0.0, 10.0, 10.0)


def test_area_ratio_matches_plan_rule():
    # A person box covering a quarter of the frame clears the 15% rule.
    assert area_ratio((0, 0, 640, 360), 1280, 720) == 0.25
    assert area_ratio((0, 0, 64, 36), 1280, 720) < 0.15


def test_letterbox_roundtrip():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    padded, scale, pad_x, pad_y = letterbox(frame, 640)
    assert padded.shape == (640, 640, 3)
    assert abs(scale - 0.5) < 1e-6
    assert pad_x == 0.0
    assert abs(pad_y - 140.0) < 1.0

    original = (100.0, 200.0, 300.0, 400.0)
    in_letterbox = (
        original[0] * scale + pad_x,
        original[1] * scale + pad_y,
        original[2] * scale + pad_x,
        original[3] * scale + pad_y,
    )
    back = unletterbox_box(in_letterbox, scale, pad_x, pad_y)
    assert all(abs(a - b) < 1e-6 for a, b in zip(back, original))


def test_torso_is_the_middle_of_the_person_box():
    torso = sub_box((0, 0, 100, 200), top=0.2, bottom=0.7, left=0.2, right=0.8)
    assert torso == (20.0, 40.0, 80.0, 140.0)


def test_eye_box_from_yunet_landmarks():
    face = np.array([100, 100, 80, 80, 120, 130, 160, 130, 140, 160, 120, 180, 160, 180, 0.9])
    box = eye_box_from_face(face, padding=0.5)
    assert box is not None
    x1, y1, x2, y2 = box
    assert x1 < 120 and x2 > 160  # covers both eyes with padding
    assert y1 < 130 < y2


def test_eye_box_rejects_degenerate_face():
    face = np.array([100, 100, 80, 80, 120, 130, 120, 130, 0, 0, 0, 0, 0, 0, 0.9])
    assert eye_box_from_face(face) is None


def test_crop_is_bounded_by_the_image():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop(image, (-50, -50, 50, 50)).shape == (50, 50, 3)
    assert crop(image, (200, 200, 300, 300)).size == 0