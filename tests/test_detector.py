# Author: David Li, SISHK
"""YOLO output decoding - tested with synthetic tensors, no model file needed."""

import numpy as np

from models.person_detector import decode_yolo


def _yolo_output(anchors, num_classes=80):
    """(1, 4 + nc, anchors) in ultralytics layout, all class scores zero."""
    return np.zeros((1, 4 + num_classes, anchors), dtype=np.float32)


def test_single_person_is_decoded_and_scaled_back():
    output = _yolo_output(1)
    output[0, 0, 0] = 320.0  # cx
    output[0, 1, 0] = 320.0  # cy
    output[0, 2, 0] = 100.0  # w
    output[0, 3, 0] = 100.0  # h
    output[0, 4, 0] = 0.9  # person score

    detections = decode_yolo(
        output, scale=1.0, pad_x=0.0, pad_y=0.0, original_size=(640, 640)
    )
    assert len(detections) == 1
    assert detections[0].box == (270.0, 270.0, 370.0, 370.0)
    assert abs(detections[0].score - 0.9) < 1e-6


def test_low_confidence_is_dropped():
    output = _yolo_output(1)
    output[0, 4, 0] = 0.1
    assert decode_yolo(output, 1.0, 0.0, 0.0, (640, 640)) == []


def test_non_person_classes_are_ignored():
    output = _yolo_output(1)
    output[0, 0, 0] = 320.0
    output[0, 1, 0] = 320.0
    output[0, 2, 0] = 100.0
    output[0, 3, 0] = 100.0
    output[0, 5, 0] = 0.99  # class 1, not "person"
    assert decode_yolo(output, 1.0, 0.0, 0.0, (640, 640)) == []


def test_transposed_output_layout_is_accepted():
    output = _yolo_output(1).transpose(0, 2, 1)  # (1, 1, 84)
    output[0, 0, 0] = 320.0
    output[0, 0, 1] = 320.0
    output[0, 0, 2] = 200.0
    output[0, 0, 3] = 200.0
    output[0, 0, 4] = 0.95
    detections = decode_yolo(output, 1.0, 0.0, 0.0, (640, 640))
    assert len(detections) == 1
    assert detections[0].box == (220.0, 220.0, 420.0, 420.0)


def test_duplicate_boxes_are_suppressed():
    output = _yolo_output(2)
    for anchor in range(2):
        output[0, 0, anchor] = 320.0
        output[0, 1, anchor] = 320.0
        output[0, 2, anchor] = 100.0
        output[0, 3, anchor] = 100.0
    output[0, 4, 0] = 0.9
    output[0, 4, 1] = 0.8
    detections = decode_yolo(output, 1.0, 0.0, 0.0, (640, 640))
    assert len(detections) == 1


def test_letterbox_padding_is_removed():
    output = _yolo_output(1)
    output[0, 0, 0] = 320.0  # centre of the 640 square
    output[0, 1, 0] = 320.0
    output[0, 2, 0] = 64.0
    output[0, 3, 0] = 64.0
    output[0, 4, 0] = 0.9

    # A 1280x720 frame: scale 0.5, 140 px of padding top and bottom.
    detections = decode_yolo(output, 0.5, 0.0, 140.0, (1280, 720))
    assert len(detections) == 1
    x1, y1, x2, y2 = detections[0].box
    assert abs(x1 - 576.0) < 1e-6
    assert abs(y1 - 296.0) < 1e-6
    assert abs(y2 - 424.0) < 1e-6