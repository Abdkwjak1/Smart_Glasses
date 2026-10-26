import numpy as np
import pytest

from smart_glasses.detection import (
    Announcer,
    Detection,
    largest,
    non_max_suppression,
    parse_outputs,
    side_of,
    to_announcements,
)

from .conftest import CLASSES, yolo_row

W, H = 640, 480


def test_parse_outputs_decodes_box_geometry():
    out = [np.array([yolo_row(0.5, 0.5, 0.2, 0.4, "dog")])]
    (d,) = parse_outputs(out, W, H, CLASSES, 0.3)
    assert d.label == "dog"
    assert d.box == (256, 144, 128, 192)  # centre-based -> top-left based
    assert d.center == (320, 240)
    assert d.confidence == pytest.approx(0.9)


def test_parse_outputs_filters_low_confidence_and_handles_many_layers():
    layer_a = np.array([yolo_row(0.2, 0.2, 0.1, 0.1, "cat", conf=0.2)])  # below threshold
    layer_b = np.array([yolo_row(0.8, 0.8, 0.1, 0.1, "car", conf=0.7)])
    found = parse_outputs([layer_a, layer_b], W, H, CLASSES, 0.3)
    assert [d.label for d in found] == ["car"]


def test_parse_outputs_empty():
    assert parse_outputs([], W, H, CLASSES, 0.3) == []
    assert parse_outputs([np.zeros((0, 85), np.float32)], W, H, CLASSES, 0.3) == []


def test_nms_removes_overlapping_duplicates_keeps_distinct():
    dup1 = Detection("dog", 0.9, (100, 100, 100, 100))
    dup2 = Detection("dog", 0.6, (105, 105, 100, 100))
    other = Detection("cat", 0.8, (400, 300, 80, 80))
    kept = non_max_suppression([dup1, dup2, other], 0.5, 0.4)
    assert set(kept) == {dup1, other}
    assert non_max_suppression([], 0.5, 0.4) == []


def test_largest_uses_real_area_not_the_old_buggy_formula():
    # The legacy code computed (w - x) * (h - y); a far-right object would get a negative area.
    small_left = Detection("cup", 0.9, (10, 10, 30, 30))
    big_right = Detection("tvmonitor", 0.9, (500, 300, 120, 160))
    mid = Detection("chair", 0.9, (200, 100, 60, 60))
    assert largest([small_left, big_right, mid], 2) == [big_right, mid]


def test_largest_handles_fewer_than_n():
    only = Detection("cup", 0.9, (0, 0, 5, 5))
    assert largest([only], 3) == [only]
    assert largest([], 3) == []


@pytest.mark.parametrize(
    "frame_w,x,expected",
    [(640, 100, "left"), (640, 400, "right"), (1280, 600, "left"), (1280, 700, "right")],
)
def test_side_scales_with_frame_width(frame_w, x, expected):
    # The legacy code hard-coded 320, wrong for any frame that isn't 640 wide.
    assert side_of(Detection("cup", 1, (x, 0, 10, 10)), frame_w) == expected


def test_to_announcements():
    ds = [Detection("cup", 1, (0, 0, 10, 10)), Detection("dog", 1, (600, 0, 10, 10))]
    assert to_announcements(ds, 640) == [("cup", "left"), ("dog", "right")]


def test_announcer_rate_limits():
    now = [0.0]
    a = Announcer(3.0, clock=lambda: now[0])
    assert a.ready()  # first call always speaks
    now[0] = 1.0
    assert not a.ready()
    now[0] = 3.0
    assert a.ready()
    now[0] = 4.0
    assert not a.ready()
