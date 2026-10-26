"""UI tests for the heads-up display: assert on actual rendered pixels."""

import numpy as np
import pytest

from smart_glasses import hud
from smart_glasses.detection import Detection

pytestmark = pytest.mark.ui

W, H = 640, 480


def px(img, x, y):
    return tuple(int(v) for v in img[y, x])


def test_render_layout_is_camera_plus_radar_under_status_bar(blank_frame):
    out = hud.render(blank_frame, [], [], "OBJECTS")
    assert out.shape == (H + hud.BAR_HEIGHT, 2 * W, 3)
    assert out.dtype == np.uint8


def test_render_does_not_mutate_the_input_frame(blank_frame):
    before = blank_frame.copy()
    hud.render(blank_frame, [Detection("cup", 0.9, (50, 50, 80, 80))], [], "x")
    assert np.array_equal(blank_frame, before)


def test_announced_detection_gets_the_amber_highlight(blank_frame):
    d = Detection("dog", 0.8, (100, 100, 200, 150))
    out = hud.draw_detections(blank_frame, [d], announced=[d])
    assert px(out, 100, 200) == hud.AMBER  # left edge of the box
    plain = hud.draw_detections(blank_frame, [d])
    assert px(plain, 100, 200) != hud.AMBER


def test_non_announced_boxes_are_drawn_too(blank_frame):
    d = Detection("dog", 0.8, (100, 100, 200, 150))
    out = hud.draw_detections(blank_frame, [d])
    assert out.sum() > 0 and px(out, 100, 200) != (0, 0, 0)


def test_box_colour_is_stable_per_label():
    assert hud._color_for("dog") == hud._color_for("dog")
    assert hud._color_for("dog") != hud._color_for("cat")


def test_radar_places_dots_on_the_correct_half():
    left = Detection("cup", 0.9, (100, 200, 40, 40))  # centre x=120
    right = Detection("dog", 0.9, (500, 200, 40, 40))  # centre x=520
    panel = hud.draw_radar((H, W), [left, right])
    assert px(panel, 120, 220) == (255, 255, 255)
    assert px(panel, 520, 220) == (255, 255, 255)
    assert px(panel, W // 2, 400) == hud.GREY  # divider


def test_radar_empty_still_shows_divider_and_labels():
    panel = hud.draw_radar((H, W), [])
    assert panel[:, W // 2].any() and panel[:30, :80].any()


def test_status_bar_text_is_drawn(blank_frame):
    with_text = hud.render(blank_frame, [], [], "OBJECTS - Esc to exit")
    without = hud.render(blank_frame, [], [], "")
    assert not np.array_equal(with_text[: hud.BAR_HEIGHT], without[: hud.BAR_HEIGHT])


def test_boxes_partially_outside_the_frame_do_not_crash(blank_frame):
    d = Detection("dog", 0.8, (-30, -20, 100, 100))
    hud.render(blank_frame, [d], [d], "edge")
