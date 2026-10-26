"""Run the pipeline on the real sample video (no model files needed: the network is simulated)."""

import cv2
import numpy as np
import pytest

from smart_glasses import bench, hud
from smart_glasses.detection import Announcer

from .conftest import CLASSES, SAMPLE_VIDEO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="sample video missing"),
]


def test_sample_video_is_readable():
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    assert cap.isOpened()
    assert (int(cap.get(3)), int(cap.get(4))) == (1280, 720)
    assert cap.get(7) > 100
    cap.release()


def test_pipeline_over_real_frames_produces_valid_hud_output_and_throttled_speech():
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    now = [0.0]
    announcer = Announcer(3.0, clock=lambda: now[0])
    spoken = frames = 0
    for i in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        now[0] = i / 30  # video timeline, 30 fps
        h, w = frame.shape[:2]
        outputs = bench.synthetic_yolo_outputs(CLASSES, seed=i)
        kept, top, ann = bench.postprocess(outputs, w, h, CLASSES)
        view = hud.render(frame, kept, top, "OBJECTS")
        assert view.shape == (h + hud.BAR_HEIGHT, 2 * w, 3) and view.dtype == np.uint8
        assert len(top) <= 3
        if announcer.ready():
            spoken += 1
        frames += 1
    cap.release()
    assert frames == 60
    assert spoken == 1  # 2 s of video at a 3 s interval -> exactly one announcement round
