"""Detector wiring (with a stub network), prompt resolution, camera and CLI plumbing."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from smart_glasses import cli
from smart_glasses.audio import Prompts
from smart_glasses.camera import open_source
from smart_glasses.config import DEFAULT
from smart_glasses.detection import ObjectDetector

from .conftest import CLASSES, SAMPLE_VIDEO, FakeSpeaker, yolo_row


class StubNet:
    def __init__(self, outputs):
        self.outputs, self.blob = outputs, None

    def setInput(self, blob):
        self.blob = blob

    def forward(self, layers):
        return self.outputs


def test_detector_end_to_end_with_stub_network():
    det = ObjectDetector()
    det.classes = CLASSES
    rows = np.array(
        [
            yolo_row(0.25, 0.5, 0.2, 0.3, "dog", 0.95),
            yolo_row(0.26, 0.5, 0.2, 0.3, "dog", 0.80),  # near-duplicate, removed by NMS
            yolo_row(0.75, 0.5, 0.2, 0.3, "car", 0.90),
        ]
    )
    det._net, det._out_layers = StubNet([rows]), ["out"]
    found = det.detect(np.zeros((480, 640, 3), np.uint8))
    assert sorted(d.label for d in found) == ["car", "dog"]
    assert det._net.blob.shape == (1, 3, DEFAULT.input_size, DEFAULT.input_size)


def test_detector_gives_actionable_error_when_model_files_are_missing(tmp_path):
    det = ObjectDetector(replace(DEFAULT, model_dir=tmp_path))
    with pytest.raises(FileNotFoundError, match="download_models.py"):
        det.load()


def test_coco_class_file_loads_with_multiword_names():
    from pathlib import Path

    names = Path(DEFAULT.classes_path).read_text().splitlines()
    assert "baseball bat" in names and len(names) == 80


def test_prompts_skip_missing_files_instead_of_crashing(tmp_path):
    speaker = FakeSpeaker()
    Prompts(speaker, replace(DEFAULT, audio_dir=tmp_path)).say("objects", "dog")
    assert speaker.played == []


def test_announce_object_plays_name_then_direction():
    speaker = FakeSpeaker()
    Prompts(speaker).announce_object("baseball bat", "left")
    assert speaker.names == ["objects/baseball bat", "directions/left"]


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="sample video missing")
def test_open_source_reads_video_files():
    cap = open_source(str(SAMPLE_VIDEO))
    assert cap.isOpened()
    cap.release()


def test_cli_parser_defaults_and_flags():
    args = cli.build_parser().parse_args([])
    assert args.camera == "0" and not args.headless
    args = cli.build_parser().parse_args(["--camera", "clip.mp4", "--headless", "-v"])
    assert (args.camera, args.headless, args.verbose) == ("clip.mp4", True, True)


def test_opencv_contrib_face_module_is_available():
    """The environment must ship opencv-contrib; plain opencv-python has no cv2.face."""
    assert hasattr(cv2, "face")
