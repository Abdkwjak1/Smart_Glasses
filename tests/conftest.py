from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from smart_glasses.audio import Prompts
from smart_glasses.config import DEFAULT, Config

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = ROOT / "assets" / "demo" / "sample.mov"
CLASSES = (ROOT / "assets" / "coco_classes.txt").read_text().splitlines()


def yolo_row(cx, cy, w, h, class_name, conf=0.9, n_classes=80):
    """One raw YOLO output row: [cx, cy, w, h, objectness, *class scores] (normalised)."""
    row = np.zeros(5 + n_classes, np.float32)
    row[:4] = (cx, cy, w, h)
    row[4] = 1.0
    row[5 + CLASSES.index(class_name)] = conf
    return row


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.reads = 0

    def read(self):
        self.reads += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        pass


class FakeSpeaker:
    def __init__(self):
        self.played: list[Path] = []

    def play(self, path):
        self.played.append(Path(path))

    @property
    def names(self):
        """Played files as 'group/name' for readable assertions."""
        return [f"{p.parent.name}/{p.stem}" for p in self.played]


class FakeListener:
    def __init__(self, *utterances):
        self._it = iter(utterances)

    def listen(self):
        return next(self._it, "0")  # fall back to "exit" so a bad script can't hang the suite


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Real audio assets, but private (temporary) user data."""
    return replace(DEFAULT, data_dir=tmp_path / "data", face_timeout_s=2.0)


@pytest.fixture
def speaker():
    return FakeSpeaker()


@pytest.fixture
def prompts(speaker, cfg):
    return Prompts(speaker, cfg)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), np.uint8)
