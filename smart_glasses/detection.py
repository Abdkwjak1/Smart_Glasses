"""Object-detection post-processing: YOLO output -> detections -> spoken announcements.

Everything here is pure numpy/OpenCV-math and works without a camera or model file, which is
what makes the core logic unit-testable and benchmarkable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT, Config


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h (top-left origin)

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.box
        return x + w // 2, y + h // 2


def side_of(detection: Detection, frame_width: int) -> str:
    """'left' or 'right' of the frame centre (matches the recorded audio prompts)."""
    return "left" if detection.center[0] < frame_width / 2 else "right"


def parse_outputs(
    outputs, frame_w: int, frame_h: int, classes: list[str], conf_threshold: float
) -> list[Detection]:
    """Decode raw YOLO layer outputs (rows of cx, cy, w, h, obj, class scores...)."""
    rows = [np.asarray(o, dtype=np.float32).reshape(-1, np.asarray(o).shape[-1]) for o in outputs]
    if not rows:
        return []
    data = np.vstack(rows)
    scores = data[:, 5:]
    class_ids = scores.argmax(axis=1)
    confs = scores[np.arange(len(scores)), class_ids]
    keep = confs > conf_threshold
    detections = []
    for (cx, cy, w, h, *_), cid, conf in zip(data[keep], class_ids[keep], confs[keep]):
        bw, bh = int(w * frame_w), int(h * frame_h)
        x, y = int(cx * frame_w - bw / 2), int(cy * frame_h - bh / 2)
        detections.append(Detection(classes[cid], float(conf), (x, y, bw, bh)))
    return detections


def non_max_suppression(
    detections: list[Detection], conf_threshold: float, iou_threshold: float
) -> list[Detection]:
    if not detections:
        return []
    boxes = [list(d.box) for d in detections]
    confs = [d.confidence for d in detections]
    idx = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, iou_threshold)
    return [detections[i] for i in np.array(idx).flatten()]


def largest(detections: list[Detection], n: int) -> list[Detection]:
    """The n biggest detections (closest objects tend to look biggest), biggest first."""
    return sorted(detections, key=lambda d: d.area, reverse=True)[:n]


def to_announcements(detections: list[Detection], frame_w: int) -> list[tuple[str, str]]:
    return [(d.label, side_of(d, frame_w)) for d in detections]


class Announcer:
    """Rate-limits spoken announcements so the user isn't flooded with audio."""

    def __init__(self, interval_s: float, clock=time.monotonic):
        self.interval_s = interval_s
        self._clock = clock
        self._last: float | None = None

    def ready(self) -> bool:
        now = self._clock()
        if self._last is None or now - self._last >= self.interval_s:
            self._last = now
            return True
        return False


class ObjectDetector:
    """YOLOv3 (OpenCV DNN) wrapper. Weights are loaded lazily on first use."""

    def __init__(self, config: Config = DEFAULT):
        self.config = config
        self._net = None
        self._out_layers: list[str] = []
        self.classes: list[str] = []

    def load(self) -> None:
        c = self.config
        missing = [p for p in (c.weights_path, c.cfg_path, c.classes_path) if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing YOLO files: {', '.join(map(str, missing))}. "
                "Run `python scripts/download_models.py`."
            )
        self._net = cv2.dnn.readNet(str(c.weights_path), str(c.cfg_path))
        self._out_layers = list(self._net.getUnconnectedOutLayersNames())
        # coco names may contain spaces ("baseball bat"), so parse line-wise
        self.classes = [
            line.strip()
            for line in Path(c.classes_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._net is None:
            self.load()
        c = self.config
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, c.scale_factor, (c.input_size, c.input_size), (0, 0, 0), swapRB=True, crop=False
        )
        self._net.setInput(blob)
        outputs = self._net.forward(self._out_layers)
        found = parse_outputs(outputs, w, h, self.classes, c.conf_threshold)
        return non_max_suppression(found, c.nms_conf_threshold, c.nms_iou_threshold)
