"""Micro-benchmarks for the hot paths (used by the perf tests and scripts/benchmark.py)."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import numpy as np

from . import hud
from .config import DEFAULT
from .detection import (
    largest,
    non_max_suppression,
    parse_outputs,
    to_announcements,
)


def timeit(fn: Callable[[], object], repeats: int = 30, warmup: int = 3) -> dict[str, float]:
    """Wall-clock stats in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1000)
    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "p50_ms": samples[len(samples) // 2],
        "p95_ms": samples[int(len(samples) * 0.95) - 1],
        "fps": 1000 / statistics.fmean(samples),
    }


def synthetic_yolo_outputs(
    classes: list[str], input_size: int = 320, seed: int = 0, hits: int = 40
):
    """Raw outputs with the exact shapes of YOLOv3 at `input_size` (3 scales x 3 anchors)."""
    rng = np.random.default_rng(seed)
    outputs = []
    for stride in (32, 16, 8):
        n = (input_size // stride) ** 2 * 3
        out = np.zeros((n, 5 + len(classes)), np.float32)
        out[:, :4] = rng.random((n, 4)) * [1, 1, 0.3, 0.3]
        out[:, 5:] = rng.random((n, len(classes))) * 0.2  # background noise below threshold
        outputs.append(out)
    flat = np.vstack(outputs)
    for i in rng.choice(len(flat), hits, replace=False):  # confident detections
        flat[i, 5 + rng.integers(len(classes))] = 0.5 + rng.random() * 0.5
    start = 0
    for i, o in enumerate(outputs):
        outputs[i] = flat[start : start + len(o)]
        start += len(o)
    return outputs


def postprocess(outputs, frame_w, frame_h, classes):
    c = DEFAULT
    found = parse_outputs(outputs, frame_w, frame_h, classes, c.conf_threshold)
    kept = non_max_suppression(found, c.nms_conf_threshold, c.nms_iou_threshold)
    top = largest(kept, c.max_announced)
    return kept, top, to_announcements(top, frame_w)


def run_suite(frame: np.ndarray, classes: list[str]) -> dict[str, dict[str, float]]:
    h, w = frame.shape[:2]
    outputs = synthetic_yolo_outputs(classes)
    kept, top, _ = postprocess(outputs, w, h, classes)
    return {
        "YOLO post-processing (parse+NMS+rank)": timeit(
            lambda: postprocess(outputs, w, h, classes)
        ),
        "HUD render": timeit(lambda: hud.render(frame, kept, top, "OBJECTS")),
    }
