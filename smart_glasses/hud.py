"""Heads-up display: renders what the glasses "see" for sighted helpers and debugging.

Pure functions on numpy images (no windows), so the output can be asserted in tests and
written to disk for the README. Text is ASCII because OpenCV's Hershey fonts have no Arabic.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection, side_of

GREEN = (80, 220, 100)
AMBER = (0, 190, 255)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)
BAR_HEIGHT = 28


def _color_for(label: str) -> tuple[int, int, int]:
    h = sum(map(ord, label))  # deterministic (unlike hash()) so colours are stable across runs
    return (60 + h * 37 % 180, 60 + h * 71 % 180, 60 + h * 113 % 180)


def draw_detections(
    frame: np.ndarray, detections: list[Detection], announced: list[Detection] | None = None
) -> np.ndarray:
    """Boxes + labels; announced (spoken) detections get a thick amber outline."""
    out = frame.copy()
    spoken = set(announced or [])
    for d in detections:
        x, y, w, h = d.box
        is_spoken = d in spoken
        color = AMBER if is_spoken else _color_for(d.label)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 4 if is_spoken else 2)
        cv2.putText(
            out,
            f"{d.label} {d.confidence:.0%}",
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_radar(shape: tuple[int, int], detections: list[Detection]) -> np.ndarray:
    """Top-down 'where is it' panel: left/right divider and one dot per detection."""
    h, w = shape
    panel = np.zeros((h, w, 3), np.uint8)
    cv2.line(panel, (w // 2, 0), (w // 2, h), GREY, 1)
    cv2.putText(panel, "LEFT", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
    cv2.putText(panel, "RIGHT", (w - 62, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1, cv2.LINE_AA)
    for d in detections:
        cx, cy = d.center
        cv2.circle(panel, (cx, cy), 10, WHITE, -1)
        text = f"{d.label} ({side_of(d, w)})"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = cx + 14 if cx + 14 + tw < w else cx - 14 - tw  # flip to the left near the edge
        cv2.putText(
            panel, text, (max(tx, 2), cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA
        )
    return panel


def render(
    frame: np.ndarray, detections: list[Detection], announced: list[Detection], status: str
) -> np.ndarray:
    """Camera view with overlays next to the radar panel, under a one-line status bar."""
    h, w = frame.shape[:2]
    view = np.hstack([draw_detections(frame, detections, announced), draw_radar((h, w), announced)])
    bar = np.full((BAR_HEIGHT, view.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
    return np.vstack([bar, view])
