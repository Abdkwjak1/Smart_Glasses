"""Camera / video-file capture."""

from __future__ import annotations

import sys

import cv2


def open_source(source: int | str = 0) -> cv2.VideoCapture:
    """Open a webcam index or a video file. DirectShow is Windows-only, so use it only there."""
    if isinstance(source, int) and sys.platform.startswith("win"):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)
