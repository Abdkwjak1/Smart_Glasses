"""Render docs/img/hud_preview.jpg: the HUD over a real frame of the sample video.

The detections are ILLUSTRATIVE (hand-placed), because the preview must not depend on the
~240 MB model files; it documents the layout of the display, not detector accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smart_glasses import hud  # noqa: E402
from smart_glasses.detection import Detection  # noqa: E402


def main() -> None:
    cap = cv2.VideoCapture(str(ROOT / "assets" / "demo" / "sample.mov"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 120)
    ok, frame = cap.read()
    cap.release()
    assert ok, "sample video unreadable"
    frame = cv2.resize(frame, (640, 360))
    dets = [
        Detection("car", 0.96, (25, 185, 345, 175)),
        Detection("person", 0.88, (594, 182, 40, 72)),
        Detection("car", 0.71, (365, 190, 100, 70)),
    ]
    view = hud.render(frame, dets, dets[:2], "OBJECTS  -  Esc to exit   (illustrative detections)")
    out = ROOT / "docs" / "img" / "hud_preview.jpg"
    cv2.imwrite(str(out), view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(out)


if __name__ == "__main__":
    main()
