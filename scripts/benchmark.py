"""Print latency / FPS for the hot paths (and real YOLO inference if model_data/ is present).

python scripts/benchmark.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smart_glasses import bench  # noqa: E402
from smart_glasses.config import DEFAULT  # noqa: E402
from smart_glasses.detection import ObjectDetector  # noqa: E402

CLASSES = DEFAULT.classes_path.read_text().splitlines()
SAMPLE = ROOT / "assets" / "demo" / "sample.mov"


def first_frame() -> np.ndarray:
    cap = cv2.VideoCapture(str(SAMPLE))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else np.zeros((720, 1280, 3), np.uint8)


def main() -> None:
    frame = first_frame()
    results = bench.run_suite(frame, CLASSES)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results["Face detection (Haar, 1280x720)"] = bench.timeit(
        lambda: detector.detectMultiScale(gray, 1.3, 5), repeats=15
    )

    if DEFAULT.weights_path.exists():
        yolo = ObjectDetector()
        yolo.load()
        results[f"YOLO inference ({DEFAULT.weights_path.name}, {DEFAULT.input_size}px)"] = (
            bench.timeit(lambda: yolo.detect(frame), repeats=10, warmup=2)
        )
    else:
        print("(model_data/ not found - skipping real YOLO inference)\n")

    host = platform.processor() or platform.machine()
    print(f"Python {platform.python_version()} | OpenCV {cv2.__version__} | {host}")
    print(f"{'stage':<46}{'mean ms':>9}{'p50':>8}{'p95':>8}{'fps':>9}")
    for name, s in results.items():
        print(
            f"{name:<46}{s['mean_ms']:>9.2f}{s['p50_ms']:>8.2f}{s['p95_ms']:>8.2f}{s['fps']:>9.0f}"
        )


if __name__ == "__main__":
    main()
