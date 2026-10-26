"""Performance budgets. Thresholds are deliberately loose (~10x headroom) so they catch real
regressions, not noisy CI machines. `python scripts/benchmark.py` prints the actual numbers."""

import cv2
import numpy as np
import pytest

from smart_glasses import bench, hud
from smart_glasses.detection import ObjectDetector

from .conftest import CLASSES, ROOT, SAMPLE_VIDEO

pytestmark = pytest.mark.perf


@pytest.fixture(scope="module")
def frame():
    return np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_postprocessing_budget(frame):
    outputs = bench.synthetic_yolo_outputs(CLASSES)
    assert sum(len(o) for o in outputs) == 6300  # YOLOv3 @320: (10²+20²+40²)*3 candidate boxes
    stats = bench.timeit(lambda: bench.postprocess(outputs, 1280, 720, CLASSES), repeats=20)
    assert stats["p95_ms"] < 50, stats  # must leave nearly the whole frame budget to the network


def test_hud_render_budget(frame):
    kept, top, _ = bench.postprocess(bench.synthetic_yolo_outputs(CLASSES), 1280, 720, CLASSES)
    stats = bench.timeit(lambda: hud.render(frame, kept, top, "OBJECTS"), repeats=20)
    assert stats["p95_ms"] < 60, stats


def test_postprocessing_scales_linearly_not_quadratically():
    small = bench.synthetic_yolo_outputs(CLASSES, input_size=320)
    large = bench.synthetic_yolo_outputs(CLASSES, input_size=640)  # 4x the candidate boxes
    t_small = bench.timeit(lambda: bench.postprocess(small, 640, 480, CLASSES), 15)["mean_ms"]
    t_large = bench.timeit(lambda: bench.postprocess(large, 640, 480, CLASSES), 15)["mean_ms"]
    assert t_large < t_small * 12


def test_registry_lookup_is_fast_with_thousands_of_people(tmp_path):
    from smart_glasses.registry import PeopleRegistry

    reg = PeopleRegistry(tmp_path / "p.csv")
    reg.path.write_text("Name,ID\n" + "\n".join(f"n{i},{i}" for i in range(1, 5001)), "utf-8-sig")
    assert bench.timeit(lambda: reg.get(5000), repeats=10)["p95_ms"] < 100


@pytest.mark.integration
@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="sample video missing")
def test_haar_face_detection_on_real_video_frames():
    cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
    frames = []
    for _ in range(15):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    assert frames
    det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    it = iter(frames * 5)
    stats = bench.timeit(lambda: det.detectMultiScale(next(it), 1.3, 5), repeats=15, warmup=0)
    assert stats["p95_ms"] < 500, stats


@pytest.mark.integration
@pytest.mark.skipif(
    not (ROOT / "model_data" / "yolov3.weights").exists(), reason="run scripts/download_models.py"
)
def test_real_yolo_inference_latency():
    det = ObjectDetector()
    det.load()
    frame = np.random.default_rng(2).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    stats = bench.timeit(lambda: det.detect(frame), repeats=5, warmup=1)
    assert stats["p50_ms"] < 5000, stats
