# Performance

Run it yourself:

```bash
python scripts/benchmark.py
```

Numbers from the development machine (x86_64 Linux, Python 3.10, OpenCV 4.10, CPU only):

| Stage | mean | p95 | fps |
|---|---:|---:|---:|
| YOLO post-processing (decode 6,300 boxes, NMS, rank) | 1.4 ms | 1.5 ms | ~700 |
| HUD render (1280×720 plus radar and status bar) | 2.9 ms | 3.0 ms | ~340 |
| Face detection (Haar, 1280×720) | 9.8 ms | 10.2 ms | ~100 |

The YOLO forward pass is **not** included. The model files weren't available when these were measured. It is the expensive part, and it depends on your hardware and on `yolov3` versus `--tiny`. The script measures it automatically once `model_data/` exists.

The performance tests in `tests/test_performance.py` enforce limits with about 10× headroom, so they catch real regressions without failing on a slow machine.
