"""Download the YOLOv3 weights + config (~240 MB) into model_data/.

The COCO class list ships with the repo (assets/coco_classes.txt).
If a host refuses the download, fetch the two files by hand into model_data/.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
}
# --tiny: 8x smaller and much faster, less accurate - good for laptops and quick demos.
TINY = {
    "yolov3.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
    "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
}


def main() -> int:
    files = TINY if "--tiny" in sys.argv else FILES
    out = ROOT / "model_data"
    out.mkdir(exist_ok=True)
    for name, url in files.items():
        dest = out / name
        if dest.exists():
            print(f"skip {name} (exists)")
            continue
        print(f"downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except urllib.error.URLError as exc:
            print(f"  failed: {exc}\n  download {url} manually and save it as {dest}")
            dest.unlink(missing_ok=True)
            return 1
    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
