"""Central configuration: paths, thresholds and the Arabic voice vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Arabic voice commands understood at the top level and in the "people" sub-menu.
CMD_OBJECTS = "اغراض"
CMD_PEOPLE = "اشخاص"
CMD_TEXT = "نصوص"
CMD_EXIT = "0"
CMD_NEW = "جديد"
CMD_KNOWN = "قديم"

# Keywords the OCR mode listens for; the index selects assets/audio/ocr/<index>.mp3.
OCR_KEYWORDS = ("عيادة", "صيدلية", "بقالية", "مخبر", "مستشفى", "مشفى", "مركز", "مكتبة", "ماركت")


@dataclass(frozen=True)
class Config:
    # --- paths ---------------------------------------------------------------------------
    assets_dir: Path = ROOT / "assets"
    audio_dir: Path = ROOT / "assets" / "audio"
    model_dir: Path = ROOT / "model_data"
    data_dir: Path = ROOT / "data"

    # --- object detection (YOLOv3) -------------------------------------------------------
    input_size: int = 320
    scale_factor: float = 1 / 255.0
    conf_threshold: float = 0.3
    nms_conf_threshold: float = 0.5
    nms_iou_threshold: float = 0.4
    max_announced: int = 3
    announce_interval_s: float = 3.0

    # --- face recognition (LBPH) ---------------------------------------------------------
    face_samples: int = 20
    face_vote_frames: int = 20
    face_timeout_s: float = 20.0
    # LBPH "confidence" is a distance: lower is better. Above this the person is unknown.
    face_max_distance: float = 90.0

    # --- speech ----------------------------------------------------------------------------
    speech_language: str = "ar-EG"
    ocr_keywords: tuple[str, ...] = field(default=OCR_KEYWORDS)

    @property
    def weights_path(self) -> Path:
        return self.model_dir / "yolov3.weights"

    @property
    def cfg_path(self) -> Path:
        return self.model_dir / "yolov3.cfg"

    @property
    def classes_path(self) -> Path:
        return self.assets_dir / "coco_classes.txt"

    @property
    def people_csv(self) -> Path:
        return self.data_dir / "people.csv"

    @property
    def names_dir(self) -> Path:
        return self.data_dir / "names"

    @property
    def face_dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    @property
    def face_model_path(self) -> Path:
        return self.data_dir / "recognizer.yml"


DEFAULT = Config()
