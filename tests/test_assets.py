"""The recorded Arabic prompts the app relies on must all exist."""

from smart_glasses.config import DEFAULT, OCR_KEYWORDS

from .conftest import CLASSES

AUDIO = DEFAULT.audio_dir


def test_every_coco_class_has_a_spoken_name():
    assert len(CLASSES) == 80
    missing = [c for c in CLASSES if not (AUDIO / "objects" / f"{c}.mp3").exists()]
    assert not missing, missing


def test_direction_prompts():
    assert (AUDIO / "directions" / "left.mp3").exists()
    assert (AUDIO / "directions" / "right.mp3").exists()


def test_one_ocr_prompt_per_keyword():
    for i in range(len(OCR_KEYWORDS)):
        assert (AUDIO / "ocr" / f"{i}.mp3").exists()


def test_command_prompts():
    for name in ("intro", "repeat", "end", "f_r", "f_r1", "f_r2", "f_r3", "r_fail"):
        assert (AUDIO / "commands" / f"{name}.mp3").exists(), name
