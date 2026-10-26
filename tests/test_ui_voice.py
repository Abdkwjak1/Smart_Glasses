"""UI tests for the spoken Arabic interface: scripted microphone in, recorded audio out."""

import numpy as np
import pytest

from smart_glasses.assistant import Assistant
from smart_glasses.config import CMD_KNOWN, CMD_NEW, CMD_OBJECTS, CMD_PEOPLE, CMD_TEXT
from smart_glasses.detection import Announcer, Detection
from smart_glasses.display import KEY_ESC, KEY_SPACE, ScriptedDisplay
from smart_glasses.registry import PeopleRegistry

from .conftest import FakeCamera, FakeListener

pytestmark = pytest.mark.ui


class FakeDetector:
    def __init__(self, per_frame):
        self.per_frame = list(per_frame)

    def detect(self, frame):
        return self.per_frame.pop(0) if self.per_frame else []


class FakeFaces:
    def __init__(self, samples=20, identified=None):
        self.samples, self.identified = samples, identified
        self.calls = []

    def register(self, pid):
        self.calls.append(("register", pid))
        return self.samples

    def train(self):
        self.calls.append(("train",))
        return self.samples

    def identify(self):
        self.calls.append(("identify",))
        return self.identified


def build(
    cfg,
    prompts,
    *,
    utterances,
    frames=(),
    keys=(),
    detector=None,
    faces=None,
    ocr_reader=None,
    synth=None,
):
    frames = list(frames) or [np.zeros((480, 640, 3), np.uint8)] * 5
    display = ScriptedDisplay(keys)
    a = Assistant(
        camera=FakeCamera(frames),
        display=display,
        listener=FakeListener(*utterances),
        prompts=prompts,
        detector=detector or FakeDetector([]),
        faces=faces or FakeFaces(),
        registry=PeopleRegistry(cfg.people_csv),
        ocr_reader=ocr_reader or (lambda s, o: []),
        name_synthesizer=synth,
        config=cfg,
        announcer=Announcer(0),
    )
    return a, display


def test_startup_plays_intro_and_exit_says_goodbye(cfg, prompts, speaker):
    a, _ = build(cfg, prompts, utterances=["0"])
    a.run()
    assert speaker.names == ["commands/intro", "commands/end"]


def test_unrecognised_command_asks_user_to_repeat(cfg, prompts, speaker):
    a, _ = build(cfg, prompts, utterances=["blah", None, "0"])
    a.run()
    assert speaker.names == ["commands/intro", "commands/repeat", "commands/repeat", "commands/end"]


def test_objects_mode_announces_three_largest_with_sides(cfg, prompts, speaker):
    ds = [
        Detection("cup", 0.9, (10, 10, 20, 20)),  # smallest -> dropped
        Detection("dog", 0.9, (50, 100, 200, 200)),  # centre x=150 -> left
        Detection("car", 0.9, (400, 100, 200, 100)),  # centre x=500 -> right
        Detection("chair", 0.9, (300, 300, 60, 60)),  # centre x=330 -> right
    ]
    a, display = build(
        cfg, prompts, utterances=[CMD_OBJECTS, "0"], detector=FakeDetector([ds]), keys=[KEY_ESC]
    )
    a.run()
    assert speaker.names == [
        "commands/intro",
        "objects/dog",
        "directions/left",
        "objects/car",
        "directions/right",
        "objects/chair",
        "directions/right",
        "commands/end",
    ]
    assert len(display.frames) == 1  # one rendered HUD frame, then Esc


def test_objects_mode_stops_when_camera_ends(cfg, prompts, speaker):
    a, _ = build(
        cfg, prompts, utterances=[CMD_OBJECTS, "0"], frames=[np.zeros((480, 640, 3), np.uint8)]
    )
    a.run()  # one frame, then camera exhausted -> returns to the menu
    assert speaker.names[-1] == "commands/end"


def test_objects_mode_is_rate_limited(cfg, prompts, speaker):
    d = [Detection("cup", 0.9, (10, 10, 50, 50))]
    now = [0.0]
    a, _ = build(
        cfg,
        prompts,
        utterances=[CMD_OBJECTS, "0"],
        detector=FakeDetector([d, d, d]),
        keys=[-1, -1, KEY_ESC],
    )
    a.announcer = Announcer(3.0, clock=lambda: now[0])
    a.run()
    assert speaker.names.count("objects/cup") == 1  # three frames, one announcement


def test_new_person_flow_registers_trains_and_stores_name(cfg, prompts, speaker):
    faces, saved = FakeFaces(), []
    a, _ = build(
        cfg,
        prompts,
        utterances=[CMD_PEOPLE, CMD_NEW, "ليلى", "0"],
        faces=faces,
        synth=lambda person, path: saved.append((person.name, path.name)),
    )
    a.run()
    assert speaker.names == ["commands/intro", "commands/f_r", "commands/f_r1", "commands/end"]
    assert faces.calls == [("register", 1), ("train",)]
    assert saved == [("ليلى", "1.mp3")]
    assert PeopleRegistry(cfg.people_csv).get(1).name == "ليلى"


def test_new_person_not_stored_when_no_face_was_captured(cfg, prompts, speaker):
    a, _ = build(
        cfg, prompts, utterances=[CMD_PEOPLE, CMD_NEW, "ليلى", "0"], faces=FakeFaces(samples=0)
    )
    a.run()
    assert "commands/r_fail" in speaker.names
    assert PeopleRegistry(cfg.people_csv).all() == []


def test_new_person_name_not_understood(cfg, prompts, speaker):
    faces = FakeFaces()
    a, _ = build(cfg, prompts, utterances=[CMD_PEOPLE, CMD_NEW, None, "0"], faces=faces)
    a.run()
    assert faces.calls == []
    assert "commands/repeat" in speaker.names


def test_tts_failure_does_not_lose_enrolment(cfg, prompts):
    def boom(person, path):
        raise ConnectionError("offline")

    a, _ = build(cfg, prompts, utterances=[CMD_PEOPLE, CMD_NEW, "سامي", "0"], synth=boom)
    a.run()
    assert PeopleRegistry(cfg.people_csv).get(1).name == "سامي"


def test_known_person_name_is_spoken(cfg, prompts, speaker):
    PeopleRegistry(cfg.people_csv).add("جميل")
    cfg.names_dir.mkdir(parents=True)
    (cfg.names_dir / "1.mp3").write_bytes(b"x")
    a, _ = build(
        cfg, prompts, utterances=[CMD_PEOPLE, CMD_KNOWN, "0"], faces=FakeFaces(identified=1)
    )
    a.run()
    assert speaker.played[-2] == cfg.names_dir / "1.mp3"


def test_unknown_face_says_not_registered(cfg, prompts, speaker):
    a, _ = build(
        cfg, prompts, utterances=[CMD_PEOPLE, CMD_KNOWN, "0"], faces=FakeFaces(identified=None)
    )
    a.run()
    assert speaker.names[-2:] == ["commands/f_r3", "commands/end"]


def test_people_submenu_unknown_choice(cfg, prompts, speaker):
    a, _ = build(cfg, prompts, utterances=[CMD_PEOPLE, "xyz", "0"])
    a.run()
    assert speaker.names == ["commands/intro", "commands/f_r", "commands/repeat", "commands/end"]


def test_text_mode_speaks_matching_landmark(cfg, prompts, speaker):
    a, _ = build(
        cfg,
        prompts,
        utterances=[CMD_TEXT, "0"],
        keys=[KEY_SPACE, KEY_ESC],
        ocr_reader=lambda src, out: [(None, "صيدلية", 0.9)],
    )
    a.run()
    assert speaker.names == ["commands/intro", "ocr/1", "commands/end"]


def test_text_mode_reports_when_nothing_is_found(cfg, prompts, speaker):
    a, _ = build(cfg, prompts, utterances=[CMD_TEXT, "0"], keys=[KEY_SPACE, KEY_ESC])
    a.run()
    assert speaker.names == ["commands/intro", "commands/r_fail", "commands/end"]


def test_text_mode_only_reads_on_space(cfg, prompts, speaker):
    calls = []
    a, _ = build(
        cfg,
        prompts,
        utterances=[CMD_TEXT, "0"],
        keys=[-1, -1, KEY_ESC],
        ocr_reader=lambda s, o: calls.append(1) or [],
    )
    a.run()
    assert calls == []


def test_display_is_always_closed(cfg, prompts):
    a, display = build(cfg, prompts, utterances=["0"])
    closed = []
    display.close = lambda: closed.append(True)
    a.run()
    assert closed == [True]
