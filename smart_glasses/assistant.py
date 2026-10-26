"""The voice-driven state machine that ties every block together.

All hardware (camera, microphone, speaker, window, models) is injected, so the whole spoken
dialogue can be exercised in tests with scripted fakes.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable

import cv2

from . import hud, ocr
from .audio import Listener, Prompts
from .config import (
    CMD_EXIT,
    CMD_KNOWN,
    CMD_NEW,
    CMD_OBJECTS,
    CMD_PEOPLE,
    CMD_TEXT,
    DEFAULT,
    Config,
)
from .detection import Announcer, largest, to_announcements
from .display import KEY_ESC, KEY_SPACE, Display
from .registry import PeopleRegistry

log = logging.getLogger("smart_glasses")


class Assistant:
    def __init__(
        self,
        *,
        camera,
        display: Display,
        listener: Listener,
        prompts: Prompts,
        detector,
        faces,
        registry: PeopleRegistry,
        ocr_reader: Callable[[str, str], list] = ocr.read_image,
        name_synthesizer: Callable | None = None,
        config: Config = DEFAULT,
        announcer: Announcer | None = None,
    ):
        self.camera = camera
        self.display = display
        self.listener = listener
        self.prompts = prompts
        self.detector = detector
        self.faces = faces
        self.registry = registry
        self.ocr_reader = ocr_reader
        self.name_synthesizer = name_synthesizer
        self.config = config
        self.announcer = announcer or Announcer(config.announce_interval_s)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        self.prompts.say("commands", "intro")
        handlers = {
            CMD_OBJECTS: self.objects_mode,
            CMD_PEOPLE: self.people_mode,
            CMD_TEXT: self.text_mode,
        }
        try:
            while True:
                command = (self.listener.listen() or "").strip()
                if command == CMD_EXIT:
                    self.prompts.say("commands", "end")
                    return
                handler = handlers.get(command)
                if handler:
                    handler()
                else:
                    log.info("Unrecognised command %r", command)
                    self.prompts.say("commands", "repeat")
        finally:
            self.display.close()

    # ------------------------------------------------------------------ modes
    def objects_mode(self) -> None:
        """Live object detection; speaks the 3 largest objects and which side they're on."""
        while True:
            ok, frame = self.camera.read()
            if not ok:
                return
            found = self.detector.detect(frame)
            top = largest(found, self.config.max_announced)
            key = self.display.show(hud.render(frame, found, top, "OBJECTS  -  Esc to exit"))
            if self.announcer.ready():
                for label, side in to_announcements(top, frame.shape[1]):
                    self.prompts.announce_object(label, side)
            if key == KEY_ESC:
                return

    def people_mode(self) -> None:
        self.prompts.say("commands", "f_r")
        choice = (self.listener.listen() or "").strip()
        if choice == CMD_NEW:
            self._register_person()
        elif choice == CMD_KNOWN:
            self._identify_person()
        else:
            self.prompts.say("commands", "repeat")

    def text_mode(self) -> None:
        """Space bar captures a frame; Esc leaves. Announces recognised landmark words."""
        while True:
            ok, frame = self.camera.read()
            if not ok:
                return
            key = self.display.show(frame)
            if key == KEY_ESC:
                return
            if key == KEY_SPACE:
                self._read_frame(frame)

    # ------------------------------------------------------------------ helpers
    def _read_frame(self, frame) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, annotated = os.path.join(tmp, "capture.png"), os.path.join(tmp, "out.jpg")
            cv2.imwrite(src, frame)
            results = self.ocr_reader(src, annotated)
        index = ocr.find_keyword(results, self.config.ocr_keywords)
        if index is None:
            self.prompts.say("commands", "r_fail")
        else:
            self.prompts.say("ocr", str(index))

    def _register_person(self) -> None:
        self.prompts.say("commands", "f_r1")
        name = (self.listener.listen() or "").strip()
        if not name:
            self.prompts.say("commands", "repeat")
            return
        person_id = self.registry.next_id()
        if self.faces.register(person_id) == 0:
            self.prompts.say("commands", "r_fail")
            return
        person = self.registry.add(name)
        assert person.id == person_id
        self.faces.train()
        if self.name_synthesizer:
            try:
                self.name_synthesizer(person, self.config.names_dir / f"{person.id}.mp3")
            except Exception:  # offline TTS must not lose the enrolment
                log.exception("Could not synthesise the spoken name for %s", person.name)

    def _identify_person(self) -> None:
        self.prompts.say("commands", "f_r2")
        person_id = self.faces.identify()
        person = self.registry.get(person_id) if person_id is not None else None
        name_audio = self.config.names_dir / f"{person_id}.mp3"
        if person and name_audio.exists():
            self.prompts.say_file(name_audio)
        else:
            self.prompts.say("commands", "f_r3")
