"""Audio playback and speech I/O behind small interfaces, so the app logic can be tested."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import DEFAULT, Config


class Speaker(Protocol):
    def play(self, path: Path) -> None: ...


class Listener(Protocol):
    def listen(self) -> str | None: ...


class PlaysoundSpeaker:
    def play(self, path: Path) -> None:
        from playsound import playsound  # lazy: optional dependency

        playsound(str(path))


class Prompts:
    """Resolves prompt names to the recorded Arabic mp3 files and plays them."""

    def __init__(self, speaker: Speaker, config: Config = DEFAULT):
        self.speaker = speaker
        self.config = config

    def path(self, group: str, name: str) -> Path:
        return self.config.audio_dir / group / f"{name}.mp3"

    def say(self, group: str, name: str) -> None:
        p = self.path(group, name)
        if p.exists():
            self.speaker.play(p)

    def say_file(self, path: Path) -> None:
        if Path(path).exists():
            self.speaker.play(path)

    def announce_object(self, label: str, side: str) -> None:
        self.say("objects", label)
        self.say("directions", side)


class GoogleListener:
    """Push-to-talk microphone listener using Google speech recognition."""

    def __init__(self, language: str = "ar-EG"):
        import speech_recognition as sr

        self._sr = sr
        self.language = language
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen(self) -> str | None:
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source)
            return self.recognizer.recognize_google(audio, language=self.language)
        except (self._sr.UnknownValueError, self._sr.RequestError):
            return None


def synthesize(text: str, path: Path, lang: str = "ar") -> None:
    """Text-to-speech to an mp3 file (needs network access)."""
    from gtts import gTTS

    path.parent.mkdir(parents=True, exist_ok=True)
    gTTS(text, lang=lang).save(str(path))
