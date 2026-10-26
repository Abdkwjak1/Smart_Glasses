"""Command-line entry point: `smart-glasses` or `python -m smart_glasses`."""

from __future__ import annotations

import argparse
import logging

from .config import DEFAULT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="smart-glasses", description=__doc__)
    p.add_argument("--camera", default="0", help="camera index or video file (default: 0)")
    p.add_argument("--headless", action="store_true", help="do not open any windows")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    from .assistant import Assistant
    from .audio import GoogleListener, PlaysoundSpeaker, Prompts, synthesize
    from .camera import open_source
    from .detection import ObjectDetector
    from .display import CvDisplay, ScriptedDisplay
    from .faces import FaceRecognizer
    from .registry import PeopleRegistry

    cfg = DEFAULT
    source = int(args.camera) if args.camera.isdigit() else args.camera
    camera = open_source(source)
    display = ScriptedDisplay() if args.headless else CvDisplay()

    def name_audio(person, path):
        synthesize("،هذا الشخص هو" + person.name, path, lang="ar")

    Assistant(
        camera=camera,
        display=display,
        listener=GoogleListener(cfg.speech_language),
        prompts=Prompts(PlaysoundSpeaker(), cfg),
        detector=ObjectDetector(cfg),
        faces=FaceRecognizer(camera, cfg, on_frame=display.show),
        registry=PeopleRegistry(cfg.people_csv),
        name_synthesizer=name_audio,
        config=cfg,
    ).run()
    camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
