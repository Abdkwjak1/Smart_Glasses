"""Window abstraction so modes can run headless (tests, SSH, CI)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import cv2
import numpy as np

KEY_ESC = 27
KEY_SPACE = 32


class Display(Protocol):
    def show(self, image: np.ndarray) -> int:
        """Show a frame and return the pressed key code (-1 if none)."""

    def close(self) -> None: ...


class CvDisplay:
    def __init__(self, title: str = "Smart Glasses"):
        self.title = title

    def show(self, image: np.ndarray) -> int:
        cv2.imshow(self.title, image)
        key = cv2.waitKey(1)
        return key & 0xFF if key != -1 else -1

    def close(self) -> None:
        cv2.destroyAllWindows()


class ScriptedDisplay:
    """Headless display that records frames and replays a scripted key sequence."""

    def __init__(self, keys: Iterable[int] = ()):
        self._keys = iter(keys)
        self.frames: list[np.ndarray] = []

    def show(self, image: np.ndarray) -> int:
        self.frames.append(image)
        return next(self._keys, -1)

    def close(self) -> None:
        pass
