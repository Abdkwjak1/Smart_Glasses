"""Face enrolment and recognition (Haar cascade detector + LBPH recogniser)."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import DEFAULT, Config


def largest_face(gray: np.ndarray, detector) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Crop the biggest face in a grayscale frame, or None. (The old code took faces[0].)"""
    faces = detector.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces.tolist(), key=lambda f: f[2] * f[3])
    return gray[y : y + h, x : x + w], (x, y, w, h)


def majority_vote(ids: list[int]) -> int | None:
    return Counter(ids).most_common(1)[0][0] if ids else None


def load_training_set(dataset_dir: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Read `user.<id>.<n>.jpg` files into (ids, grayscale images)."""
    ids, faces = [], []
    for path in sorted(Path(dataset_dir).glob("user.*.*.jpg")):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        ids.append(int(path.name.split(".")[1]))
        faces.append(img)
    return np.array(ids, dtype=np.int32), faces


class FaceRecognizer:
    def __init__(self, camera, config: Config = DEFAULT, on_frame=None):
        self.camera = camera
        self.config = config
        self.on_frame = on_frame or (lambda frame: None)  # lets the UI show a preview
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def _recognizer(self):
        return cv2.face.LBPHFaceRecognizer_create()

    def register(self, person_id: int) -> int:
        """Capture face samples from the camera; returns the number saved."""
        out = self.config.face_dataset_dir
        out.mkdir(parents=True, exist_ok=True)
        saved, deadline = 0, time.monotonic() + self.config.face_timeout_s
        while saved < self.config.face_samples and time.monotonic() < deadline:
            ok, frame = self.camera.read()
            if not ok:
                break
            found = largest_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.detector)
            if found:
                saved += 1
                cv2.imwrite(str(out / f"user.{person_id}.{saved}.jpg"), found[0])
            self.on_frame(frame)
        return saved

    def train(self) -> int:
        ids, faces = load_training_set(self.config.face_dataset_dir)
        if not len(faces):
            return 0
        rec = self._recognizer()
        rec.train(faces, ids)
        self.config.face_model_path.parent.mkdir(parents=True, exist_ok=True)
        rec.save(str(self.config.face_model_path))
        return len(faces)

    def identify(self) -> int | None:
        """Vote over several frames; returns the person ID or None if nobody is recognised."""
        c = self.config
        if not c.face_model_path.exists():
            return None
        rec = self._recognizer()
        rec.read(str(c.face_model_path))
        votes: list[int] = []
        seen, deadline = 0, time.monotonic() + c.face_timeout_s
        while seen < c.face_vote_frames and time.monotonic() < deadline:
            ok, frame = self.camera.read()
            if not ok:
                break
            found = largest_face(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.detector)
            if found:
                seen += 1
                pid, distance = rec.predict(found[0])
                if distance <= c.face_max_distance:
                    votes.append(pid)
            self.on_frame(frame)
        # Require a majority of the sampled frames, otherwise treat as unknown.
        winner = majority_vote(votes)
        if winner is not None and votes.count(winner) * 2 > max(seen, 1):
            return winner
        return None
