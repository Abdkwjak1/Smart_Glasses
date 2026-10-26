import cv2
import numpy as np
import pytest

from smart_glasses.faces import FaceRecognizer, largest_face, load_training_set, majority_vote

from .conftest import FakeCamera


class WholeFrameDetector:
    """Stand-in for the Haar cascade: reports the entire frame as one face."""

    def detectMultiScale(self, gray, *_):
        h, w = gray.shape
        return np.array([[0, 0, w, h]])


class NoFaceDetector:
    def detectMultiScale(self, gray, *_):
        return np.empty((0, 4), int)


def face_image(seed, size=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (size, size, 3), dtype=np.uint8)


def test_majority_vote():
    assert majority_vote([2, 1, 2, 3, 2]) == 2
    assert majority_vote([]) is None


def test_largest_face_picks_the_biggest_not_the_first():
    class Two:
        def detectMultiScale(self, *_):
            return np.array([[0, 0, 10, 10], [20, 20, 40, 40]])

    gray = np.zeros((100, 100), np.uint8)
    crop, box = largest_face(gray, Two())
    assert box == (20, 20, 40, 40) and crop.shape == (40, 40)
    assert largest_face(gray, NoFaceDetector()) is None


def test_register_saves_requested_samples(cfg):
    cam = FakeCamera([face_image(i) for i in range(50)])
    rec = FaceRecognizer(cam, cfg)
    rec.detector = WholeFrameDetector()
    assert rec.register(7) == cfg.face_samples
    assert len(list(cfg.face_dataset_dir.glob("user.7.*.jpg"))) == cfg.face_samples


def test_register_with_no_face_returns_zero_instead_of_hanging(cfg):
    rec = FaceRecognizer(FakeCamera([face_image(0)] * 30), cfg)
    rec.detector = NoFaceDetector()
    assert rec.register(1) == 0


def test_load_training_set_parses_ids_from_filenames(cfg):
    cfg.face_dataset_dir.mkdir(parents=True)
    for pid, n in [(3, 1), (3, 2), (12, 1)]:
        cv2.imwrite(str(cfg.face_dataset_dir / f"user.{pid}.{n}.jpg"), face_image(pid)[:, :, 0])
    ids, faces = load_training_set(cfg.face_dataset_dir)
    assert sorted(ids.tolist()) == [3, 3, 12] and len(faces) == 3


def test_train_then_identify_end_to_end(cfg):
    """Enrol two synthetic 'people', then recognise one of them through the camera."""

    def enrol(pid, seed):
        rec = FaceRecognizer(FakeCamera([face_image(seed)] * 40), cfg)
        rec.detector = WholeFrameDetector()
        rec.register(pid)

    enrol(1, seed=10)
    enrol(2, seed=20)
    trainer = FaceRecognizer(FakeCamera([]), cfg)
    assert trainer.train() == 2 * cfg.face_samples

    seeker = FaceRecognizer(FakeCamera([face_image(20)] * 40), cfg)
    seeker.detector = WholeFrameDetector()
    assert seeker.identify() == 2


def test_identify_rejects_strangers(cfg):
    rec = FaceRecognizer(FakeCamera([face_image(10)] * 40), cfg)
    rec.detector = WholeFrameDetector()
    rec.register(1)
    rec.train()
    stranger = FaceRecognizer(FakeCamera([face_image(999)] * 40), cfg)
    stranger.detector = WholeFrameDetector()
    assert stranger.identify() is None  # distance above face_max_distance -> unknown


def test_identify_without_trained_model_is_none(cfg):
    assert FaceRecognizer(FakeCamera([]), cfg).identify() is None


def test_train_without_samples_is_noop(cfg):
    assert FaceRecognizer(FakeCamera([]), cfg).train() == 0


@pytest.mark.parametrize("n", [0, 1])
def test_identify_with_camera_failure_terminates(cfg, n):
    rec = FaceRecognizer(FakeCamera([]), cfg)
    assert rec.identify() is None
