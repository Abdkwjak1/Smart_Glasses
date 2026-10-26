"""Arabic text reading: find landmark keywords (pharmacy, clinic, ...) in OCR output."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def find_keyword(results: Iterable[Sequence], keywords: Sequence[str]) -> int | None:
    """Index into `keywords` of the first keyword found in the OCR results, else None.

    `results` is what ArabicOcr returns: items shaped like (bbox, text, confidence).
    Scanning is in reading order, so the first match in the text wins.
    """
    lookup = {kw: i for i, kw in enumerate(keywords)}
    for item in results:
        for word in str(item[1]).split():
            if word in lookup:
                return lookup[word]
    return None


def read_image(image_path: str, annotated_path: str):
    """Run Arabic OCR. ArabicOcr pulls in torch, so it is imported only when needed."""
    try:
        from ArabicOcr import arabicocr
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Text mode needs the optional OCR extra: pip install ArabicOcr") from exc
    return arabicocr.arabic_ocr(image_path, annotated_path)
