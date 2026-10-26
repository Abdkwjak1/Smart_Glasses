from smart_glasses.config import OCR_KEYWORDS
from smart_glasses.ocr import find_keyword


def result(text):
    return ([[0, 0], [1, 1]], text, 0.9)


def test_finds_keyword_and_returns_its_index():
    assert find_keyword([result("صيدلية")], OCR_KEYWORDS) == 1
    assert find_keyword([result("اهلا"), result("مركز")], OCR_KEYWORDS) == 6


def test_keyword_inside_a_longer_line():
    assert find_keyword([result("صيدلية الشفاء المركزية")], OCR_KEYWORDS) == 1


def test_first_match_in_reading_order_wins():
    assert find_keyword([result("مكتبة"), result("عيادة")], OCR_KEYWORDS) == 7


def test_no_match_or_no_text():
    assert find_keyword([result("مرحبا")], OCR_KEYWORDS) is None
    assert find_keyword([], OCR_KEYWORDS) is None


def test_failure_sound_not_played_after_a_success_on_the_last_line():
    # Regression: the legacy loop played "no text found" whenever the LAST line was reached,
    # even if an earlier/the last line matched.
    assert find_keyword([result("x"), result("ماركت")], OCR_KEYWORDS) == 8
