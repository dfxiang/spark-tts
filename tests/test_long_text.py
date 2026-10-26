from api.long_text import normalize_text, split_text_by_punctuation


def test_splits_at_punctuation_within_requested_limits() -> None:
    text = "甲乙丙丁戊己庚辛壬癸，" * 18
    chunks = split_text_by_punctuation(text, min_chars=32, max_chars=128)

    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert all(32 <= len(chunk) <= 128 for chunk in chunks)
    assert all(chunk.endswith("，") for chunk in chunks)


def test_does_not_hard_cut_an_unpunctuated_span() -> None:
    text = "甲" * 200
    chunks = split_text_by_punctuation(text, min_chars=32, max_chars=128)

    assert chunks == [text]


def test_short_text_stays_as_one_chunk() -> None:
    text = "短文本。"
    assert split_text_by_punctuation(text) == [text]


def test_removes_whitespace_and_invisible_characters_before_splitting() -> None:
    text = " 第一\u200b段\t文本。\u3000第二段文本。 "
    assert normalize_text(text) == "第一段文本。第二段文本。"
    assert split_text_by_punctuation(text) == ["第一段文本。第二段文本。"]


def test_preserves_single_spaces_between_english_words() -> None:
    text = "I need\n to\u200b cancel   my appointment for tomorrow."
    assert normalize_text(text) == "I need to cancel my appointment for tomorrow."
