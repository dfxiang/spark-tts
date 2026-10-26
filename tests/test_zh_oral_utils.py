from api.zh_oral_utils import to_koubo_text


def test_converts_chinese_context_numbers_for_speech() -> None:
    assert (
        to_koubo_text("我9:05到，花了123万，增长12.5%。")
        == "我九点零五到，花了一百二十三万，增长百分之十二点五。"
    )
    assert to_koubo_text("1983年10月3日") == "一九八三年十月三日"


def test_preserves_english_context_and_plain_english() -> None:
    assert to_koubo_text("I have 123 apples.") == "I have 123 apples."
    assert to_koubo_text("我买了 3 个 iPhone 15。") == "我买了 三 个 iPhone 15。"
    assert to_koubo_text("28.64%") == "百分之二十八点六四"
