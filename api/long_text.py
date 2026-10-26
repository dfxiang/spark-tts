"""Punctuation-aware chunking for long text synthesis."""

from __future__ import annotations

import unicodedata

PUNCTUATION = "，,、：:。！？!?；;…"
CLOSING_MARKS = "”’」』）》】〕〉"


def normalize_text(text: str) -> str:
    """Remove invisible formatting while preserving English word boundaries."""
    visible: list[str] = []
    pending_space = False
    for character in text:
        category = unicodedata.category(character)
        if character.isspace():
            pending_space = bool(visible)
            continue
        if category.startswith("C"):
            continue
        if "\ufe00" <= character <= "\ufe0f":
            continue
        if pending_space and visible:
            previous = visible[-1]
            if (
                previous.isascii()
                and character.isascii()
                and not previous.isspace()
                and not character.isspace()
            ):
                visible.append(" ")
        pending_space = False
        visible.append(character)
    return "".join(visible)


def _punctuation_boundaries(text: str) -> list[int]:
    boundaries: set[int] = {len(text)}
    index = 0
    while index < len(text):
        if text[index] not in PUNCTUATION:
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in PUNCTUATION:
            end += 1
        while end < len(text) and text[end] in CLOSING_MARKS:
            end += 1
        boundaries.add(end)
        index = end
    return sorted(boundaries)


def _split_at_punctuation_limits(
    text: str,
    boundaries: list[int],
    min_chars: int,
    max_chars: int,
) -> list[str] | None:
    target = (min_chars + max_chars) / 2
    states: dict[int, tuple[int, float, int | None]] = {0: (0, 0.0, None)}
    for end in boundaries:
        best: tuple[int, float, int | None] | None = None
        for start, (count, deviation, _) in states.items():
            length = end - start
            if length < min_chars or length > max_chars:
                continue
            candidate = (
                count + 1,
                deviation + abs(length - target),
                start,
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None:
            states[end] = best

    if len(text) not in states:
        return None

    chunks: list[str] = []
    end = len(text)
    while end > 0:
        _, _, previous = states[end]
        if previous is None:
            raise RuntimeError("invalid long-text splitter state")
        chunks.append(text[previous:end])
        end = previous
    chunks.reverse()
    return chunks


def _split_without_hard_cuts(
    text: str,
    boundaries: list[int],
    min_chars: int,
    max_chars: int,
) -> list[str]:
    """Use punctuation boundaries even when a long unpunctuated span is oversized."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        candidates = [boundary for boundary in boundaries if boundary > start]
        in_range = [
            boundary
            for boundary in candidates
            if min_chars <= boundary - start <= max_chars
        ]
        if in_range:
            end = max(in_range)
        else:
            after_minimum = [
                boundary for boundary in candidates if boundary - start >= min_chars
            ]
            end = after_minimum[0] if after_minimum else candidates[0]
        chunks.append(text[start:end])
        start = end
    return chunks


def split_text_by_punctuation(
    text: str,
    *,
    min_chars: int = 32,
    max_chars: int = 128,
) -> list[str]:
    """Split at punctuation only; max_chars is a soft limit when unavoidable."""
    if min_chars < 1:
        raise ValueError("min_chars must be positive")
    if max_chars < min_chars:
        raise ValueError("max_chars must be at least min_chars")

    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    boundaries = _punctuation_boundaries(text)
    exact = _split_at_punctuation_limits(
        text,
        boundaries,
        min_chars,
        max_chars,
    )
    if exact is not None:
        return exact
    return _split_without_hard_cuts(
        text,
        boundaries,
        min_chars,
        max_chars,
    )
