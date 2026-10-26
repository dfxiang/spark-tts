from __future__ import annotations

import wave
from collections import OrderedDict
from pathlib import Path
from threading import RLock

import torch

from cli.SparkTTS import SparkTTS


class FakeAudioTokenizer:
    def __init__(self) -> None:
        self.calls = 0
        self.semantic_tokens: torch.Tensor | None = None

    def tokenize(self, _: Path) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return torch.tensor([[1, 2]]), torch.tensor([[3, 4, 5]])

    def detokenize(
        self,
        _: torch.Tensor,
        semantic_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.semantic_tokens = semantic_tokens
        return torch.zeros(16000)


class FakeModelInputs(dict):
    def __init__(self) -> None:
        self.input_ids = torch.tensor([[10, 11]])
        super().__init__(input_ids=self.input_ids)

    def to(self, _: torch.device) -> FakeModelInputs:
        return self


class FakeLLMTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, *_: object, **__: object) -> FakeModelInputs:
        return FakeModelInputs()


class FakeLLMModel:
    def generate(self, **_: object) -> torch.Tensor:
        return torch.tensor(
            [[10, 11, 155761, 155762, 155763, 165157]]
        )


def write_reference(path: Path) -> None:
    with wave.open(str(path), "wb") as reference:
        reference.setnchannels(1)
        reference.setsampwidth(2)
        reference.setframerate(16000)
        reference.writeframes(b"\0\0" * 16000)


def make_model() -> SparkTTS:
    model = SparkTTS.__new__(SparkTTS)
    model.device = torch.device("cpu")
    model.prompt_cache_size = 2
    model._prompt_cache = OrderedDict()
    model._prompt_cache_lock = RLock()
    model.audio_tokenizer = FakeAudioTokenizer()
    model._semantic_token_start = 155761
    model._semantic_token_count = 8192
    model._global_token_start = 151665
    model._global_token_count = 4096
    return model


def test_inference_extracts_semantic_tokens_without_text_decoding(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    write_reference(reference)
    model = make_model()
    model.tokenizer = FakeLLMTokenizer()
    model.model = FakeLLMModel()

    waveform = model.inference("target text", reference, None)

    assert waveform.shape == (16000,)
    assert torch.equal(
        model.audio_tokenizer.semantic_tokens,
        torch.tensor([[0, 1, 2]]),
    )


def test_reference_audio_tokens_and_prompt_strings_are_cached(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.wav"
    write_reference(reference)
    model = make_model()

    prompt, global_ids = model.process_prompt(
        "target text",
        reference,
        "reference text",
    )
    model.process_prompt("another target", reference, "reference text")

    assert model.audio_tokenizer.calls == 1
    assert prompt.startswith("<|task_tts|><|start_content|>reference text")
    assert "<|bicodec_global_1|><|bicodec_global_2|>" in prompt
    assert (
        "<|bicodec_semantic_3|><|bicodec_semantic_4|>"
        "<|bicodec_semantic_5|>"
    ) in prompt
    assert torch.equal(global_ids, torch.tensor([[1, 2]]))


def test_generated_bicodec_tokens_are_extracted_directly() -> None:
    model = make_model()
    generated = torch.tensor(
        [
            155761,
            155762,
            155763,
            151665,
            151666,
            165157,
        ]
    )

    semantic = model._extract_audio_token_ids(
        generated,
        model._semantic_token_start,
        model._semantic_token_count,
    )
    global_tokens = model._extract_audio_token_ids(
        generated,
        model._global_token_start,
        model._global_token_count,
    )

    assert torch.equal(semantic, torch.tensor([0, 1, 2]))
    assert torch.equal(global_tokens, torch.tensor([0, 1]))
