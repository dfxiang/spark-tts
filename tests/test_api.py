from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from api.app import Settings, create_app
from api.long_text import normalize_text, split_text_by_punctuation
from api.zh_oral_utils import to_koubo_text


class FakeEngine:
    loaded = True
    device = "fake-cpu"

    def synthesize(self, text: str, reference_audio: Path, prompt_text: str | None):
        assert text
        assert reference_audio.is_file()
        assert prompt_text == "This is the reference transcript."
        return np.full(16000, 0.08, dtype=np.float32)


class BatchFakeEngine(FakeEngine):
    def __init__(self) -> None:
        self.batch_texts: list[str] | None = None

    def synthesize_batch(
        self,
        texts: list[str],
        reference_audio: Path,
        prompt_text: str | None,
    ) -> list[np.ndarray]:
        self.batch_texts = texts
        assert reference_audio.is_file()
        assert prompt_text == "This is the reference transcript."
        return [
            np.full(8000, 0.08, dtype=np.float32)
            for _ in texts
        ]


class LongSilenceFakeEngine(FakeEngine):
    def synthesize(self, text: str, reference_audio: Path, prompt_text: str | None):
        super().synthesize(text, reference_audio, prompt_text)
        return np.concatenate(
            (
                np.full(8000, 0.08, dtype=np.float32),
                np.zeros(32000, dtype=np.float32),
                np.full(8000, 0.08, dtype=np.float32),
            )
        )


class SilentSegmentBatchFakeEngine(BatchFakeEngine):
    def synthesize_batch(
        self,
        texts: list[str],
        reference_audio: Path,
        prompt_text: str | None,
    ) -> list[np.ndarray]:
        super().synthesize_batch(texts, reference_audio, prompt_text)
        return [
            np.full(8000, 0.08, dtype=np.float32),
            np.zeros(16000, dtype=np.float32),
        ]


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        model_dir=tmp_path / "model",
        voice_dir=tmp_path / "voices",
        max_reference_seconds=5,
    )


def wav_bytes() -> bytes:
    output = BytesIO()
    sf.write(
        output,
        np.zeros(16000, dtype=np.float32),
        16000,
        format="WAV",
        subtype="PCM_16",
    )
    return output.getvalue()


def test_health_and_openai_model_list(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), FakeEngine()))

    page = client.get("/")
    assert page.status_code == 200
    assert "Spark Voice Studio" in page.text
    assert 'id="resultMetrics"' in page.text
    assert 'id="voiceSelect"' in page.text
    assert 'id="voiceMetadata"' in page.text
    assert 'id="referenceAudio"' not in page.text
    assert "const MAX_TEXT_CHARS = 4096;" in page.text
    assert '"/v1/audio/long_text"' in page.text

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    models = client.get("/v1/models")
    assert models.status_code == 200
    assert models.json()["data"][0]["id"] == "spark-tts"

    voice_manager = client.get("/voices")
    assert voice_manager.status_code == 200
    assert "声音管理" in voice_manager.text


def test_voice_upload_and_openai_speech_endpoint(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path), FakeEngine())
    client = TestClient(app)

    voice_response = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Test voice",
        },
    )
    assert voice_response.status_code == 201, voice_response.text
    voice_id = voice_response.json()["id"]

    speech_response = client.post(
        "/v1/audio/speech",
        json={
            "model": "tts-1",
            "input": "A short test sentence.",
            "voice": voice_id,
            "response_format": "wav",
            "speed": 1.0,
        },
    )
    assert speech_response.status_code == 200, speech_response.text
    assert speech_response.content.startswith(b"RIFF")
    assert speech_response.headers["content-type"] == "audio/wav"
    assert speech_response.headers["x-spark-tts-voice"] == voice_id
    assert float(speech_response.headers["duration"]) == 1.0
    assert float(speech_response.headers["rtf"]) > 0
    assert speech_response.headers["batch-size"] == "1"


def test_voice_name_lookup_is_available_after_restart(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    first_client = TestClient(create_app(settings, FakeEngine()))
    voice_response = first_client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Named Voice",
        },
    )
    assert voice_response.status_code == 201

    restarted_client = TestClient(create_app(settings, FakeEngine()))
    response = restarted_client.post(
        "/v1/audio/speech",
        json={
            "input": "Use the display name.",
            "voice": "named voice",
            "response_format": "wav",
        },
    )
    assert response.status_code == 200, response.text


def test_voice_management_crud(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), FakeEngine()))
    created = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Editable Voice",
        },
    )
    assert created.status_code == 201
    voice_id = created.json()["id"]
    assert created.json()["duration_seconds"] == 1.0

    listed = client.get("/api/voices").json()["data"]
    assert listed[0]["id"] == voice_id
    assert listed[0]["name"] == "Editable Voice"

    updated = client.patch(
        f"/api/voices/{voice_id}",
        json={"name": "Renamed Voice", "prompt_text": "Updated transcript."},
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Renamed Voice"
    assert updated.json()["prompt_text"] == "Updated transcript."

    detail = client.get(f"/api/voices/{voice_id}")
    assert detail.status_code == 200
    assert detail.json()["name"] == "Renamed Voice"

    audio = client.get(f"/api/voices/{voice_id}/audio")
    assert audio.status_code == 200
    assert audio.content.startswith(b"RIFF")

    replaced = client.put(
        f"/api/voices/{voice_id}/audio",
        files={"audio": ("replacement.wav", wav_bytes(), "audio/wav")},
    )
    assert replaced.status_code == 200
    assert replaced.json()["duration_seconds"] == 1.0

    deleted = client.delete(f"/api/voices/{voice_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/voices/{voice_id}").status_code == 404


def test_duplicate_voice_names_are_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), FakeEngine()))
    for _ in range(2):
        response = client.post(
            "/api/voices",
            files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
            data={
                "prompt_text": "This is the reference transcript.",
                "name": "Duplicate",
            },
        )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_voice_name"


def test_default_name_uses_chinese_filename_and_avoids_collision(
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(make_settings(tmp_path), FakeEngine()))
    responses = [
        client.post(
            "/api/voices",
            files={"audio": ("中文参考.wav", wav_bytes(), "audio/wav")},
            data={"prompt_text": "This is the reference transcript."},
        )
        for _ in range(2)
    ]

    assert responses[0].status_code == 201
    assert responses[0].json()["name"] == "中文参考.wav"
    assert responses[1].status_code == 201
    assert responses[1].json()["name"] == "中文参考 (2).wav"


def test_unknown_voice_uses_openai_error_shape(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), FakeEngine()))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "spark-tts",
            "input": "Hello.",
            "voice": "missing_voice",
            "response_format": "wav",
        },
    )

    assert response.status_code == 404
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "voice"
    assert payload["error"]["code"] == "voice_not_found"


def test_batch_speech_preserves_order_and_normalizes_text(tmp_path: Path) -> None:
    engine = BatchFakeEngine()
    client = TestClient(create_app(make_settings(tmp_path), engine))

    voice_response = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Batch voice",
        },
    )
    voice_id = voice_response.json()["id"]
    response = client.post(
        "/v1/audio/speech/batch",
        json={
            "model": "spark-tts",
            "inputs": ["我9:05到。", "增长12.5%。"],
            "voice": voice_id,
            "response_format": "wav",
            "speed": 1.0,
        },
    )

    assert response.status_code == 200, response.text
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-spark-tts-segments"] == "2"
    assert float(response.headers["duration"]) == 1.0
    assert float(response.headers["rtf"]) > 0
    assert response.headers["batch-size"] == "2"
    assert engine.batch_texts == [
        "我九点零五到。",
        "增长百分之十二点五。",
    ]


def test_single_speech_drops_an_internal_long_silence(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), LongSilenceFakeEngine()))
    voice_response = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Silence voice",
        },
    )
    voice_id = voice_response.json()["id"]

    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "A sentence with a decoder dropout.",
            "voice": voice_id,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200, response.text
    assert float(response.headers["duration"]) == 1.0
    decoded, _ = sf.read(BytesIO(response.content), dtype="float32")
    assert decoded.size == 16000


def test_batch_drops_a_fully_silent_segment(tmp_path: Path) -> None:
    client = TestClient(
        create_app(make_settings(tmp_path), SilentSegmentBatchFakeEngine())
    )
    voice_response = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Batch silence voice",
        },
    )
    voice_id = voice_response.json()["id"]

    response = client.post(
        "/v1/audio/speech/batch",
        json={
            "inputs": ["First segment.", "Second segment."],
            "voice": voice_id,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-spark-tts-segments"] == "1"
    assert float(response.headers["duration"]) == 0.5


def test_long_text_splits_and_uses_batch_engine(tmp_path: Path) -> None:
    engine = BatchFakeEngine()
    client = TestClient(create_app(make_settings(tmp_path), engine))

    voice_response = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={
            "prompt_text": "This is the reference transcript.",
            "name": "Long text voice",
        },
    )
    voice_id = voice_response.json()["id"]
    text = (
        "第 1 段\u200b增长28.64%，用于验证长文本接口按标点拆分。\r\n"
        "I need to cancel my appointment for tomorrow.\r\n"
    ) * 8
    response = client.post(
        "/v1/audio/long_text",
        json={
            "model": "spark-tts",
            "input": text,
            "voice": voice_id,
            "response_format": "wav",
            "speed": 1.0,
            "min_segment_chars": 32,
            "max_segment_chars": 128,
        },
    )

    assert response.status_code == 200, response.text
    assert response.content.startswith(b"RIFF")
    assert int(response.headers["x-spark-tts-segments"]) > 1
    assert engine.batch_texts is not None
    normalized = normalize_text(text)
    assert "I need to cancel my appointment for tomorrow." in normalized
    expected_segments = [
        to_koubo_text(segment)
        for segment in split_text_by_punctuation(
            normalized,
            min_chars=32,
            max_chars=128,
        )
    ]
    assert engine.batch_texts == expected_segments
    assert all("\u200b" not in segment for segment in engine.batch_texts)
    assert all("\r" not in segment and "\n" not in segment for segment in engine.batch_texts)
    assert all("  " not in segment for segment in engine.batch_texts)
    assert all("28.64%" not in segment for segment in engine.batch_texts)
    assert float(response.headers["duration"]) > 0
    assert float(response.headers["rtf"]) > 0
    assert response.headers["batch-size"] == str(min(4, len(expected_segments)))


def test_long_text_rejects_more_than_4096_characters(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), BatchFakeEngine()))
    response = client.post(
        "/v1/audio/long_text",
        json={
            "input": "字" * 4097,
            "voice": "some_voice",
            "response_format": "wav",
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "input_too_long"


def test_validation_errors_use_openai_error_shape(tmp_path: Path) -> None:
    client = TestClient(create_app(make_settings(tmp_path), FakeEngine()))
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "spark-tts",
            "input": "Hello.",
            "voice": "some_voice",
            "speed": 9,
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "speed"


def test_api_key_can_protect_openai_endpoint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    settings = Settings(**{**settings.__dict__, "api_key": "test-secret"})
    client = TestClient(create_app(settings, FakeEngine()))

    response = client.get("/api/voices")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"

    authorized = client.get(
        "/api/voices",
        headers={"Authorization": "Bearer test-secret"},
    )
    assert authorized.status_code == 200
    assert authorized.json() == {"object": "list", "data": []}


def test_voice_metadata_is_persisted(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path), FakeEngine())
    client = TestClient(app)
    response = client.post(
        "/api/voices",
        files={"audio": ("reference.wav", wav_bytes(), "audio/wav")},
        data={"prompt_text": "This is the reference transcript.", "name": "Persisted"},
    )
    voice_id = response.json()["id"]
    metadata = json.loads(
        (tmp_path / "voices" / voice_id / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["name"] == "Persisted"
    assert metadata["prompt_text"] == "This is the reference transcript."
