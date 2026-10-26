"""FastAPI adapter that exposes Spark-TTS through an OpenAI-compatible TTS API."""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from api.audio_cleaning import clean_generated_waveform
from api.long_text import normalize_text, split_text_by_punctuation
from api.zh_oral_utils import to_koubo_text

logger = logging.getLogger("spark_tts_api")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
SUPPORTED_AUDIO_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}
VOICE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class APIError(Exception):
    """An error that is serialized using OpenAI's error response shape."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str,
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code


class AudioProcessingError(Exception):
    """Raised when uploaded audio cannot be converted or validated."""


class VoiceUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(default=None, max_length=120)
    prompt_text: str | None = None


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    voice_dir: Path
    device: str = "cpu"
    model_id: str = "spark-tts"
    api_key: str | None = None
    max_upload_mb: int = 20
    max_reference_seconds: float = 30.0
    max_text_chars: int = 2000
    max_long_text_chars: int = 4096
    max_batch_items: int = 128
    ffmpeg_bin: str = "ffmpeg"
    torch_threads: int | None = None
    llm_dtype: str = "auto"
    prompt_cache_size: int = 16
    preload_model: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        torch_threads = os.getenv("SPARK_TTS_TORCH_THREADS")
        return cls(
            model_dir=Path(
                os.getenv(
                    "SPARK_TTS_MODEL_DIR",
                    str(PROJECT_ROOT / "pretrained_models" / "Spark-TTS-0.5B"),
                )
            ).expanduser(),
            voice_dir=Path(
                os.getenv("SPARK_TTS_VOICE_DIR", str(PROJECT_ROOT / "data" / "voices"))
            ).expanduser(),
            device=os.getenv("SPARK_TTS_DEVICE", "cpu"),
            model_id=os.getenv("SPARK_TTS_MODEL_ID", "spark-tts"),
            api_key=os.getenv("SPARK_TTS_API_KEY") or None,
            max_upload_mb=int(os.getenv("SPARK_TTS_MAX_UPLOAD_MB", "20")),
            max_reference_seconds=float(
                os.getenv("SPARK_TTS_MAX_REFERENCE_SECONDS", "30")
            ),
            max_text_chars=int(os.getenv("SPARK_TTS_MAX_TEXT_CHARS", "2000")),
            max_long_text_chars=int(
                os.getenv("SPARK_TTS_MAX_LONG_TEXT_CHARS", "4096")
            ),
            max_batch_items=int(os.getenv("SPARK_TTS_MAX_BATCH_ITEMS", "128")),
            ffmpeg_bin=os.getenv("SPARK_TTS_FFMPEG", "ffmpeg"),
            torch_threads=int(torch_threads) if torch_threads else None,
            llm_dtype=os.getenv("SPARK_TTS_LLM_DTYPE", "auto"),
            prompt_cache_size=int(os.getenv("SPARK_TTS_PROMPT_CACHE_SIZE", "16")),
            preload_model=_env_bool("SPARK_TTS_PRELOAD", True),
        )


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "spark-tts"
    input: str
    voice: str = Field(
        min_length=1,
        max_length=120,
        description="Voice ID or unique display name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class BatchSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "spark-tts"
    inputs: list[str] = Field(min_length=1, max_length=128)
    voice: str = Field(
        min_length=1,
        max_length=120,
        description="Voice ID or unique display name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "wav"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class LongTextRequest(SpeechRequest):
    min_segment_chars: int = Field(default=32, ge=1, le=512)
    max_segment_chars: int = Field(default=128, ge=1, le=512)


def _openai_error(
    message: str,
    *,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


def _find_executable(name_or_path: str) -> str | None:
    if os.path.sep in name_or_path:
        path = Path(name_or_path).expanduser()
        return str(path) if path.is_file() else None
    return shutil.which(name_or_path)


def _run_ffmpeg(command: list[str], *, action: str) -> None:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioProcessingError(f"{action}失败：ffmpeg 无法运行。") from exc

    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioProcessingError(
            f"{action}失败。请确认参考音频格式有效。"
            + (f" ffmpeg: {detail[-400:]}" if detail else "")
        )


def _transcode_with_soundfile(source: Path, destination: Path) -> None:
    try:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioProcessingError(
            "无法读取参考音频。请上传 WAV、MP3、M4A、OGG 或 WebM 文件。"
        ) from exc

    if audio.size == 0:
        raise AudioProcessingError("参考音频为空。")

    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        try:
            import soxr

            audio = soxr.resample(audio, sample_rate, 16000)
        except ImportError:
            duration = audio.shape[0] / sample_rate
            target_size = max(1, round(duration * 16000))
            source_positions = np.linspace(0, audio.shape[0] - 1, audio.shape[0])
            target_positions = np.linspace(0, audio.shape[0] - 1, target_size)
            audio = np.interp(target_positions, source_positions, audio).astype(
                np.float32
            )

    sf.write(destination, audio, 16000, format="WAV", subtype="PCM_16")


def _transcode_reference(source: Path, destination: Path, settings: Settings) -> float:
    ffmpeg = _find_executable(settings.ffmpeg_bin)
    if ffmpeg:
        _run_ffmpeg(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            action="转换参考音频",
        )
    else:
        _transcode_with_soundfile(source, destination)

    try:
        info = sf.info(destination)
    except Exception as exc:
        raise AudioProcessingError("转换后的参考音频无法读取。") from exc

    duration = info.frames / info.samplerate if info.samplerate else 0.0
    if duration <= 0:
        raise AudioProcessingError("参考音频为空。")
    if duration > settings.max_reference_seconds:
        raise AudioProcessingError(
            f"参考音频过长：{duration:.1f} 秒，最大允许 "
            f"{settings.max_reference_seconds:.0f} 秒。"
        )
    return duration


@dataclass(frozen=True)
class VoiceRecord:
    id: str
    name: str
    prompt_text: str
    created_at: str | None
    duration_seconds: float | None
    directory: Path

    @property
    def reference_audio(self) -> Path:
        return self.directory / "reference.wav"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prompt_text": self.prompt_text,
            "created_at": self.created_at,
            "duration_seconds": self.duration_seconds,
        }


class VoiceStore:
    """Filesystem-backed voice store with ID and unique-name lookup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.voice_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._by_id: dict[str, VoiceRecord] = {}
        self._ids_by_name: dict[str, set[str]] = {}
        self.refresh_index()

    @staticmethod
    def _normalized_name(name: str) -> str:
        return name.strip().casefold()

    def _read_record(self, voice_dir: Path) -> VoiceRecord | None:
        metadata_path = voice_dir / "metadata.json"
        reference_audio = voice_dir / "reference.wav"
        if not voice_dir.is_dir() or not metadata_path.is_file():
            return None
        if not reference_audio.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Ignoring invalid voice metadata at %s", metadata_path)
            return None
        name = str(metadata.get("name") or voice_dir.name).strip()
        try:
            duration = (
                float(metadata["duration_seconds"])
                if metadata.get("duration_seconds") is not None
                else None
            )
        except (TypeError, ValueError):
            duration = None
        return VoiceRecord(
            id=voice_dir.name,
            name=name,
            prompt_text=str(metadata.get("prompt_text") or ""),
            created_at=metadata.get("created_at"),
            duration_seconds=duration,
            directory=voice_dir,
        )

    def refresh_index(self) -> None:
        """Rebuild the startup name index from metadata directories."""
        with self._lock:
            by_id: dict[str, VoiceRecord] = {}
            ids_by_name: dict[str, set[str]] = {}
            for item in self.settings.voice_dir.iterdir():
                record = self._read_record(item)
                if record is None:
                    continue
                by_id[record.id] = record
                ids_by_name.setdefault(
                    self._normalized_name(record.name),
                    set(),
                ).add(record.id)
            self._by_id = by_id
            self._ids_by_name = ids_by_name

    def _find(self, voice: str) -> VoiceRecord:
        with self._lock:
            record = self._by_id.get(voice)
            if record is not None:
                return record
            name_key = self._normalized_name(voice)
            ids = set(self._ids_by_name.get(name_key, set()))
            if not ids:
                self.refresh_index()
                record = self._by_id.get(voice)
                if record is not None:
                    return record
                ids = set(self._ids_by_name.get(name_key, set()))
            if len(ids) == 1:
                return self._by_id[next(iter(ids))]
            if len(ids) > 1:
                raise APIError(
                    409,
                    f"声音名称“{voice}”对应多个 ID，请改用 voice ID："
                    + ", ".join(sorted(ids)),
                    error_type="invalid_request_error",
                    param="voice",
                    code="voice_name_ambiguous",
                )
        raise APIError(
            404,
            f"找不到声音“{voice}”。请先通过 /api/voices 上传参考音频。",
            error_type="invalid_request_error",
            param="voice",
            code="voice_not_found",
        )

    def _validate_name(self, name: str, *, exclude_id: str | None = None) -> str:
        cleaned = name.strip()
        if not cleaned:
            raise APIError(
                400,
                "声音名称不能为空。",
                error_type="invalid_request_error",
                param="name",
                code="empty_voice_name",
            )
        if len(cleaned) > 120:
            raise APIError(
                400,
                "声音名称不能超过 120 个字符。",
                error_type="invalid_request_error",
                param="name",
                code="voice_name_too_long",
            )
        ids = self._ids_by_name.get(self._normalized_name(cleaned), set())
        if ids - ({exclude_id} if exclude_id else set()):
            raise APIError(
                409,
                f"声音名称“{cleaned}”已存在，请使用其它名称。",
                error_type="invalid_request_error",
                param="name",
                code="duplicate_voice_name",
            )
        return cleaned

    def _available_default_name(self, filename: str) -> str:
        base = Path(filename).name.strip() or "未命名声音"
        if len(base) > 120:
            suffix = Path(base).suffix
            keep = max(1, 120 - len(suffix))
            base = base[:keep] + suffix
        candidate = base
        index = 2
        while self._ids_by_name.get(self._normalized_name(candidate)):
            suffix = f" ({index})"
            stem = Path(base).stem
            extension = Path(base).suffix
            available_stem = 120 - len(suffix) - len(extension)
            candidate = stem[: max(1, available_stem)] + suffix + extension
            index += 1
        return candidate

    @staticmethod
    def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def get(self, voice: str) -> VoiceRecord:
        return self._find(voice)

    def resolve(self, voice: str) -> tuple[Path, str | None]:
        record = self._find(voice)
        return record.reference_audio, record.prompt_text or None

    def list_voices(self) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._by_id.values())
        records.sort(key=lambda item: item.created_at or "", reverse=True)
        return [record.to_dict() for record in records]

    def create(
        self,
        source: Path,
        *,
        name: str,
        prompt_text: str,
        default_name: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            cleaned_name = (
                self._validate_name(name)
                if name.strip()
                else self._available_default_name(default_name or "")
            )
            voice_id = f"voice_{uuid.uuid4().hex[:12]}"
            staging_dir = self.settings.voice_dir / f".staging_{uuid.uuid4().hex}"
            final_dir = self.settings.voice_dir / voice_id
            staging_dir.mkdir(parents=True)

            try:
                duration = _transcode_reference(
                    source, staging_dir / "reference.wav", self.settings
                )
                metadata = {
                    "id": voice_id,
                    "name": cleaned_name,
                    "prompt_text": prompt_text.strip(),
                    "created_at": datetime.now(UTC).isoformat(),
                    "duration_seconds": round(duration, 3),
                }
                self._write_metadata(staging_dir / "metadata.json", metadata)
                staging_dir.rename(final_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
            self.refresh_index()
            return self._by_id[voice_id].to_dict()

    def update_metadata(
        self,
        voice: str,
        *,
        name: str | None = None,
        prompt_text: str | None = None,
    ) -> VoiceRecord:
        with self._lock:
            record = self._find(voice)
            metadata_path = record.directory / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if name is not None:
                if self._normalized_name(name) == self._normalized_name(record.name):
                    cleaned_name = record.name
                else:
                    cleaned_name = self._validate_name(
                        name,
                        exclude_id=record.id,
                    )
                metadata["name"] = cleaned_name
            if prompt_text is not None:
                metadata["prompt_text"] = prompt_text.strip()
            self._write_metadata(metadata_path, metadata)
            self.refresh_index()
            return self._by_id[record.id]

    def replace_audio(self, voice: str, source: Path) -> VoiceRecord:
        with self._lock:
            record = self._find(voice)
            temporary = record.directory / f".reference-{uuid.uuid4().hex}.wav"
            try:
                duration = _transcode_reference(source, temporary, self.settings)
                os.replace(temporary, record.reference_audio)
                metadata_path = record.directory / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["duration_seconds"] = round(duration, 3)
                self._write_metadata(metadata_path, metadata)
            finally:
                temporary.unlink(missing_ok=True)
            self.refresh_index()
            return self._by_id[record.id]

    def delete(self, voice: str) -> None:
        with self._lock:
            record = self._find(voice)
            root = self.settings.voice_dir.resolve()
            target = record.directory.resolve()
            if target.parent != root:
                raise APIError(
                    400,
                    "无效的声音目录。",
                    error_type="invalid_request_error",
                    param="voice",
                    code="invalid_voice",
                )
            shutil.rmtree(target)
            self.refresh_index()


class SparkTTSEngine:
    """Lazily loaded, serialized CPU/GPU inference wrapper."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._device: str | None = None
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def llm_dtype(self) -> str:
        if self._model is None:
            return self.settings.llm_dtype
        return str(self._model.llm_dtype)

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        if not self.settings.model_dir.is_dir():
            raise FileNotFoundError(
                f"Spark-TTS model directory does not exist: {self.settings.model_dir}"
            )

        import torch

        from cli.SparkTTS import SparkTTS

        requested_device = self.settings.device.strip().lower()
        if requested_device == "auto":
            if torch.cuda.is_available():
                requested_device = "cuda:0"
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                requested_device = "mps"
            else:
                requested_device = "cpu"

        if self.settings.torch_threads:
            torch.set_num_threads(self.settings.torch_threads)

        logger.info(
            "Loading Spark-TTS model from %s on %s",
            self.settings.model_dir,
            requested_device,
        )
        self._device = requested_device
        self._model = SparkTTS(
            self.settings.model_dir,
            torch.device(requested_device),
            llm_dtype=self.settings.llm_dtype,
            prompt_cache_size=self.settings.prompt_cache_size,
        )
        if requested_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(requested_device))
        logger.info("Spark-TTS model loaded")
        return self._model

    def preload(self) -> None:
        """Load the model during application startup instead of first request."""
        with self._lock:
            self._load()

    def synthesize(
        self,
        text: str,
        reference_audio: Path,
        prompt_text: str | None,
    ) -> np.ndarray:
        with self._lock:
            started_at = time.perf_counter()
            model = self._load()
            waveform = model.inference(
                text=text,
                prompt_speech_path=str(reference_audio),
                prompt_text=prompt_text,
            )

        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()
        audio = clean_generated_waveform(waveform)
        if audio.size == 0:
            raise RuntimeError("Spark-TTS returned fully silent audio")
        elapsed = time.perf_counter() - started_at
        audio_seconds = audio.size / 16000
        logger.info(
            "Synthesized %.2fs audio in %.2fs (RTF %.3f, text_chars=%d)",
            audio_seconds,
            elapsed,
            elapsed / audio_seconds if audio_seconds else 0.0,
            len(text),
        )
        return audio


def _atempo_filter(speed: float) -> str:
    """Build an ffmpeg filter chain for speeds outside atempo's 0.5-2.0 range."""

    factors: list[str] = []
    remaining = speed
    while remaining < 0.5:
        factors.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        factors.append("atempo=2.0")
        remaining /= 2.0
    factors.append(f"atempo={remaining:.6f}")
    return ",".join(factors)


def _pcm16_bytes(waveform: np.ndarray) -> bytes:
    audio = np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def encode_audio(
    waveform: np.ndarray,
    sample_rate: int,
    response_format: str,
    speed: float,
    settings: Settings,
) -> bytes:
    """Encode Spark-TTS waveform bytes in an OpenAI TTS response format."""

    if response_format not in SUPPORTED_AUDIO_FORMATS:
        raise AudioProcessingError(f"不支持的音频格式：{response_format}")

    raw_pcm = _pcm16_bytes(waveform)
    if response_format == "pcm" and speed == 1.0:
        return raw_pcm
    if response_format == "wav" and speed == 1.0:
        output = BytesIO()
        sf.write(
            output,
            np.asarray(waveform, dtype=np.float32).reshape(-1),
            sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        return output.getvalue()

    ffmpeg = _find_executable(settings.ffmpeg_bin)
    if not ffmpeg:
        raise AudioProcessingError(
            "当前服务未安装 ffmpeg，无法生成该格式或调整语速。"
        )

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
    ]
    if speed != 1.0:
        command.extend(["-filter:a", _atempo_filter(speed)])

    if response_format == "mp3":
        command.extend(["-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k"])
    elif response_format == "opus":
        command.extend(["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"])
    elif response_format == "aac":
        command.extend(["-f", "adts", "-c:a", "aac", "-b:a", "128k"])
    elif response_format == "flac":
        command.extend(["-f", "flac", "-c:a", "flac"])
    elif response_format == "wav":
        command.extend(["-f", "wav", "-c:a", "pcm_s16le"])
    else:
        command.extend(["-f", "s16le"])

    command.append("pipe:1")
    try:
        result = subprocess.run(
            command,
            input=raw_pcm,
            check=False,
            capture_output=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioProcessingError("音频编码失败：ffmpeg 无法运行。") from exc

    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioProcessingError(
            "音频编码失败。"
            + (f" ffmpeg: {detail[-400:]}" if detail else "")
        )
    return result.stdout


def create_app(
    settings: Settings | None = None,
    engine: SparkTTSEngine | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = engine or SparkTTSEngine(settings)
    voice_store = VoiceStore(settings)

    def synthesize_texts(
        texts: list[str],
        reference_audio: Path,
        prompt_text: str | None,
    ) -> list[np.ndarray]:
        batch_method = getattr(engine, "synthesize_batch", None)
        if callable(batch_method):
            waveforms = batch_method(texts, reference_audio, prompt_text)
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(texts))) as executor:
                futures = [
                    executor.submit(
                        engine.synthesize,
                        text,
                        reference_audio,
                        prompt_text,
                    )
                    for text in texts
                ]
                waveforms = [future.result() for future in futures]

        cleaned = [
            audio
            for waveform in waveforms
            if (
                audio := clean_generated_waveform(waveform)
            ).size
        ]
        if not cleaned:
            raise RuntimeError("Spark-TTS returned fully silent audio for every segment")
        return cleaned

    def combine_waveforms(waveforms: list[np.ndarray]) -> np.ndarray:
        non_empty = [
            np.asarray(waveform, dtype=np.float32).reshape(-1)
            for waveform in waveforms
            if np.asarray(waveform).size
        ]
        if not non_empty:
            raise RuntimeError("speech generation returned no audio")
        return np.concatenate(non_empty)

    def telemetry_headers(
        *,
        audio_seconds: float,
        inference_started_at: float,
        speed_ratio: float,
        batch_size: int,
        voice: str,
        segments: int | None = None,
    ) -> dict[str, str]:
        elapsed = time.perf_counter() - inference_started_at
        headers = {
            "Cache-Control": "no-store",
            "X-Spark-TTS-Voice": voice,
            "X-Spark-TTS-Model": settings.model_id,
            "X-Spark-TTS-Sample-Rate": "16000",
            "duration": f"{audio_seconds / max(speed_ratio, 0.001):.3f}",
            "rtf": f"{elapsed / audio_seconds:.4f}",
            "batch-size": str(batch_size),
        }
        if segments is not None:
            headers["X-Spark-TTS-Segments"] = str(segments)
        return headers

    async def stage_audio_upload(audio: UploadFile) -> tuple[Path, Path]:
        suffix = Path(audio.filename or "").suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = ".bin"
        temp_dir = Path(tempfile.mkdtemp(prefix="spark-tts-upload-"))
        source = temp_dir / f"reference{suffix}"
        max_bytes = settings.max_upload_mb * 1024 * 1024
        try:
            size = 0
            with source.open("wb") as output:
                while chunk := await audio.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise APIError(
                            413,
                            f"参考音频不能超过 {settings.max_upload_mb} MB。",
                            error_type="invalid_request_error",
                            param="audio",
                            code="upload_too_large",
                        )
                    output.write(chunk)
            if size == 0:
                raise APIError(
                    400,
                    "参考音频为空。",
                    error_type="invalid_request_error",
                    param="audio",
                    code="empty_upload",
                )
        except Exception:
            await audio.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        return temp_dir, source

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload_model:
            preload = getattr(engine, "preload", None)
            if callable(preload):
                logger.info("Preloading Spark-TTS model")
                try:
                    await run_in_threadpool(preload)
                except Exception:
                    # Keep the HTTP service available so health checks and API
                    # errors can report a model-initialization failure.
                    logger.exception("Failed to preload Spark-TTS model")
        yield

    app = FastAPI(
        title="Spark-TTS OpenAI-compatible API",
        description=(
            "OpenAI-compatible speech synthesis with Spark-TTS voice cloning. "
            "Upload a reference voice through /api/voices, then use its ID or "
            "unique display name as the OpenAI voice parameter."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def require_api_key(
        authorization: Annotated[str | None, Header()] = None,
    ) -> None:
        if not settings.api_key:
            return
        supplied_key = (
            authorization.removeprefix("Bearer ").strip()
            if authorization and authorization.startswith("Bearer ")
            else ""
        )
        if not hmac.compare_digest(supplied_key, settings.api_key):
            raise APIError(
                401,
                "Invalid API key.",
                error_type="authentication_error",
                code="invalid_api_key",
            )

    @app.exception_handler(APIError)
    async def handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_openai_error(
                exc.message,
                error_type=exc.error_type,
                param=exc.param,
                code=exc.code,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = [
            str(part)
            for part in first_error.get("loc", [])
            if part not in {"body", "query", "path"}
        ]
        param = ".".join(location) or None
        message = first_error.get("msg", "Invalid request.")
        return JSONResponse(
            status_code=400,
            content=_openai_error(
                message,
                error_type="invalid_request_error",
                param=param,
                code="invalid_request",
            ),
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/voices", include_in_schema=False)
    async def voices_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "voices.html")

    @app.get("/healthz", tags=["system"])
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": settings.model_id,
            "model_loaded": engine.loaded,
            "device": engine.device or settings.device,
            "llm_dtype": getattr(engine, "llm_dtype", settings.llm_dtype),
            "prompt_cache_size": settings.prompt_cache_size,
            "auth_required": bool(settings.api_key),
        }

    @app.get("/v1/models", tags=["OpenAI compatibility"])
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "sparkaudio",
                }
            ],
        }

    @app.post(
        "/api/voices",
        status_code=201,
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def create_voice(
        audio: Annotated[UploadFile, File(description="Reference voice audio")],
        prompt_text: Annotated[
            str, Form(description="Transcript of the reference audio")
        ] = "",
        name: Annotated[str, Form(description="Display name")] = "",
    ) -> dict[str, Any]:
        if len(prompt_text) > settings.max_text_chars:
            raise APIError(
                400,
                f"参考音频文本不能超过 {settings.max_text_chars} 个字符。",
                error_type="invalid_request_error",
                param="prompt_text",
                code="prompt_text_too_long",
            )

        temp_dir, source = await stage_audio_upload(audio)
        try:
            try:
                voice = await run_in_threadpool(
                    voice_store.create,
                    source,
                    name=name,
                    prompt_text=prompt_text,
                    default_name=Path(audio.filename or "").name,
                )
            except AudioProcessingError as exc:
                raise APIError(
                    400,
                    str(exc),
                    error_type="invalid_request_error",
                    param="audio",
                    code="invalid_audio",
                ) from exc
            return voice
        finally:
            await audio.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.get(
        "/api/voices",
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def list_voices() -> dict[str, Any]:
        return {"object": "list", "data": voice_store.list_voices()}

    @app.get(
        "/api/voices/{voice}",
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def get_voice(voice: str) -> dict[str, Any]:
        return voice_store.get(voice).to_dict()

    @app.get(
        "/api/voices/{voice}/audio",
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def get_voice_audio(voice: str) -> FileResponse:
        record = voice_store.get(voice)
        return FileResponse(
            record.reference_audio,
            media_type="audio/wav",
            filename=f"{record.id}.wav",
        )

    @app.patch(
        "/api/voices/{voice}",
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def update_voice(
        voice: str,
        payload: VoiceUpdateRequest,
    ) -> dict[str, Any]:
        updates = payload.model_dump(exclude_unset=True)
        if "name" in updates and updates["name"] is None:
            raise APIError(
                400,
                "name 不能为 null。",
                error_type="invalid_request_error",
                param="name",
                code="invalid_voice_name",
            )
        prompt_text = updates.get("prompt_text")
        if prompt_text is not None and len(prompt_text) > settings.max_text_chars:
            raise APIError(
                400,
                f"参考文本不能超过 {settings.max_text_chars} 个字符。",
                error_type="invalid_request_error",
                param="prompt_text",
                code="prompt_text_too_long",
            )
        record = await run_in_threadpool(
            voice_store.update_metadata,
            voice,
            name=updates.get("name"),
            prompt_text=prompt_text,
        )
        return record.to_dict()

    @app.put(
        "/api/voices/{voice}/audio",
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def replace_voice_audio(
        voice: str,
        audio: Annotated[UploadFile, File(description="Replacement reference audio")],
    ) -> dict[str, Any]:
        temp_dir, source = await stage_audio_upload(audio)
        try:
            try:
                record = await run_in_threadpool(
                    voice_store.replace_audio,
                    voice,
                    source,
                )
            except AudioProcessingError as exc:
                raise APIError(
                    400,
                    str(exc),
                    error_type="invalid_request_error",
                    param="audio",
                    code="invalid_audio",
                ) from exc
            return record.to_dict()
        finally:
            await audio.close()
            shutil.rmtree(temp_dir, ignore_errors=True)

    @app.delete(
        "/api/voices/{voice}",
        status_code=204,
        dependencies=[Depends(require_api_key)],
        tags=["voice management"],
    )
    async def delete_voice(voice: str) -> Response:
        await run_in_threadpool(voice_store.delete, voice)
        return Response(status_code=204)

    @app.post(
        "/v1/audio/speech",
        dependencies=[Depends(require_api_key)],
        tags=["OpenAI compatibility"],
        response_class=Response,
        responses={
            200: {
                "content": {
                    media_type: {}
                    for media_type in AUDIO_MEDIA_TYPES.values()
                },
                "description": "Generated audio bytes.",
            }
        },
    )
    async def create_speech(payload: SpeechRequest) -> Response:
        text = payload.input.strip()
        if not text:
            raise APIError(
                400,
                "input 不能为空。",
                error_type="invalid_request_error",
                param="input",
                code="empty_input",
            )
        if len(text) > settings.max_text_chars:
            raise APIError(
                400,
                f"input 不能超过 {settings.max_text_chars} 个字符。",
                error_type="invalid_request_error",
                param="input",
                code="input_too_long",
            )

        oral_text = to_koubo_text(text)
        if oral_text != text:
            logger.info(
                "Applied Chinese oral transcription (%d -> %d chars)",
                len(text),
                len(oral_text),
            )
        text = oral_text

        reference_audio, prompt_text = voice_store.resolve(payload.voice)
        inference_started_at = time.perf_counter()
        try:
            waveform = await run_in_threadpool(
                engine.synthesize,
                text,
                reference_audio,
                prompt_text,
            )
            waveform = clean_generated_waveform(waveform)
            if waveform.size == 0:
                raise RuntimeError("Spark-TTS returned fully silent audio")
        except FileNotFoundError as exc:
            logger.exception("Spark-TTS model is unavailable")
            raise APIError(
                503,
                "Spark-TTS 模型尚未就绪，请检查模型目录和服务日志。",
                error_type="server_error",
                code="model_unavailable",
            ) from exc
        except Exception as exc:
            logger.exception("Speech generation failed")
            raise APIError(
                500,
                "语音生成失败，请检查服务日志。",
                error_type="server_error",
                code="speech_generation_failed",
            ) from exc

        audio_seconds = (
            np.asarray(waveform, dtype=np.float32).reshape(-1).size / 16000
        )
        try:
            output = await run_in_threadpool(
                encode_audio,
                waveform,
                16000,
                payload.response_format,
                payload.speed,
                settings,
            )
        except AudioProcessingError as exc:
            raise APIError(
                500,
                str(exc),
                error_type="server_error",
                code="audio_encoding_failed",
            ) from exc

        return Response(
            content=output,
            media_type=AUDIO_MEDIA_TYPES[payload.response_format],
            headers=telemetry_headers(
                audio_seconds=audio_seconds,
                inference_started_at=inference_started_at,
                speed_ratio=payload.speed,
                batch_size=1,
                voice=payload.voice,
            ),
        )

    @app.post(
        "/v1/audio/speech/batch",
        dependencies=[Depends(require_api_key)],
        tags=["OpenAI compatibility"],
        response_class=Response,
        responses={
            200: {
                "content": {
                    media_type: {}
                    for media_type in AUDIO_MEDIA_TYPES.values()
                },
                "description": "Generated audio bytes in input order.",
            }
        },
    )
    async def create_speech_batch(payload: BatchSpeechRequest) -> Response:
        if len(payload.inputs) > settings.max_batch_items:
            raise APIError(
                400,
                f"一次最多提交 {settings.max_batch_items} 个分段。",
                error_type="invalid_request_error",
                param="inputs",
                code="too_many_inputs",
            )

        texts: list[str] = []
        for index, raw_text in enumerate(payload.inputs):
            text = normalize_text(raw_text)
            if not text:
                raise APIError(
                    400,
                    f"inputs[{index}] 不能为空。",
                    error_type="invalid_request_error",
                    param=f"inputs.{index}",
                    code="empty_input",
                )
            if len(text) > settings.max_text_chars:
                raise APIError(
                    400,
                    f"inputs[{index}] 不能超过 {settings.max_text_chars} 个字符。",
                    error_type="invalid_request_error",
                    param=f"inputs.{index}",
                    code="input_too_long",
                )
            texts.append(to_koubo_text(text))

        if sum(len(text) for text in texts) > settings.max_long_text_chars:
            raise APIError(
                400,
                f"批量文本总长度不能超过 {settings.max_long_text_chars} 个字符。",
                error_type="invalid_request_error",
                param="inputs",
                code="batch_text_too_long",
            )

        reference_audio, prompt_text = voice_store.resolve(payload.voice)
        inference_started_at = time.perf_counter()
        try:
            waveforms = await run_in_threadpool(
                synthesize_texts,
                texts,
                reference_audio,
                prompt_text,
            )
        except FileNotFoundError as exc:
            logger.exception("Spark-TTS model is unavailable")
            raise APIError(
                503,
                "Spark-TTS 模型尚未就绪，请检查模型目录和服务日志。",
                error_type="server_error",
                code="model_unavailable",
            ) from exc
        except Exception as exc:
            logger.exception("Batch speech generation failed")
            raise APIError(
                500,
                "批量语音生成失败，请检查服务日志。",
                error_type="server_error",
                code="speech_generation_failed",
            ) from exc

        combined = combine_waveforms(waveforms)
        audio_seconds = combined.size / 16000
        try:
            output = await run_in_threadpool(
                encode_audio,
                combined,
                16000,
                payload.response_format,
                payload.speed,
                settings,
            )
        except AudioProcessingError as exc:
            raise APIError(
                500,
                str(exc),
                error_type="server_error",
                code="audio_encoding_failed",
            ) from exc

        return Response(
            content=output,
            media_type=AUDIO_MEDIA_TYPES[payload.response_format],
            headers=telemetry_headers(
                audio_seconds=audio_seconds,
                inference_started_at=inference_started_at,
                speed_ratio=payload.speed,
                batch_size=min(
                    int(getattr(engine, "max_batch_size", 4)),
                    len(texts),
                ),
                voice=payload.voice,
                segments=len(waveforms),
            ),
        )

    @app.post(
        "/v1/audio/long_text",
        dependencies=[Depends(require_api_key)],
        tags=["OpenAI compatibility"],
        response_class=Response,
        responses={
            200: {
                "content": {
                    media_type: {}
                    for media_type in AUDIO_MEDIA_TYPES.values()
                },
                "description": "Generated long-form audio bytes.",
            }
        },
    )
    async def create_long_speech(payload: LongTextRequest) -> Response:
        text = normalize_text(payload.input)
        if not text:
            raise APIError(
                400,
                "input 不能为空。",
                error_type="invalid_request_error",
                param="input",
                code="empty_input",
            )
        if len(text) > settings.max_long_text_chars:
            raise APIError(
                400,
                f"长文本不能超过 {settings.max_long_text_chars} 个字符。",
                error_type="invalid_request_error",
                param="input",
                code="input_too_long",
            )
        if payload.min_segment_chars > payload.max_segment_chars:
            raise APIError(
                400,
                "min_segment_chars 不能大于 max_segment_chars。",
                error_type="invalid_request_error",
                param="min_segment_chars",
                code="invalid_segment_range",
            )

        segments = split_text_by_punctuation(
            text,
            min_chars=payload.min_segment_chars,
            max_chars=payload.max_segment_chars,
        )
        if not segments:
            raise APIError(
                400,
                "input 中没有可合成的文本。",
                error_type="invalid_request_error",
                param="input",
                code="empty_input",
            )
        segments = [to_koubo_text(segment) for segment in segments]

        reference_audio, prompt_text = voice_store.resolve(payload.voice)
        inference_started_at = time.perf_counter()
        try:
            waveforms = await run_in_threadpool(
                synthesize_texts,
                segments,
                reference_audio,
                prompt_text,
            )
        except FileNotFoundError as exc:
            logger.exception("Spark-TTS model is unavailable")
            raise APIError(
                503,
                "Spark-TTS 模型尚未就绪，请检查模型目录和服务日志。",
                error_type="server_error",
                code="model_unavailable",
            ) from exc
        except Exception as exc:
            logger.exception("Long-text speech generation failed")
            raise APIError(
                500,
                "长文本语音生成失败，请检查服务日志。",
                error_type="server_error",
                code="speech_generation_failed",
            ) from exc

        combined = combine_waveforms(waveforms)
        audio_seconds = combined.size / 16000
        try:
            output = await run_in_threadpool(
                encode_audio,
                combined,
                16000,
                payload.response_format,
                payload.speed,
                settings,
            )
        except AudioProcessingError as exc:
            raise APIError(
                500,
                str(exc),
                error_type="server_error",
                code="audio_encoding_failed",
            ) from exc

        return Response(
            content=output,
            media_type=AUDIO_MEDIA_TYPES[payload.response_format],
            headers=telemetry_headers(
                audio_seconds=audio_seconds,
                inference_started_at=inference_started_at,
                speed_ratio=payload.speed,
                batch_size=min(
                    int(getattr(engine, "max_batch_size", 4)),
                    len(segments),
                ),
                voice=payload.voice,
                segments=len(segments),
            ),
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
