# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MODIFIED: Spark-TTS engineering optimization fork, 2026.
# See the repository Git history for the complete change record.

import logging
import math
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.reference_audio import reference_audio_for_prompt
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

logger = logging.getLogger(__name__)


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(
        self,
        model_dir: Path,
        device: torch.device = torch.device("cuda:0"),
        llm_dtype: str | torch.dtype | None = None,
        prompt_cache_size: int = 16,
        attn_implementation: str = "sdpa",
        llm_backend: object | None = None,
    ):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
            llm_dtype: LLM weight dtype. ``None`` or ``auto`` uses BF16 on CUDA
                and FP32 elsewhere.
            prompt_cache_size: Number of reference-audio token sets to cache.
            attn_implementation: Transformers attention implementation.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self.llm_dtype = self._resolve_llm_dtype(llm_dtype)
        self.prompt_cache_size = max(0, int(prompt_cache_size))
        self.attn_implementation = attn_implementation
        self._injected_llm_backend = llm_backend
        self._prompt_cache: OrderedDict[
            tuple[str, int, int], tuple[torch.Tensor, torch.Tensor, str, str]
        ] = OrderedDict()
        self._prompt_cache_lock = RLock()
        self._initialize_inference()

    def _resolve_llm_dtype(
        self, requested_dtype: str | torch.dtype | None
    ) -> torch.dtype:
        """Resolve the LLM dtype without upcasting CUDA models to FP32."""
        if requested_dtype is None or requested_dtype == "auto":
            if self.device.type != "cuda":
                return torch.float32
            compute_major, _ = torch.cuda.get_device_capability(self.device)
            return torch.bfloat16 if compute_major >= 8 else torch.float16
        if isinstance(requested_dtype, torch.dtype):
            return requested_dtype

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        try:
            return dtype_map[str(requested_dtype).lower()]
        except KeyError as exc:
            raise ValueError(
                "llm_dtype must be one of: auto, bfloat16, float16, float32"
            ) from exc

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        load_kwargs = {"torch_dtype": self.llm_dtype}
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation
        if self._injected_llm_backend is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                f"{self.model_dir}/LLM",
                **load_kwargs,
            )
            self.model.to(self.device).eval()
        else:
            self.model = self._injected_llm_backend
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)

        audio_config = load_config(f"{self.model_dir}/BiCodec/config.yaml")[
            "audio_tokenizer"
        ]
        self._semantic_token_start = self.tokenizer.convert_tokens_to_ids(
            "<|bicodec_semantic_0|>"
        )
        self._semantic_token_count = int(
            audio_config["quantizer"]["codebook_size"]
        )
        self._global_token_start = self.tokenizer.convert_tokens_to_ids(
            "<|bicodec_global_0|>"
        )
        self._global_token_count = math.prod(
            audio_config["speaker_encoder"]["fsq_levels"]
        )
        logger.info(
            "LLM loaded with dtype=%s attention=%s",
            self.llm_dtype,
            self.attn_implementation or "default",
        )

    @staticmethod
    def _format_audio_tokens(token_ids: torch.Tensor, token_type: str) -> str:
        """Format all token IDs after a single CPU transfer."""
        values = token_ids.detach().cpu().reshape(-1).tolist()
        return "".join(
            f"<|bicodec_{token_type}_{token_id}|>" for token_id in values
        )

    def _reference_token_cache_key(
        self, prompt_speech_path: Path
    ) -> tuple[str, int, int]:
        path = Path(prompt_speech_path).expanduser().resolve()
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def _get_reference_tokens(
        self, prompt_speech_path: Path
    ) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        """Return cached CPU token IDs and preformatted prompt token strings."""
        cache_key = self._reference_token_cache_key(prompt_speech_path)
        if self.prompt_cache_size:
            with self._prompt_cache_lock:
                cached = self._prompt_cache.get(cache_key)
                if cached is not None:
                    self._prompt_cache.move_to_end(cache_key)
                    return cached

        with reference_audio_for_prompt(prompt_speech_path) as prompt_audio:
            global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
                prompt_audio
            )
        # Keep tiny ID tensors on CPU. This also avoids one CUDA synchronization
        # per token while formatting the prompt strings.
        global_token_ids = global_token_ids.detach().cpu().clone()
        semantic_token_ids = semantic_token_ids.detach().cpu().clone()
        if global_token_ids.ndim == 1:
            global_token_ids = global_token_ids.unsqueeze(0)
        if semantic_token_ids.ndim == 1:
            semantic_token_ids = semantic_token_ids.unsqueeze(0)

        cached = (
            global_token_ids,
            semantic_token_ids,
            self._format_audio_tokens(global_token_ids, "global"),
            self._format_audio_tokens(semantic_token_ids, "semantic"),
        )
        if self.prompt_cache_size:
            with self._prompt_cache_lock:
                self._prompt_cache[cache_key] = cached
                self._prompt_cache.move_to_end(cache_key)
                while len(self._prompt_cache) > self.prompt_cache_size:
                    self._prompt_cache.popitem(last=False)
        return cached

    def clear_prompt_cache(self) -> None:
        """Clear cached reference-audio tokens."""
        with self._prompt_cache_lock:
            self._prompt_cache.clear()

    def _extract_audio_token_ids(
        self,
        generated_ids: torch.Tensor,
        token_start: int,
        token_count: int,
    ) -> torch.Tensor:
        """Extract contiguous BiCodec token IDs directly from generated IDs."""
        token_end = token_start + token_count
        mask = (generated_ids >= token_start) & (generated_ids < token_end)
        return (generated_ids[mask] - token_start).to(
            device=self.device, dtype=torch.long
        )

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        (
            global_token_ids,
            semantic_token_ids,
            global_tokens,
            semantic_tokens,
        ) = self._get_reference_tokens(prompt_speech_path)

        # Prepare the input tokens for the model
        if prompt_text is not None:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    @torch.inference_mode()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)

        # Generate speech using the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=3000,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = generated_ids[0, model_inputs.input_ids.shape[1] :]
        pred_semantic_ids = self._extract_audio_token_ids(
            generated_ids,
            self._semantic_token_start,
            self._semantic_token_count,
        ).unsqueeze(0)
        if pred_semantic_ids.numel() == 0:
            raise RuntimeError("Spark-TTS did not generate any semantic tokens")

        if gender is not None:
            global_token_ids = self._extract_audio_token_ids(
                generated_ids,
                self._global_token_start,
                self._global_token_count,
            ).unsqueeze(0)
            if global_token_ids.numel() == 0:
                raise RuntimeError("Spark-TTS did not generate any global tokens")

        # Convert semantic tokens back to waveform
        wav = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids,
        )

        return wav
