"""A small torchaudio-compatible MelSpectrogram fallback."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return torch.where(
        freq >= min_log_hz,
        min_log_mel + torch.log(freq / min_log_hz) / logstep,
        freq / f_sp,
    )


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return torch.where(
        mel >= min_log_mel,
        min_log_hz * torch.exp(logstep * (mel - min_log_mel)),
        f_sp * mel,
    )


def _create_slaney_fb_matrix(
    *,
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
) -> torch.Tensor:
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs, dtype=torch.float32)
    m_min = _hz_to_mel(torch.tensor(f_min, dtype=torch.float32))
    m_max = _hz_to_mel(torch.tensor(f_max, dtype=torch.float32))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2, dtype=torch.float32)
    f_pts = _mel_to_hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    ramps = f_pts.unsqueeze(1) - all_freqs.unsqueeze(0)

    lower = -ramps[:-2] / f_diff[:-1].unsqueeze(1)
    upper = ramps[2:] / f_diff[1:].unsqueeze(1)
    fb = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, upper))
    fb *= (2.0 / (f_pts[2:] - f_pts[:-2])).unsqueeze(1)
    return fb


class MelSpectrogram(nn.Module):
    """Match the torchaudio MelSpectrogram settings used by Spark-TTS."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        f_min: float,
        f_max: float | None,
        n_mels: int,
        power: float = 1.0,
        norm: str = "slaney",
        mel_scale: str = "slaney",
    ) -> None:
        super().__init__()
        if norm != "slaney" or mel_scale != "slaney":
            raise ValueError("the fallback supports only slaney normalization")
        if power != 1.0:
            raise ValueError("the fallback supports only magnitude power=1")

        self.n_fft = n_fft
        self.hop_length = hop_length
        max_freq = sample_rate // 2 if f_max is None else f_max
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer(
            "fb",
            _create_slaney_fb_matrix(
                n_freqs=n_fft // 2 + 1,
                f_min=f_min,
                f_max=max_freq,
                n_mels=n_mels,
                sample_rate=sample_rate,
            ),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(dtype=self.window.dtype)
        if self.window.numel() < self.n_fft:
            pad = self.n_fft - self.window.numel()
            window = F.pad(self.window, (pad // 2, pad - pad // 2))
        else:
            window = self.window
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        return torch.matmul(self.fb, spectrum)
