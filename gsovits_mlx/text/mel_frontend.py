"""Mel/spectrogram front-end (module/mel_processing.py) in pure MLX.

spectrogram_torch: reflect-pad (n_fft-hop)/2, STFT hann, magnitude, (optionally mel
projection + ln). spectral_normalize: ln(clamp(spec, min=1e-5) * 1) [dynamic range
compression C=1, clip_factor 1.0].
"""

from __future__ import annotations

import math

import numpy as np
import mlx.core as mx


def _hann_window(n: int) -> np.ndarray:
    return np.hanning(n + 1).astype(np.float32)[:n]  # torch.hann_window periodic=True


def _reflect_pad(x: np.ndarray, left: int, right: int) -> np.ndarray:
    # torch F.pad reflect (1D numpy, last axis)
    return np.pad(x, ((0, 0), (left, right)), mode="reflect")


def _librosa_mel(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    """librosa.filters.mel equivalent (slaney scale + slaney norm)."""
    def _hz_to_mel(f):
        f_min = 0.0
        f_sp = 200.0 / 3
        mels = (f - f_min) / f_sp
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(f >= min_log_hz, min_log_mel + np.log(f / min_log_hz) / logstep, mels)

    def _mel_to_hz(m):
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * m
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = np.log(6.4) / 27.0
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), freqs)

    fmin_mel = _hz_to_mel(fmin)
    fmax_mel = _hz_to_mel(sr / 2 if fmax is None else fmax)
    mels = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
    freqs = _mel_to_hz(mels)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - left) / max(center - left, 1e-10)
        down = (right - fft_freqs) / max(right - center, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, down))
    # slaney norm
    enorm = 2.0 / (freqs[2:n_mels + 2] - freqs[:n_mels])
    fb *= enorm[:, None]
    return fb


def stft_magnitude(y: mx.array, n_fft: int, hop_size: int, win_size: int,
                   center: bool = False) -> mx.array:
    """y (B, T) -> magnitude (B, n_fft//2+1, T')."""
    x = np.array(y.astype(mx.float32), dtype=np.float32)
    pad = (n_fft - hop_size) // 2
    if pad:
        x = _reflect_pad(x, pad, pad)
    win = _hann_window(win_size)
    b, t = x.shape
    n_frames = 1 + (t - n_fft) // hop_size if t >= n_fft else 0
    idx = np.arange(n_fft)[None, :] + hop_size * np.arange(n_frames)[:, None]
    frames = x[:, idx] * win[None, None, :]  # (B, T', n_fft)
    spec = np.fft.rfft(frames, axis=-1)
    mag = np.sqrt(spec.real**2 + spec.imag**2 + 1e-8)
    return mx.array(mag.transpose(0, 2, 1))  # (B, F, T')


def spectrogram(y: mx.array, n_fft: int, hop_size: int, win_size: int,
                center: bool = False) -> mx.array:
    return stft_magnitude(y, n_fft, hop_size, win_size, center)


def spectral_normalize(spec: mx.array, C: float = 1.0) -> mx.array:
    """torch spectral_normalize_torch: ln(max(spec, clip))*C, clip 1e-5."""
    return mx.log(mx.maximum(spec * C, 1e-5))


def mel_spectrogram(y: mx.array, n_fft: int, num_mels: int, sampling_rate: int,
                    hop_size: int, win_size: int, fmin: float = 0.0,
                    fmax: float | None = None, center: bool = False) -> mx.array:
    """(B, T) audio -> log-mel (B, num_mels, T'). Matches module.mel_processing."""
    spec = stft_magnitude(y, n_fft, hop_size, win_size, center)
    mel_fb = _librosa_mel(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel = mx.array(mel_fb) @ spec
    return spectral_normalize(mel)
