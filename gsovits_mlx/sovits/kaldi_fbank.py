"""Kaldi fbank (torchaudio.compliance.kaldi.fbank) — numpy port for the sv encoder.

Only the parameter set used by GPT_SoVITS sv.py: num_mel_bins=80, sample_frequency
16000, dither=0, frame_length 25 ms, frame_shift 10 ms, povey window, snip_edges,
remove_dc_offset, preemphasis 0.97, raw_energy, round_to_power_of_two, use_power,
use_log_fbank, low 20 / high 0(=nyquist). Reference: GPT_SoVITS/eres2net/kaldi.py.
"""
from __future__ import annotations

import math

import numpy as np


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + freq / 700.0)


def _inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def _get_mel_banks(num_bins: int, window_length_padded: int, sample_freq: float,
                   low_freq: float, high_freq: float) -> np.ndarray:
    assert num_bins > 3 and window_length_padded % 2 == 0
    num_fft_bins = window_length_padded // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / window_length_padded
    mel_low = _mel_scale(np.array(low_freq))
    mel_high = _mel_scale(np.array(high_freq))
    mel_delta = (mel_high - mel_low) / (num_bins + 1)
    bin_idx = np.arange(num_bins)[:, None]
    left = mel_low + bin_idx * mel_delta
    center = mel_low + (bin_idx + 1.0) * mel_delta
    right = mel_low + (bin_idx + 2.0) * mel_delta
    mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins))[None, :]
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    return np.maximum(0.0, np.minimum(up, down))


def fbank(waveform: np.ndarray, num_mel_bins: int = 80, sample_frequency: float = 16000.0,
          frame_length_ms: float = 25.0, frame_shift_ms: float = 10.0,
          dither: float = 0.0, preemphasis: float = 0.97,
          low_freq: float = 20.0, high_freq: float = 0.0) -> np.ndarray:
    """waveform: (N,) float32 -> (T, num_mel_bins) log-fbank, kaldi semantics."""
    wav = waveform.astype(np.float32)
    window_shift = int(sample_frequency * frame_shift_ms / 1000.0)
    window_size = int(sample_frequency * frame_length_ms / 1000.0)          # 400
    padded = 1
    while padded < window_size:
        padded *= 2                                                          # 512
    assert 2 <= window_size <= len(wav)

    if len(wav) < window_size:
        return np.zeros((0, num_mel_bins), np.float32)
    m = 1 + (len(wav) - window_size) // window_shift                        # snip_edges
    frames = np.lib.stride_tricks.as_strided(
        wav, shape=(m, window_size),
        strides=(wav.strides[0] * window_shift, wav.strides[0])).copy()

    if dither != 0.0:  # sv.py always uses dither=0
        frames = frames + np.random.randn(*frames.shape) * dither
    frames -= frames.mean(axis=1, keepdims=True)                            # remove_dc_offset
    # raw energy BEFORE preemphasis/window (use_energy=False: only used to log-strip later? no)
    # kaldi with use_energy=False still computes it but does not emit it; skip.
    offset = np.pad(frames, ((0, 0), (1, 0)), mode="edge")
    frames = frames - preemphasis * offset[:, :-1]
    # povey = hann(periodic=False)^0.85
    i = np.arange(window_size)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * i / (window_size - 1))
    frames = frames * (hann ** 0.85).astype(np.float32)
    frames = np.pad(frames, ((0, 0), (0, padded - window_size)))

    spec = np.fft.rfft(frames, n=padded, axis=1)                            # (m, 257)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    bins = _get_mel_banks(num_mel_bins, padded, sample_frequency, low_freq, high_freq)
    bins = np.pad(bins, ((0, 0), (0, 1)))                                   # kaldi pads right col 0
    mel_energies = power @ bins.T                                           # (m, bins)
    mel_energies = np.maximum(mel_energies, np.finfo(np.float32).eps)
    return np.log(mel_energies).astype(np.float32)
