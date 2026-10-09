"""MLX-differentiable STFT (magnitude + mel) for s2 training.

Pure mx ops (no numpy boundary crossing): reflect pad, periodic hann,
strided frame gather, rfft, magnitude sqrt(|X|^2 + 1e-8), optional mel
basis + log clamp 1e-5 — mirroring official mel_processing_torch (which is
itself fully differentiable in torch; the shared numpy stft front-end is
NOT — it severs the autodiff graph and was silently zeroing the mel loss
gradient in the trainer).
"""

from __future__ import annotations

import mlx.core as mx


def _hann_window_mx(n: int, dtype=mx.float32) -> mx.array:
    # periodic hann: 0.5 - 0.5 cos(2 pi k / n)   (torch.hann_window periodic)
    k = mx.arange(n).astype(dtype)
    return 0.5 - 0.5 * mx.cos(2.0 * 3.141592653589793 * k / n)


def _reflect_pad_mx(x: mx.array, left: int, right: int) -> mx.array:
    """(B, T) reflect pad without crossing into numpy."""
    if left == 0 and right == 0:
        return x
    parts = [x]
    if left > 0:
        # x[1:left+1] reversed along T
        l = x[:, 1:left + 1][:, ::-1]
        parts.insert(0, l)
    if right > 0:
        t = x.shape[1]
        r = x[:, t - right - 1:t - 1][:, ::-1]
        parts.append(r)
    return mx.concatenate(parts, axis=1)


def stft_mag_mx(y: mx.array, n_fft: int, hop_size: int, win_size: int) -> mx.array:
    """(B, T) -> magnitude (B, n_fft//2+1, T'); differentiable end-to-end."""
    x = _reflect_pad_mx(y.astype(mx.float32), (n_fft - hop_size) // 2,
                        (n_fft - hop_size) // 2)
    b, t = x.shape
    n_frames = 1 + (t - n_fft) // hop_size if t >= n_fft else 0
    win = _hann_window_mx(win_size)
    # gather frames: (B, T', n_fft)
    idx = mx.arange(n_fft)[None, :] + hop_size * mx.arange(n_frames)[:, None]
    frames = x[:, idx] * win[None, None, :]
    spec = mx.fft.rfft(frames, axis=-1)
    mag = mx.sqrt(spec.real * spec.real + spec.imag * spec.imag + 1e-8)
    return mag.transpose(0, 2, 1)


def mel_spec_mx(y: mx.array, mel_basis: mx.array, n_fft: int, hop_size: int,
                win_size: int) -> mx.array:
    """(B, T) -> log-mel (B, n_mels, T'); official mel_spectrogram_torch."""
    spec = stft_mag_mx(y, n_fft, hop_size, win_size)
    return mx.log(mx.maximum(mel_basis @ spec, 1e-5))
