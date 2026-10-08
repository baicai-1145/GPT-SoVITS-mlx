"""torchaudio Resample (sinc_interp_hann default) in pure numpy.

Port of torchaudio.functional._get_sinc_resample_kernel +
_apply_sinc_resample_kernel (torchaudio 2.11, kernel built in float64,
cached as float32, conv in input dtype). Used by the dataset sv stage to
replicate 2-get-sv.py's torchaudio.transforms.Resample(32000, 16000)
without a torch dependency.

Verified: bit-level (max 0 diff after fp32 cast) vs torchaudio 2.11 CPU
on the gate corpus (tests/test_prepare_data.py heavy gate).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import i0


def _sinc_resample_kernel(orig_freq: int, new_freq: int,
                          lowpass_filter_width: int = 6,
                          rolloff: float = 0.99,
                          method: str = "sinc_interp_hann",
                          beta: float | None = None):
    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig = int(orig_freq) // gcd
    new = int(new_freq) // gcd
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base_freq)

    idx = np.arange(-width, width + orig, dtype=np.float64)[None, None] / orig
    t = np.arange(0, -new, -1, dtype=np.float64)[:, None, None] / new + idx
    t *= base_freq
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)

    if method == "sinc_interp_hann":
        window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:  # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        window = i0(beta * np.sqrt(1 - (t / lowpass_filter_width) ** 2)) / i0(beta)

    t *= math.pi
    scale = base_freq / orig
    with np.errstate(divide="ignore", invalid="ignore"):
        kernels = np.where(t == 0, 1.0, np.sin(t) / t)
    kernels = (kernels * window * scale).astype(np.float32)
    return kernels, width, orig, new


def resample_torchaudio(x: np.ndarray, orig_freq: int, new_freq: int) -> np.ndarray:
    """1-D float32 resample, torchaudio sinc_interp_hann semantics."""
    if orig_freq == new_freq:
        return x.copy()
    kernel, width, orig, new = _sinc_resample_kernel(orig_freq, new_freq)
    w = np.pad(x.astype(np.float32, copy=False), (width, width + orig))
    # torch conv1d cross-correlation with kernel (new, 1, taps) stride orig
    k = kernel[0]  # (1, taps) -> correlate
    taps = k.shape[1]
    n_out = (len(w) - taps) // orig + 1
    out = np.empty((new, n_out), np.float32)  # kernel rows = new phases
    for p in range(new):
        # conv1d: out[p, j] = sum_t w[j*orig + t] * kernel[p, 0, t]
        idx = np.arange(n_out)[:, None] * orig + np.arange(taps)[None, :]
        out[p] = (w[idx] @ k[p])
    target = int(np.ceil(new * len(x) / orig))
    return out.T.reshape(-1)[:target]
