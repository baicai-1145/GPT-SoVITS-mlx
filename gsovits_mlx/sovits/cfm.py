"""CFM (v3/v4 Euler flow) and CFMV5 rolling-chunk sampler. Pure MLX."""

from __future__ import annotations

import mlx.core as mx

V5_TEMPERATURE = 0.875
V5_ROLLING_TAIL_FRAMES = 32
V5_MAX_TARGET_CHUNK_FRAMES = 640
V5_REFERENCE_FRAMES = 500
V5_TOTAL_CHUNK_FRAMES = 1000
V5_DEFAULT_CFG = 1.30
V5_VERSIONS = {"v5", "v5dev", "v5turbo"}


def sampling_defaults(version, legacy_steps=32):
    if version == "v5turbo":
        return 4, 0.0
    if version in V5_VERSIONS:
        return 32, V5_DEFAULT_CFG
    return legacy_steps, 0.0


def resolve_sampling(version, sample_steps=None, cfg_rate=None, legacy_steps=32):
    steps, cfg = sampling_defaults(version, legacy_steps)
    return (steps if sample_steps is None else int(sample_steps),
            cfg if cfg_rate is None else float(cfg_rate))


def validate_v5_sampling(sample_steps, cfg_rate):
    steps = int(sample_steps)
    cfg = float(cfg_rate)
    if steps not in (4, 8, 16, 32):
        raise ValueError("Euler steps must be one of 4, 8, 16, 32")
    if not mx.isfinite(mx.array(cfg)) or cfg < 0 or cfg > 2:
        raise ValueError("CFG must be a finite value between 0 and 2")
    return steps, cfg


class CFM:
    """models.CFM inference (v3/v4). estimator is a DiT instance."""

    def __init__(self, in_channels: int, estimator):
        self.sigma_min = 1e-6
        self.estimator = estimator
        self.in_channels = in_channels

    def inference(self, mu: mx.array, x_lens, prompt: mx.array, n_timesteps: int,
                  temperature: float = 1.0, inference_cfg_rate: float = 0.0,
                  key: mx.array | None = None):
        """mu: (B, T, C) condition; prompt: (B, C, T_p) reference mel (normed).
        Returns generated (B, C, T)."""
        B, T = mu.shape[0], mu.shape[1]
        # Euler STATE stays fp32 (mixed-precision policy: fp16 GEMMs inside the
        # DiT, fp32 accumulation between steps) — torch reference accumulates
        # fp32; fp16 state drifts ~6e-3 over 32 steps.
        dtype = mx.float32
        x = mx.random.normal((B, self.in_channels, T), key=key).astype(dtype) * temperature
        prompt_len = prompt.shape[-1]
        prompt_x = mx.zeros_like(x)
        prompt_x[:, :, :prompt_len] = prompt[:, :, :prompt_len]
        x[:, :, :prompt_len] = 0
        mu_t = mx.transpose(mu, (0, 2, 1))  # (B, T, C)

        d = 1.0 / n_timesteps
        x_lens_t = mx.array([T], dtype=mx.int32)
        t_val = 0.0
        text_cache = None
        text_cfg_cache = None
        dt_cache = None
        for _ in range(n_timesteps):
            t_tensor = mx.full((B,), t_val, dtype=dtype)
            d_tensor = mx.full((B,), d, dtype=dtype)
            v_pred, text_emb, dt = self.estimator(
                x, prompt_x, x_lens_t, t_tensor, d_tensor, mu_t,
                use_grad_ckpt=False, drop_audio_cond=False, drop_text=False,
                infer=True, text_cache=text_cache, dt_cache=dt_cache,
            )
            v_pred = mx.transpose(v_pred, (0, 2, 1))
            text_cache = text_emb
            dt_cache = dt
            if inference_cfg_rate > 1e-5:
                neg, text_cfg_emb, _ = self.estimator(
                    x, prompt_x, x_lens_t, t_tensor, d_tensor, mu_t,
                    use_grad_ckpt=False, drop_audio_cond=True, drop_text=True,
                    infer=True, text_cache=text_cfg_cache, dt_cache=dt_cache,
                )
                neg = mx.transpose(neg, (0, 2, 1))
                text_cfg_cache = text_cfg_emb
                v_pred = v_pred + (v_pred - neg) * inference_cfg_rate
            x = x + d * v_pred
            t_val += d
            x[:, :, :prompt_len] = 0
        return x


class CFMV5:
    """v5_inference.CFMV5."""

    def __init__(self, in_channels: int, estimator, use_static_cache: bool = True):
        self.in_channels = in_channels
        self.estimator = estimator
        self.use_conditioner_cache = True
        self.use_static_cache = use_static_cache

    def inference(self, mu: mx.array, x_lens, prompt: mx.array, n_timesteps,
                  temperature: float = V5_TEMPERATURE, inference_cfg_rate: float = V5_DEFAULT_CFG,
                  key: mx.array | None = None):
        steps, cfg = validate_v5_sampling(n_timesteps, inference_cfg_rate)
        batch, frames = mu.shape[0], mu.shape[1]
        prompt_len = prompt.shape[-1]
        # Euler STATE stays fp32 (see CFM.inference note).
        dtype = mx.float32
        x = mx.random.normal((batch, self.in_channels, frames), key=key).astype(dtype) * V5_TEMPERATURE
        x[:, :, :prompt_len] = 0
        prompt_x = mx.zeros_like(x).astype(dtype)
        prompt_x[:, :, :prompt_len] = prompt.astype(dtype)
        condition = mx.transpose(mu, (0, 2, 1))
        cache_enabled = bool(self.use_conditioner_cache and self.use_static_cache)
        cache = self.estimator.prepare_static_cache(prompt_x, x_lens, condition) if cache_enabled else None
        text_cache = None
        step = 1.0 / steps
        for index in range(steps):
            time = mx.full((batch,), index * step, dtype=dtype)
            velocity, text_embedding, _ = self.estimator(
                x, prompt_x, x_lens, time, None, condition,
                drop_audio_cond=False, drop_text=False, static_cache=cache,
                infer=True, text_cache=text_cache,
            )
            if self.use_conditioner_cache and cache is None:
                text_cache = text_embedding
            velocity = mx.transpose(velocity, (0, 2, 1))
            if cfg > 1e-5:
                negative, _, _ = self.estimator(
                    x, prompt_x, x_lens, time, None, condition,
                    drop_audio_cond=True, drop_text=False, static_cache=cache,
                    infer=True, text_cache=text_cache,
                )
                negative = mx.transpose(negative, (0, 2, 1))
                velocity = velocity + cfg * (velocity - negative)
            x = x + step * velocity
            x[:, :, :prompt_len] = 0
        return x


def synthesize_v5_mel(model, reference_features: mx.array, target_features: mx.array,
                      reference_mel: mx.array, sample_steps=None, cfg_rate=None,
                      key: mx.array | None = None) -> mx.array:
    """model: SynthesizerTrnV3 (v5 family). reference_features: fea_ref (B,C,Tref)."""
    sample_steps, cfg_rate = resolve_sampling(model.version, sample_steps, cfg_rate)
    steps, cfg = validate_v5_sampling(sample_steps, cfg_rate)
    reference_frames = min(reference_mel.shape[-1], reference_features.shape[-1])
    if reference_frames < 1:
        raise ValueError("V5 requires nonempty reference mel and semantic features")
    reference_mel = reference_mel[:, :, :reference_frames][:, :, -V5_REFERENCE_FRAMES:]
    reference_features = reference_features[:, :, :reference_frames][:, :, -V5_REFERENCE_FRAMES:]
    reference_frames = reference_mel.shape[-1]
    chunk_frames = min(V5_TOTAL_CHUNK_FRAMES - reference_frames, V5_MAX_TARGET_CHUNK_FRAMES)
    original_mel = reference_mel
    original_features = reference_features
    rolling_mel = reference_mel
    rolling_features = reference_features
    results = []
    for index, start in enumerate(range(0, target_features.shape[-1], chunk_frames)):
        target = target_features[:, :, start : start + chunk_frames]
        if index == 0:
            prompt = original_mel
            features = original_features
        else:
            tail = min(V5_ROLLING_TAIL_FRAMES, reference_frames,
                       rolling_mel.shape[-1], rolling_features.shape[-1])
            prefix = reference_frames - tail
            prompt = mx.concatenate([original_mel[:, :, :prefix], rolling_mel[:, :, -tail:]], axis=-1)
            features = mx.concatenate([original_features[:, :, :prefix], rolling_features[:, :, -tail:]], axis=-1)
        mu = mx.transpose(mx.concatenate([features, target], axis=-1), (0, 2, 1))
        lengths = mx.array([mu.shape[1]], dtype=mx.int32)
        generated = model.cfm.inference(mu, lengths, prompt, steps,
                                        inference_cfg_rate=cfg, key=key)
        generated = generated[:, :, prompt.shape[-1]:]
        results.append(generated)
        rolling_mel = generated[:, :, -reference_frames:].astype(reference_mel.dtype)
        rolling_features = target[:, :, -reference_frames:]
    if not results:
        raise ValueError("V5 requires nonempty target semantic features")
    return mx.concatenate(results, axis=-1)
