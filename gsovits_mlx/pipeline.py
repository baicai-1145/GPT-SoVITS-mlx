"""GPT-SoVITS MLX inference pipeline (batch=1).

Covers:
- v2/v2Pro/v2ProPlus: SynthesizerTrn.decode (HiFi-GAN vocoder built-in, 32 kHz)
- v3: decode_encp -> chunked CFM -> BigVGAN 24 kHz (100-mel 256x)
- v4: decode_encp -> chunked CFM -> HiFi-GAN vocoder 48 kHz (100-mel 320 hop)

Reference: TTS_infer_pack/TTS.py using_vocoder_synthesis / batched_infer.
"""

from __future__ import annotations

import json
import os

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .gpt.t2s import Text2SemanticDecoder
from .io import load_mlx_safetensors
from .sovits.models_v1v2 import SynthesizerTrn
from .sovits.models_v3 import SynthesizerTrnV3
from .text.mel_frontend import mel_spectrogram, spectrogram

SPEC_MIN = -12.0
SPEC_MAX = 2.0


def norm_spec(x: mx.array) -> mx.array:
    return (x - SPEC_MIN) / (SPEC_MAX - SPEC_MIN) * 2 - 1


def denorm_spec(x: mx.array) -> mx.array:
    return (x + 1) / 2 * (SPEC_MAX - SPEC_MIN) + SPEC_MIN


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Linear interpolation resample (numpy; good enough for 16k hubert input)."""
    if sr_in == sr_out:
        return x
    n_out = int(len(x) * sr_out / sr_in)
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def load_audio_16k(path: str) -> np.ndarray:
    """Decode any audio file to float32 mono 16 kHz using ffmpeg."""
    import subprocess
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
           "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_sovits_v1v2(path: str, version: str):
    arrays = load_mlx_safetensors(os.path.join(path, "sovits.safetensors"))
    meta = json.load(open(os.path.join(path, "sovits.json")))
    hps = meta["model_hps"]
    g = arrays.get
    m = SynthesizerTrn(1025, 20, hps["inter_channels"], hps["hidden_channels"],
                       hps["filter_channels"], hps["n_heads"], hps["n_layers"],
                       hps["kernel_size"], hps["p_dropout"], hps["resblock"],
                       hps["resblock_kernel_sizes"], hps["resblock_dilation_sizes"],
                       hps["upsample_rates"], hps["upsample_initial_channel"],
                       hps["upsample_kernel_sizes"], gin_channels=hps["gin_channels"],
                       semantic_frame_rate=hps["semantic_frame_rate"], version=version)
    ep = m.enc_p
    ep.ssl_proj.weight = g("enc_p.ssl_proj.weight"); ep.ssl_proj.bias = g("enc_p.ssl_proj.bias")
    ep.text_embedding = g("enc_p.text_embedding")
    ep.proj.weight = g("enc_p.proj.weight"); ep.proj.bias = g("enc_p.proj.bias")
    for enc, prefix in ((ep.encoder_ssl, "enc_p.enc_ssl"),
                        (ep.encoder_text, "enc_p.enc_text"),
                        (ep.encoder2, "enc_p.enc2")):
        for i in range(len(enc.attn_layers)):
            at = enc.attn_layers[i]
            for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
                getattr(at, nm).weight = g(f"{prefix}.{i}.attn.{nm}.weight")
                getattr(at, nm).bias = g(f"{prefix}.{i}.attn.{nm}.bias")
            if f"{prefix}.{i}.attn.emb_rel_k" in arrays:
                at.emb_rel_k = g(f"{prefix}.{i}.attn.emb_rel_k")
                at.emb_rel_v = g(f"{prefix}.{i}.attn.emb_rel_v")
            n1, n2 = enc.norm_layers_1[i], enc.norm_layers_2[i]
            n1.gamma, n1.beta = g(f"{prefix}.{i}.norm1"), g(f"{prefix}.{i}.norm1.b")
            n2.gamma, n2.beta = g(f"{prefix}.{i}.norm2"), g(f"{prefix}.{i}.norm2.b")
            f = enc.ffn_layers[i]
            f.conv_1.weight = g(f"{prefix}.{i}.ffn.conv1.weight")
            f.conv_1.bias = g(f"{prefix}.{i}.ffn.conv1.bias")
            f.conv_2.weight = g(f"{prefix}.{i}.ffn.conv2.weight")
            f.conv_2.bias = g(f"{prefix}.{i}.ffn.conv2.bias")
    mr = ep.mrte
    mr.c_pre.weight, mr.c_pre.bias = g("enc_p.mrte.c_pre.weight"), g("enc_p.mrte.c_pre.bias")
    mr.text_pre.weight, mr.text_pre.bias = g("enc_p.mrte.text_pre.weight"), g("enc_p.mrte.text_pre.bias")
    mr.c_post.weight, mr.c_post.bias = g("enc_p.mrte.c_post.weight"), g("enc_p.mrte.c_post.bias")
    ca = mr.cross_attention
    for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
        getattr(ca, nm).weight = g(f"enc_p.mrte.cross_attn.{nm}.weight")
        getattr(ca, nm).bias = g(f"enc_p.mrte.cross_attn.{nm}.bias")
    if "enc_p.mrte.cross_attn.emb_rel_k" in arrays:
        ca.emb_rel_k = g("enc_p.mrte.cross_attn.emb_rel_k")
        ca.emb_rel_v = g("enc_p.mrte.cross_attn.emb_rel_v")
    m.ssl_proj.weight, m.ssl_proj.bias = g("ssl_proj.weight"), g("ssl_proj.bias")
    m.quantizer.embed = g("quantizer.codebook")
    # ref_enc
    re_ = m.ref_enc
    re_.spectral_0.weight = g("ref_enc.spectral.0.weight")
    re_.spectral_0.bias = g("ref_enc.spectral.0.bias")
    re_.spectral_1.weight = g("ref_enc.spectral.3.weight")
    re_.spectral_1.bias = g("ref_enc.spectral.3.bias")
    for i, t in enumerate((re_.temporal_0, re_.temporal_1)):
        t.w_1 = g(f"ref_enc.temporal.{i}.w1"); t.b_1 = g(f"ref_enc.temporal.{i}.b1")
    sa = re_.slf_attn
    sa.w_qs.weight, sa.w_qs.bias = g("ref_enc.slf_attn.w_qs"), g("ref_enc.slf_attn.w_qs.b")
    sa.w_ks.weight, sa.w_ks.bias = g("ref_enc.slf_attn.w_ks"), g("ref_enc.slf_attn.w_ks.b")
    sa.w_vs.weight, sa.w_vs.bias = g("ref_enc.slf_attn.w_vs"), g("ref_enc.slf_attn.w_vs.b")
    sa.fc.weight, sa.fc.bias = g("ref_enc.slf_attn.fc"), g("ref_enc.slf_attn.fc.b")
    re_.fc.weight, re_.fc.bias = g("ref_enc.fc"), g("ref_enc.fc.b")
    # flow (v2 checkpoints have no top-level flow.pre/post)
    fl = m.flow
    if "flow.pre.weight" in arrays:
        fl.pre.weight, fl.pre.bias = g("flow.pre.weight"), g("flow.pre.bias")
        fl.post.weight, fl.post.bias = g("flow.post.weight"), g("flow.post.bias")
    wn_layers = [f for f in m.flow.flows if hasattr(f, "enc")]
    has_cond = "flow.0.enc.cond" in arrays
    for fi, rc_layer in enumerate(wn_layers):
        rc_layer.pre.weight, rc_layer.pre.bias = g(f"flow.{fi}.pre.weight"), g(f"flow.{fi}.pre.bias")
        rc_layer.post.weight, rc_layer.post.bias = g(f"flow.{fi}.post.weight"), g(f"flow.{fi}.post.bias")
        enc = rc_layer.enc
        if has_cond:
            enc.cond_layer.weight = g(f"flow.{fi}.enc.cond")
            enc.cond_layer.bias = g(f"flow.{fi}.enc.cond.b")
        wn = enc
        for wi in range(len(wn.in_layers)):
            wn.in_layers[wi].weight = g(f"flow.{fi}.enc.in.{wi}")
            wn.in_layers[wi].bias = g(f"flow.{fi}.enc.in.{wi}.b")
            wn.res_skip_layers[wi].weight = g(f"flow.{fi}.enc.skip.{wi}")
            wn.res_skip_layers[wi].bias = g(f"flow.{fi}.enc.skip.{wi}.b")
    # dec
    dec = m.dec
    dec.conv_pre.weight, dec.conv_pre.bias = g("dec.conv_pre.weight"), g("dec.conv_pre.bias")
    for i in range(len(dec.ups)):
        dec.ups[i].weight = g(f"dec.ups.{i}.weight"); dec.ups[i].bias = g(f"dec.ups.{i}.bias")
    for i, rb in enumerate(dec.resblocks):
        for cn, blocks in (("convs1", rb.convs1), ("convs2", rb.convs2)):
            for j, cv in enumerate(blocks):
                cv.weight = g(f"dec.resblocks.{i}.{cn}.{j}.weight")
                cv.bias = g(f"dec.resblocks.{i}.{cn}.{j}.bias")
    dec.cond.weight, dec.cond.bias = g("dec.cond.weight"), g("dec.cond.bias")
    dec.conv_post.weight, dec.conv_post.bias = g("dec.conv_post.weight"), g("dec.conv_post.bias")
    if version in ("v2Pro", "v2ProPlus"):
        m.sv_emb.weight, m.sv_emb.bias = g("sv_emb.weight"), g("sv_emb.bias")
        m.ge_to512.weight, m.ge_to512.bias = g("ge_to512.weight"), g("ge_to512.bias")
        m.prelu_weight = g("prelu.weight").reshape(-1)
    mx.eval(m.parameters())
    return m, meta


def load_gpt(path: str) -> Text2SemanticDecoder:
    arrays = load_mlx_safetensors(os.path.join(path, "gpt.safetensors"))
    meta = json.load(open(os.path.join(path, "gpt.json")))
    gpt = Text2SemanticDecoder(meta["config"])
    gpt.load(dict(arrays))
    mx.eval(list(gpt.__dict__.values()))
    return gpt
