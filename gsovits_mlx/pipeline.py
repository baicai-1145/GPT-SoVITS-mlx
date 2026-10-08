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
from .io import eval_tree, load_mlx_safetensors, release
from .model_cache import resident_model
from .sovits.models_v1v2 import SynthesizerTrn
from .sovits.models_v3 import SynthesizerTrnV3
from .sovits.dit import load_dit_params
from .text.mel_frontend import mel_spectrogram, spectrogram
from .vocoder.bigvgan import BigVGAN

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
    """Decode any audio file to float32 mono 16 kHz.

    Uses soundfile (libsndfile) with numpy linear resampling. ffmpeg f32le
    output on this machine produced out-of-range samples (max 1.52 on a
    clean int16 file), so it is avoided.
    """
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != 16000:
        mono = resample_linear(mono, sr, 16000)
    return mono


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@resident_model("sovits")
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
    _load_text_encoder_shared(m.enc_p, g, arrays)
    _load_ref_enc(m.ref_enc, g)
    m.ssl_proj.weight, m.ssl_proj.bias = g("ssl_proj.weight"), g("ssl_proj.bias")
    m.quantizer.embed = g("quantizer.codebook")
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
        m.prelu_weight = g("prelu.weight").reshape(1, -1, 1)
    # m.param holds array copies; the mmap-backed mapping can be dropped now.
    m.param = None
    eval_tree(m)
    release(arrays)
    return m, meta


@resident_model("gpt")
def load_gpt(path: str) -> Text2SemanticDecoder:
    arrays = load_mlx_safetensors(os.path.join(path, "gpt.safetensors"))
    meta = json.load(open(os.path.join(path, "gpt.json")))
    gpt = Text2SemanticDecoder(meta["config"])
    gpt.load(dict(arrays))
    eval_tree(gpt)
    release(arrays)
    _prewarm_ar_batch(gpt)
    return gpt


def _prewarm_ar_batch(gpt: Text2SemanticDecoder) -> None:
    """Shape-prewarm for batched AR (cold-start mitigation).

    MLX compiles Metal kernels per shape-specialization: the first request
    at a new batch width B pays a full compile wave inside its latency
    (measured: B=8 first-run 5.74s vs warm 2.26s; warmup at B=2 covers only
    B=2 shapes). MLX 0.32.2 has no persistent kernel cache, so the only
    lever is running one tiny dummy decode per planned B at load time.
    This also sizes the buffer pool and ramps GPU clocks before the first
    real request. GSOVITS_AR_PREWARM="2,4,8" (comma list, "0" disables;
    default off to keep CLI one-shot loads fast).
    """
    import os
    import mlx.core as mx
    spec = os.environ.get("GSOVITS_AR_PREWARM", "")
    if not spec or spec == "0":
        return
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            B = int(token)
        except ValueError:
            continue
        if B < 1:
            continue
        try:
            ph = mx.zeros((1, 8), mx.int32)
            bt = mx.zeros((1, 1024, 8), mx.float32)
            pr = mx.zeros((1, 4), mx.int32)
            if B == 1:
                # serial shapes: trace python paths + trigger the system
                # MTLCompiler cache for mixed_gemv/SDPA first launches
                gpt.infer(ph, bt, pr, fast_cache=True)
            else:
                gpt.infer_batch([(ph, bt, pr)] * B, uniforms=[[0.5] * 4] * B)
        except Exception:
            # prewarm is best-effort: never block model load
            pass


# ---------------------------------------------------------------------------
# SoVITS v3/v4/v5 (SynthesizerTrnV3) + BigVGAN vocoder
# ---------------------------------------------------------------------------

def _load_text_encoder_shared(ep, g, arrays):
    """enc_p weights shared by v1/v2/v3 layouts."""
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


def _load_ref_enc(re_, g):
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


@resident_model("sovits")
def load_sovits_v3(path: str, version: str = "v3"):
    """Converted SynthesizerTrnV3 (v3/v4/v5 family) from sovits.safetensors + sovits.json."""
    arrays = load_mlx_safetensors(os.path.join(path, "sovits.safetensors"))
    meta = json.load(open(os.path.join(path, "sovits.json")))
    hps = meta["model_hps"]
    g = arrays.get
    # v1/v2-only constructor fields are unused by SynthesizerTrnV3
    m = SynthesizerTrnV3(1025, 20, hps["inter_channels"], hps["hidden_channels"],
                         768, 2, 6, 3, 0.1, "1", [3, 7, 11], [[1, 3, 5]] * 3,
                         [10, 8, 2, 2, 2], 512, [16, 16, 8, 2, 2],
                         gin_channels=hps["gin_channels"],
                         semantic_frame_rate=hps.get("semantic_frame_rate") or "25hz",
                         version=version)
    _load_text_encoder_shared(m.enc_p, g, arrays)
    _load_ref_enc(m.ref_enc, g)
    if "ssl_proj.weight" in arrays:
        m.ssl_proj.weight, m.ssl_proj.bias = g("ssl_proj.weight"), g("ssl_proj.bias")
    m.quantizer.embed = g("quantizer.codebook")
    # bridge (torch Sequential(Conv1d, LeakyReLU) -> bridge_0 + leaky_relu in forward)
    m.bridge_0.weight, m.bridge_0.bias = g("bridge.weight"), g("bridge.bias")
    # wns1 (WNEncoder: pre -> WN(gin) -> proj)
    m.wns1.pre.weight, m.wns1.pre.bias = g("wns1.pre.weight"), g("wns1.pre.bias")
    m.wns1.proj.weight, m.wns1.proj.bias = g("wns1.proj.weight"), g("wns1.proj.bias")
    m.wns1.enc.cond_layer.weight = g("wns1.cond")
    m.wns1.enc.cond_layer.bias = g("wns1.cond.b")
    n_wn = len(m.wns1.enc.in_layers)
    for wi in range(n_wn):
        m.wns1.enc.in_layers[wi].weight = g(f"wns1.in.{wi}")
        m.wns1.enc.in_layers[wi].bias = g(f"wns1.in.{wi}.b")
        m.wns1.enc.res_skip_layers[wi].weight = g(f"wns1.skip.{wi}")
        m.wns1.enc.res_skip_layers[wi].bias = g(f"wns1.skip.{wi}.b")
    if "linear_mel.weight" in arrays:
        m.linear_mel.weight, m.linear_mel.bias = g("linear_mel.weight"), g("linear_mel.bias")
    load_dit_params(m.cfm.estimator, arrays, meta["dit"]["depth"],
                    meta["dit"]["text_blocks"], has_d_embed=version not in {"v5", "v5dev", "v5turbo"})
    eval_tree(m)
    release(arrays)
    return m, meta


@resident_model("vocoder")
def load_vocoder_v4(path: str, dtype=None):
    """Converted v4/v5 Generator; optionally cast parameters at load time."""
    from .sovits.models_v1v2 import Generator
    arrays = load_mlx_safetensors(os.path.join(path, "vocoder.safetensors"))
    h = json.load(open(os.path.join(path, "vocoder.json")))

    def g(key):
        value = arrays.get(key)
        return value.astype(dtype) if dtype is not None and value is not None else value

    voc = Generator(h["initial_channel"], h["resblock"], h["resblock_kernel_sizes"],
                    h["resblock_dilation_sizes"], h["upsample_rates"],
                    h["upsample_initial_channel"], h["upsample_kernel_sizes"],
                    gin_channels=h["gin_channels"])
    voc.conv_pre.weight, voc.conv_pre.bias = g("conv_pre.weight"), g("conv_pre.bias")
    for i in range(len(voc.ups)):
        voc.ups[i].weight = g(f"ups.{i}.weight")
        voc.ups[i].bias = g(f"ups.{i}.bias")
    for b, rb in enumerate(voc.resblocks):
        for j in range(len(rb.convs1)):
            rb.convs1[j].weight = g(f"resblocks.{b}.convs1.{j}.weight")
            rb.convs1[j].bias = g(f"resblocks.{b}.convs1.{j}.bias")
            rb.convs2[j].weight = g(f"resblocks.{b}.convs2.{j}.weight")
            rb.convs2[j].bias = g(f"resblocks.{b}.convs2.{j}.bias")
    voc.conv_post.weight = g("conv_post.weight")
    voc.conv_post.bias = g("conv_post.bias")
    eval_tree(voc)
    release(arrays)
    return voc


@resident_model("sv")
def load_sv_encoder(path: str):
    """Converted ERes2NetV2 speaker encoder (sv.safetensors + sv.json)."""
    from .sovits.sv_encoder import AFF, BN2d, BasicAFFBlock, BasicResBlock, Conv2d, ERes2NetV2
    arrays = load_mlx_safetensors(os.path.join(path, "sv.safetensors"))
    h = json.load(open(os.path.join(path, "sv.json")))
    g = arrays.get
    m = ERes2NetV2(base_width=h["base_width"], scale=h["scale"], expansion=h["expansion"],
                   num_blocks=tuple(h["num_blocks"]), m_channels=h["m_channels"])

    def load_bn(bn: BN2d, prefix: str):
        bn.weight, bn.bias = g(prefix + ".weight"), g(prefix + ".bias")
        bn.running_mean, bn.running_var = g(prefix + ".running_mean"), g(prefix + ".running_var")

    def load_conv(c: Conv2d, prefix: str):
        c.weight = g(prefix + ".weight")
        c.bias = g(prefix + ".bias")

    def load_block(blk, prefix: str, aff: bool):
        load_conv(blk.conv1, prefix + ".conv1"); load_bn(blk.bn1, prefix + ".bn1")
        for i, (cv, bn) in enumerate(zip(blk.convs, blk.bns)):
            load_conv(cv, f"{prefix}.convs.{i}"); load_bn(bn, f"{prefix}.bns.{i}")
        if aff:
            for j, fu in enumerate(blk.fuses):
                load_conv(fu.c0, f"{prefix}.fuse_models.{j}.local_att.0")
                load_bn(fu.bn0, f"{prefix}.fuse_models.{j}.local_att.1")
                load_conv(fu.c1, f"{prefix}.fuse_models.{j}.local_att.3")
                load_bn(fu.bn1, f"{prefix}.fuse_models.{j}.local_att.4")
        load_conv(blk.conv3, prefix + ".conv3"); load_bn(blk.bn3, prefix + ".bn3")
        if blk.sc_conv is not None:
            load_conv(blk.sc_conv, prefix + ".shortcut.0")
            load_bn(blk.sc_bn, prefix + ".shortcut.1")

    load_conv(m.conv1, "conv1"); load_bn(m.bn1, "bn1")
    n_l1 = len(m.layer1)
    for i, blk in enumerate(m.layer1):
        load_block(blk, f"layer1.{i}", aff=False)
    for i, blk in enumerate(m.layer2):
        load_block(blk, f"layer2.{i}", aff=False)
    for i, blk in enumerate(m.layer3):
        load_block(blk, f"layer3.{i}", aff=True)
    for i, blk in enumerate(m.layer4):
        load_block(blk, f"layer4.{i}", aff=True)
    load_conv(m.layer3_ds, "layer3_ds")
    fu = m.fuse34
    load_conv(fu.c0, "fuse34.local_att.0"); load_bn(fu.bn0, "fuse34.local_att.1")
    load_conv(fu.c1, "fuse34.local_att.3"); load_bn(fu.bn1, "fuse34.local_att.4")
    # verify all arrays consumed
    missing = [k for k in arrays if ("running" in k or k.endswith(".weight") or k.endswith(".bias"))
               and not k.endswith("num_batches_tracked")
               and all(a.get(k) is None for a in [arrays])]
    eval_tree(m)
    release(arrays)
    return m, h


@resident_model("vocoder")
def load_bigvgan(path: str):
    """Converted BigVGAN v2 (bigvgan.safetensors + bigvgan.json)."""
    arrays = load_mlx_safetensors(os.path.join(path, "bigvgan.safetensors"))
    h = json.load(open(os.path.join(path, "bigvgan.json")))
    g = arrays.get
    voc = BigVGAN(h)
    voc.conv_pre.weight, voc.conv_pre.bias = g("conv_pre.weight"), g("conv_pre.bias")
    for i in range(len(voc.ups)):
        voc.ups[i].weight = g(f"ups.{i}.weight")
        voc.ups[i].bias = g(f"ups.{i}.bias")
    n_rb = len(voc.resblocks)
    for b in range(n_rb):
        rb = voc.resblocks[b]
        for j in range(len(rb.convs1)):
            rb.convs1[j].weight = g(f"resblocks.{b}.convs1.{j}.weight")
            rb.convs1[j].bias = g(f"resblocks.{b}.convs1.{j}.bias")
            rb.convs2[j].weight = g(f"resblocks.{b}.convs2.{j}.weight")
            rb.convs2[j].bias = g(f"resblocks.{b}.convs2.{j}.bias")
        for a, act in enumerate(rb.activations):
            core = act.act
            core.alpha = g(f"resblocks.{b}.act.{a}.alpha")
            if hasattr(core, "beta"):
                core.beta = g(f"resblocks.{b}.act.{a}.beta")
    post = voc.activation_post.act
    post.alpha = g("post.alpha")
    if hasattr(post, "beta"):
        post.beta = g("post.beta")
    voc.conv_post.weight = g("conv_post.weight")
    if voc.use_bias_at_final:
        voc.conv_post.bias = g("conv_post.bias")
    eval_tree(voc)
    release(arrays)
    return voc


# ---------------------------------------------------------------------------
# v3/v4 chunked CFM decode (TTS.py using_vocoder_synthesis)
# ---------------------------------------------------------------------------

def decode_encp_v3(model: SynthesizerTrnV3, codes: mx.array, text: mx.array,
                   refer: mx.array, ge: mx.array | None = None,
                   speed: float = 1.0):
    """SynthesizerTrnV3.decode_encp — codes (B,1,T), text (B,Tph), refer (B,Cspec,T)."""
    if ge is None:
        refer_mask = mx.ones((refer.shape[0], 1, refer.shape[2]), dtype=refer.dtype)
        ge = model.ref_enc(refer[:, :704] * refer_mask, refer_mask)
    return model.decode_encp(codes, text, refer=refer, ge=ge, speed=speed)


def cfm_chunked_decode_v3(model: SynthesizerTrnV3, fea_ref: mx.array, fea_todo: mx.array,
                          mel2: mx.array, sample_steps: int = 32,
                          inference_cfg_rate: float = 0.0,
                          key: mx.array | None = None) -> mx.array:
    """TTS.py v3 chunked CFM loop (T_ref=468, T_chunk=934, 100-mel). Returns the
    predicted mel (B, 100, T_target); denorm NOT applied."""
    return _cfm_chunked_decode(model, fea_ref, fea_todo, mel2, sample_steps,
                               inference_cfg_rate, key, T_ref=468, T_chunk=934)


def cfm_chunked_decode_v4(model: SynthesizerTrnV3, fea_ref: mx.array, fea_todo: mx.array,
                          mel2: mx.array, sample_steps: int = 32,
                          inference_cfg_rate: float = 0.0,
                          key: mx.array | None = None) -> mx.array:
    """TTS.py v4 chunked CFM loop (vocoder_configs: T_ref=500, T_chunk=1000)."""
    return _cfm_chunked_decode(model, fea_ref, fea_todo, mel2, sample_steps,
                               inference_cfg_rate, key, T_ref=500, T_chunk=1000)


def _cfm_chunked_decode(model: SynthesizerTrnV3, fea_ref: mx.array, fea_todo: mx.array,
                        mel2: mx.array, sample_steps: int,
                        inference_cfg_rate: float, key: mx.array | None,
                        T_ref: int, T_chunk: int, *, memory_efficient: bool = True) -> mx.array:
    """Shared chunked CFM loop (using_vocoder_synthesis): crops the prompt to the
    last T_ref frames, then slides chunk_len = T_chunk - T_min windows; each chunk's
    CFM result beyond the prompt becomes the next prompt.

    memory_efficient=True evaluates and frees each chunk as soon as the next
    rolling prompt is sliced off (physical footprint = one chunk + running
    result buffer) instead of retaining the whole chunk graph list; array
    contents are identical.
    """
    T_min = min(mel2.shape[2], fea_ref.shape[2])
    mel2 = mel2[:, :, :T_min]
    fea_ref = fea_ref[:, :, :T_min]
    if T_min > T_ref:
        mel2 = mel2[:, :, -T_ref:]
        fea_ref = fea_ref[:, :, -T_ref:]
        T_min = T_ref
    chunk_len = T_chunk - T_min
    cfm_resss: list[mx.array] = []
    idx = 0
    while True:
        fea_todo_chunk = fea_todo[:, :, idx : idx + chunk_len]
        if fea_todo_chunk.shape[-1] == 0:
            break
        idx += chunk_len
        fea = mx.concatenate([fea_ref, fea_todo_chunk], 2).transpose(0, 2, 1)
        cfm_res = model.cfm.inference(
            fea, mx.array([fea.shape[1]]), mel2, sample_steps,
            inference_cfg_rate=inference_cfg_rate, key=key)
        cfm_res = cfm_res[:, :, mel2.shape[2]:]
        mel2 = cfm_res[:, :, -T_min:]
        fea_ref = fea_todo_chunk[:, :, -T_min:]
        if memory_efficient:
            # Realize this chunk (and the DiT graphs behind it) now so the
            # eager graph dies here instead of piling up across chunks; the
            # rolling mel2/fea_ref slices keep their (now realized) buffers.
            cfm_resss.append(cfm_res)
            mx.eval(cfm_res)
            fea = None
            fea_todo_chunk = None
            cfm_res = None
        else:
            cfm_resss.append(cfm_res)
    return mx.concatenate(cfm_resss, 2)
