"""s2 (SoVITS v1/v2/v2Pro/v2ProPlus) GAN trainer in MLX.

Port of the official GPT-SoVITS s2_train.py train_and_evaluate semantics on
top of the task-1 shared core (train.GradScaler / train.AdamW / ExponentialLR
below / train.ckpt exporters).

Training-step semantics replicated from s2_train.py (read line-by-line):

  1. fp16 autocast forward of net_g (full SynthesizerTrn.forward including
     the autocast-disabled fp32 ssl_proj+quantizer block; with
     freeze_quantizer=True the forward eval()s ssl_proj+quantizer every
     step, so the VQ self.training branches (commit loss, EMA update) never
     run: kl_ssl is EXACTLY 0.0, and loss_gen_all's `+ kl_ssl * 1` term is
     replicated verbatim as a logged 0.0)
  2. mel = spec_to_mel(spec); y_mel = slice_segments(mel, ids_slice, 32);
     y_hat_mel = mel_spectrogram(y_hat) — a FRESH stft on the generated
     20480-sample slice (magnitude -> mel fb -> ln clamp 1e-5).
  3. D step first: net_d(y, y_hat.detach()) with fp32 LSGAN losses;
     scaler.scale(loss).backward(); scaler.unscale_(optim_d);
     clip_grad_value_(None) — a NO-OP on grads (official commons.py only
     clamps when clip_value is not None); scaler.step(optim_d).
  4. G step: net_d(y, y_hat) feature maps; fp32 losses:
     mel*45 + kl*1.0 + feature*2 + generator LS + kl_ssl*1; backward;
     unscale_; no-op clip; scaler.step(optim_g); scaler.update() — the
     SINGLE shared GradScaler updates ONLY here (the D step ran on the
     pre-update scale, exactly like torch when update() is deferred).
  5. ExponentialLR gamma=0.999875 per EPOCH (stepped after each epoch, plus
     the official resume fast-forward loop).

Model: training-mode SynthesizerTrn reusing inference modules from
gsovits_mlx/sovits (TextEncoder / Generator / ResidualCouplingBlock /
MelStyleEncoder / MRTE / WN) plus a training PosteriorEncoder (enc_q:
z = (m + randn*exp(logs))*mask with g DETACHED, per official) and the v2Pro
sv-conditioning path. Everything is driven through a FLAT {name: mx.array}
dict of torch-checkpoint names (MLX tensor layouts) so the task-1
optimizers and ckpt exporters work unchanged; modules re-bind references
from the fp16 working copy each forward (the moral equivalent of torch
autocast's per-forward cast; masters stay fp32 in the optimizer).

fp16 policy: masters fp32 (AdamW groups); per step a fp16 copy is made for
every non-quantizer param. The ssl_proj/quantizer block is computed in
fp32 always (official autocast(enabled=False)); v2Pro sv_emb/ge_to512/
prelu are also kept fp32 (models_local/v2pro exports them F32).
"""

from __future__ import annotations

import math

import mlx.core as mx

from ..sovits.models_v1v2 import (
    Generator,
    MelStyleEncoder,
    ResidualCouplingBlock,
    TextEncoder,
    _nearest_interp,
)
from ..utils.layers import Conv1d, WN, sequence_mask

__all__ = [
    "SynthesizerTrnTrain", "spec_to_mel", "mel_spectrogram_train",
    "slice_segments", "rand_slice_segments", "feature_loss",
    "discriminator_loss", "generator_loss", "kl_loss", "ExponentialLR",
]


# ---------------------------------------------------------------------------
# spectrogram / mel (module/mel_processing.py)
# ---------------------------------------------------------------------------

def spec_to_mel(spec: mx.array, mel_basis: mx.array) -> mx.array:
    """mel = ln(clamp(mel_fb @ spec, 1e-5)); spec (B, F, T) fp32."""
    return mx.log(mx.maximum(mel_basis @ spec, 1e-5))


def mel_spectrogram_train(y: mx.array, mel_basis: mx.array, n_fft: int,
                          hop_size: int, win_size: int) -> mx.array:
    """Fresh mel on generated audio (B, T) -> (B, n_mels, T').

    Official mel_spectrogram_torch: reflect pad (n_fft-hop)/2 both sides,
    hann window, sqrt(|stft|^2 + 1e-8), mel fb matmul, ln clamp 1e-5.
    DIFFERENTIABLE pure-mx path (gsovits_mlx.train.s2_mel): the shared
    numpy stft front-end severs the autodiff graph (numpy boundary) and
    silently zeroes the mel-loss gradient — caught by the 20-step torch
    reference (loss_mel diverged 22->82 while torch stayed ~20).
    """
    from .s2_mel import mel_spec_mx
    return mel_spec_mx(y.astype(mx.float32), mel_basis, n_fft, hop_size,
                       win_size)


def slice_segments(x: mx.array, ids_str, segment_size: int) -> mx.array:
    """torch commons.slice_segments on (B, C, T)."""
    ret = []
    for i in range(x.shape[0]):
        idx = int(ids_str[i]) if not isinstance(ids_str, int) else ids_str
        ret.append(x[i, :, idx: idx + segment_size])
    return mx.stack(ret, 0)


def rand_slice_segments(x: mx.array, x_lengths: mx.array, segment_size: int,
                        key: mx.array | None = None):
    """torch commons.rand_slice_segments: ids = floor(rand*b @ (len-seg+1))."""
    b = x.shape[0]
    ids_str_max = x_lengths - segment_size + 1
    r = mx.random.uniform(shape=(b,), key=key)
    ids_str = (r * ids_str_max).astype(mx.int32)
    return slice_segments(x, ids_str, segment_size), ids_str


# ---------------------------------------------------------------------------
# losses (module/losses.py) — computed in fp32
# ---------------------------------------------------------------------------

def feature_loss(fmap_r, fmap_g) -> mx.array:
    loss = mx.zeros(())
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = mx.stop_gradient(rl).astype(mx.float32)
            gl = gl.astype(mx.float32)
            loss = loss + mx.mean(mx.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = mx.zeros(())
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.astype(mx.float32)
        dg = dg.astype(mx.float32)
        r_loss = mx.mean((1 - dr) ** 2)
        g_loss = mx.mean(dg ** 2)
        loss = loss + r_loss + g_loss
        r_losses.append(float(r_loss))
        g_losses.append(float(g_loss))
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = mx.zeros(())
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.astype(mx.float32)
        l = mx.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss = loss + l
    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> mx.array:
    z_p = z_p.astype(mx.float32)
    logs_q = logs_q.astype(mx.float32)
    m_p = m_p.astype(mx.float32)
    logs_p = logs_p.astype(mx.float32)
    z_mask = z_mask.astype(mx.float32)
    kl = logs_p - logs_q - 0.5
    kl = kl + 0.5 * ((z_p - m_p) ** 2) * mx.exp(-2.0 * logs_p)
    kl = mx.sum(kl * z_mask)
    return kl / mx.sum(z_mask)


# ---------------------------------------------------------------------------
# weight binding helpers (flat torch-name dict -> module attributes)
# ---------------------------------------------------------------------------

def bind_wn(wn: WN, params: dict, prefix: str) -> None:
    """Bind WN params; weight-norm convs use live (g, v) pairs when present.

    torch modules.WN wraps cond_layer/in_layers/res_skip_layers in
    weight_norm; the checkpoint carries weight_g (out,1,1)+weight_v
    (out,in,k). The WN class's inner Conv1d layers get weight_g/weight_v
    attributes; utils.layers.Conv1d must support the live parameterization
    (it computes the effective weight per call when weight_v is set).
    """
    from ..utils.layers import Conv1d as _C1
    _C1.use_live_weight_norm = True  # noqa: B010 — flag documented below
    if f"{prefix}.cond.weight" in params:
        wn.cond_layer.weight = params[f"{prefix}.cond.weight"]
        wn.cond_layer.bias = params[f"{prefix}.cond.bias"]
    elif f"{prefix}.cond_layer.weight_g" in params:
        _set_wn_conv(wn.cond_layer,
                     params[f"{prefix}.cond_layer.weight_g"].reshape(-1),
                     params[f"{prefix}.cond_layer.weight_v"],
                     params[f"{prefix}.cond_layer.bias"])
    for i in range(len(wn.in_layers)):
        _bind_wn_conv(wn.in_layers[i], params, f"{prefix}.in_layers.{i}")
        _bind_wn_conv(wn.res_skip_layers[i], params, f"{prefix}.res_skip_layers.{i}")


def _set_wn_conv(conv, g, v, bias) -> None:
    conv.weight_g = g
    conv.weight_v = v
    conv.bias = bias


def _bind_wn_conv(conv, params: dict, prefix: str) -> None:
    if f"{prefix}.weight_g" in params:
        _set_wn_conv(conv, params[f"{prefix}.weight_g"].reshape(-1),
                     params[f"{prefix}.weight_v"], params[f"{prefix}.bias"])
    else:
        conv.weight = params[f"{prefix}.weight"]
        conv.bias = params[f"{prefix}.bias"]


def _bind_encoder(enc, params: dict, prefix: str) -> None:
    for i in range(len(enc.attn_layers)):
        at = enc.attn_layers[i]
        for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
            getattr(at, nm).weight = params[f"{prefix}.attn_layers.{i}.{nm}.weight"]
            getattr(at, nm).bias = params[f"{prefix}.attn_layers.{i}.{nm}.bias"]
        if f"{prefix}.attn_layers.{i}.emb_rel_k" in params:
            at.emb_rel_k = params[f"{prefix}.attn_layers.{i}.emb_rel_k"]
            at.emb_rel_v = params[f"{prefix}.attn_layers.{i}.emb_rel_v"]
        enc.norm_layers_1[i].gamma = params[f"{prefix}.norm_layers_1.{i}.gamma"]
        enc.norm_layers_1[i].beta = params[f"{prefix}.norm_layers_1.{i}.beta"]
        enc.norm_layers_2[i].gamma = params[f"{prefix}.norm_layers_2.{i}.gamma"]
        enc.norm_layers_2[i].beta = params[f"{prefix}.norm_layers_2.{i}.beta"]
        ffn = enc.ffn_layers[i]
        ffn.conv_1.weight = params[f"{prefix}.ffn_layers.{i}.conv_1.weight"]
        ffn.conv_1.bias = params[f"{prefix}.ffn_layers.{i}.conv_1.bias"]
        ffn.conv_2.weight = params[f"{prefix}.ffn_layers.{i}.conv_2.weight"]
        ffn.conv_2.bias = params[f"{prefix}.ffn_layers.{i}.conv_2.bias"]


def _bind_mrte(mrte, params: dict, prefix: str) -> None:
    for nm in ("c_pre", "text_pre", "c_post"):
        getattr(mrte, nm).weight = params[f"{prefix}.{nm}.weight"]
        getattr(mrte, nm).bias = params[f"{prefix}.{nm}.bias"]
    ca = mrte.cross_attention
    for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
        getattr(ca, nm).weight = params[f"{prefix}.cross_attention.{nm}.weight"]
        getattr(ca, nm).bias = params[f"{prefix}.cross_attention.{nm}.bias"]
    if f"{prefix}.cross_attention.emb_rel_k" in params:
        ca.emb_rel_k = params[f"{prefix}.cross_attention.emb_rel_k"]
        ca.emb_rel_v = params[f"{prefix}.cross_attention.emb_rel_v"]


def _bind_ref_enc(re_, params: dict) -> None:
    re_.spectral_0.weight = params["ref_enc.spectral.0.fc.weight"]
    re_.spectral_0.bias = params["ref_enc.spectral.0.fc.bias"]
    re_.spectral_1.weight = params["ref_enc.spectral.3.fc.weight"]
    re_.spectral_1.bias = params["ref_enc.spectral.3.fc.bias"]
    for i, t in enumerate((re_.temporal_0, re_.temporal_1)):
        t.w_1 = params[f"ref_enc.temporal.{i}.conv1.conv.weight"]
        t.b_1 = params[f"ref_enc.temporal.{i}.conv1.conv.bias"]
    sa = re_.slf_attn
    sa.w_qs.weight = params["ref_enc.slf_attn.w_qs.weight"]
    sa.w_qs.bias = params["ref_enc.slf_attn.w_qs.bias"]
    sa.w_ks.weight = params["ref_enc.slf_attn.w_ks.weight"]
    sa.w_ks.bias = params["ref_enc.slf_attn.w_ks.bias"]
    sa.w_vs.weight = params["ref_enc.slf_attn.w_vs.weight"]
    sa.w_vs.bias = params["ref_enc.slf_attn.w_vs.bias"]
    sa.fc.weight = params["ref_enc.slf_attn.fc.weight"]
    sa.fc.bias = params["ref_enc.slf_attn.fc.bias"]
    re_.fc.weight = params["ref_enc.fc.fc.weight"]
    re_.fc.bias = params["ref_enc.fc.fc.bias"]


class _WeightNormMixin:
    """Live weight-norm for Conv1d-style layers: params (weight_g, weight_v).

    torch.nn.utils.weight_norm(dim=0): w = g * v / ||v||_2 over all dims
    except dim 0. Norm computed in fp32; effective weight cast to the
    compute dtype at use. When only `.weight` is set (no _v), behaves as a
    plain conv (inference-export style).
    """

    def _effective_weight(self, dtype):
        if getattr(self, "weight_v", None) is None:
            return self.weight.astype(dtype)
        v32 = self.weight_v.astype(mx.float32)
        norm = mx.sqrt(mx.sum(v32 * v32, axis=(1, 2), keepdims=True))
        w = self.weight_g[:, None, None].astype(mx.float32) * v32 / norm
        return w.astype(dtype)


class WNConv1d(_WeightNormMixin):
    """Conv1d on (B,C,T) with optional live weight norm."""

    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, dilation=1):
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.weight = None
        self.weight_g = self.weight_v = None
        self.bias = mx.zeros((out_ch,))

    def __call__(self, x):
        dtype = x.dtype
        x = mx.transpose(x, (0, 2, 1))
        if self.padding:
            x = mx.pad(x, [(0, 0), (self.padding, self.padding), (0, 0)])
        out = mx.conv1d(x, self._effective_weight(dtype), stride=self.stride,
                        padding=0, dilation=self.dilation)
        out = mx.transpose(out, (0, 2, 1))
        return out + self.bias[None, :, None].astype(dtype)


class WNConvTranspose1d(_WeightNormMixin):
    """ConvTranspose1d on (B,C,T) with optional live weight norm.

    torch weight (in, out, k); MLX conv_transpose1d weight (in, k, out)... we
    store the MLX-layout (out, k, in) like the exports do and transpose to
    the kernel layout at call.
    """

    def __init__(self):
        self.weight = None  # stored MLX-layout (out, k, in) post-extract
        self.weight_g = self.weight_v = None
        self.bias = None
        self.stride = 1
        self.padding = 0

    def __call__(self, x):
        dtype = x.dtype
        w = self._effective_weight(dtype)  # (out, k, in)
        w_t = mx.transpose(w, (2, 0, 1))   # -> (in, k, out) kernel layout
        x = mx.transpose(x, (0, 2, 1))
        out = mx.conv_transpose1d(x, w_t, stride=self.stride, padding=self.padding)
        out = mx.transpose(out, (0, 2, 1))
        return out + self.bias[None, :, None].astype(dtype)


class PosteriorEncoderTrain:
    """module.models.PosteriorEncoder: z = (m + randn*exp(logs))*mask; g detached."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int,
                 kernel_size: int = 5, dilation_rate: int = 1, n_layers: int = 16,
                 gin_channels: int = 0):
        self.pre = WNConv1d(in_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      gin_channels=gin_channels)
        self.proj = WNConv1d(hidden_channels, out_channels * 2, 1)
        self.out_channels = out_channels

    def bind(self, params: dict) -> None:
        self.pre.weight = params["enc_q.pre.weight"]
        self.pre.bias = params["enc_q.pre.bias"]
        bind_wn(self.enc, params, "enc_q.enc")
        self.proj.weight = params["enc_q.proj.weight"]
        self.proj.bias = params["enc_q.proj.bias"]

    def __call__(self, x, x_lengths, g=None, key=None):
        if g is not None:
            g = mx.stop_gradient(g)
        x_mask = sequence_mask(x_lengths, x.shape[2]).astype(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = stats[:, : self.out_channels], stats[:, self.out_channels:]
        noise = mx.random.normal(m.shape, key=key)
        z = (m + noise * mx.exp(logs)) * x_mask
        return z, m, logs, x_mask


# ---------------------------------------------------------------------------
# training SynthesizerTrn
# ---------------------------------------------------------------------------

class SynthesizerTrnTrain:
    """Training-mode SynthesizerTrn driven by a flat torch-name param dict."""

    def __init__(self, version: str, spec_channels: int = 1025,
                 segment_size: int = 32, inter_channels: int = 192,
                 hidden_channels: int = 192, filter_channels: int = 768,
                 n_heads: int = 2, n_layers: int = 6, kernel_size: int = 3,
                 resblock: str = "1", resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5),) * 3,
                 upsample_rates=(10, 8, 2, 2, 2), upsample_initial_channel: int = 512,
                 upsample_kernel_sizes=(16, 16, 8, 2, 2), gin_channels: int = 512,
                 semantic_frame_rate: str = "25hz", freeze_quantizer: bool = True,
                 n_layers_q: int = 16):
        self.version = version
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.semantic_frame_rate = semantic_frame_rate
        self.freeze_quantizer = freeze_quantizer
        self.is_v2pro = version in ("v2Pro", "v2ProPlus")
        n_symbols = 322 if version == "v1" else 732

        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels,
                                 n_heads, n_layers, kernel_size, 0.1,
                                 version=version, n_symbols=n_symbols)
        self.dec = Generator(inter_channels, resblock, list(resblock_kernel_sizes),
                             [list(d) for d in resblock_dilation_sizes],
                             list(upsample_rates), upsample_initial_channel,
                             list(upsample_kernel_sizes), gin_channels=gin_channels)
        self.enc_q = PosteriorEncoderTrain(spec_channels, inter_channels,
                                           hidden_channels, 5, 1, n_layers_q,
                                           gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4,
                                          gin_channels=gin_channels)
        self.ref_enc = MelStyleEncoder(spec_channels if version == "v1" else 704,
                                       style_vector_dim=gin_channels)
        ssl_dim = 768
        assert semantic_frame_rate in ("25hz", "50hz")
        if semantic_frame_rate == "25hz":
            self.ssl_proj = Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = Conv1d(ssl_dim, ssl_dim, 1, stride=1)
        self.quantizer_embed: mx.array | None = None
        self.sv_emb_weight = self.sv_emb_bias = None
        self.ge_to512_weight = self.ge_to512_bias = None
        self.prelu_weight = None

    # -- bind the flat dict ---------------------------------------------------
    def bind(self, params: dict) -> None:
        p = params
        g = p.get
        ep = self.enc_p
        ep.ssl_proj.weight = g("enc_p.ssl_proj.weight")
        ep.ssl_proj.bias = g("enc_p.ssl_proj.bias")
        ep.text_embedding = g("enc_p.text_embedding.weight")
        ep.proj.weight = g("enc_p.proj.weight")
        ep.proj.bias = g("enc_p.proj.bias")
        _bind_encoder(ep.encoder_ssl, p, "enc_p.encoder_ssl")
        _bind_encoder(ep.encoder_text, p, "enc_p.encoder_text")
        _bind_encoder(ep.encoder2, p, "enc_p.encoder2")
        _bind_mrte(ep.mrte, p, "enc_p.mrte")
        dec = self.dec
        dec.conv_pre.weight = g("dec.conv_pre.weight")
        dec.conv_pre.bias = g("dec.conv_pre.bias")
        for i in range(len(dec.ups)):
            up = dec.ups[i]
            if f"dec.ups.{i}.weight_g" in p:
                up.weight_g = g(f"dec.ups.{i}.weight_g").reshape(-1)
                up.weight_v = g(f"dec.ups.{i}.weight_v")
            else:
                up.weight = g(f"dec.ups.{i}.weight")
            up.bias = g(f"dec.ups.{i}.bias")
        for i, rb in enumerate(dec.resblocks):
            for cn, blocks in (("convs1", rb.convs1), ("convs2", rb.convs2)):
                for j, cv in enumerate(blocks):
                    pre = f"dec.resblocks.{i}.{cn}.{j}"
                    if f"{pre}.weight_g" in p:
                        cv.weight_g = g(f"{pre}.weight_g").reshape(-1)
                        cv.weight_v = g(f"{pre}.weight_v")
                    else:
                        cv.weight = g(f"{pre}.weight")
                    cv.bias = g(f"{pre}.bias")
        dec.cond.weight = g("dec.cond.weight")
        dec.cond.bias = g("dec.cond.bias")
        dec.conv_post.weight = g("dec.conv_post.weight")
        dec.conv_post.bias = g("dec.conv_post.bias") \
            if "dec.conv_post.bias" in p else mx.zeros((1,))  # is_bias=False
        fl = self.flow
        wn_layers = [f for f in fl.flows if hasattr(f, "enc")]
        for fi, rc in enumerate(wn_layers):
            rc.pre.weight = g(f"flow.flows.{fi*2}.pre.weight")
            rc.pre.bias = g(f"flow.flows.{fi*2}.pre.bias")
            rc.post.weight = g(f"flow.flows.{fi*2}.post.weight")
            rc.post.bias = g(f"flow.flows.{fi*2}.post.bias")
            bind_wn(rc.enc, p, f"flow.flows.{fi*2}.enc")
        self.enc_q.bind(p)
        _bind_ref_enc(self.ref_enc, p)
        self.ssl_proj.weight = g("ssl_proj.weight")
        self.ssl_proj.bias = g("ssl_proj.bias")
        self.quantizer_embed = g("quantizer.vq.layers.0._codebook.embed")
        if self.is_v2pro:
            self.sv_emb_weight = g("sv_emb.weight")
            self.sv_emb_bias = g("sv_emb.bias")
            self.ge_to512_weight = g("ge_to512.weight")
            self.ge_to512_bias = g("ge_to512.bias")
            self.prelu_weight = g("prelu.weight").reshape(1, -1, 1)

    # -- forward (official SynthesizerTrn.forward) ---------------------------
    def forward(self, ssl, y, y_lengths, text, text_lengths,
                sv_emb=None, key: mx.array | None = None, quantized_in: mx.array | None = None):
        """Returns (y_hat, kl_ssl, ids_slice, y_mask, (z,z_p,m_p,logs_p,m_q,logs_q),
        stats_ssl).

        kl_ssl is the RVQ commit loss VALUE (official: mse(quantized, x)
        computed even with freeze_quantizer=True — net_g is in train mode, so
        the VQ layer's self.training branch runs inside the no_grad block;
        it contributes no gradients but enters loss_gen_all with weight 1).
        """
        y_mask = sequence_mask(y_lengths, y.shape[2]).astype(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
        if self.is_v2pro:
            sv = sv_emb.astype(ge.dtype) @ self.sv_emb_weight.T.astype(ge.dtype) \
                + self.sv_emb_bias.astype(ge.dtype)
            ge = ge + sv[:, :, None]
            ge = mx.where(ge > 0, ge, 0.25 * (mx.exp(mx.minimum(ge, 60.0)) - 1.0))
            ge512 = mx.transpose(
                mx.transpose(ge, (0, 2, 1)) @ self.ge_to512_weight.T.astype(ge.dtype)
                + self.ge_to512_bias.astype(ge.dtype), (0, 2, 1))
        # frozen fp32 quantizer block (autocast disabled in official).
        # freeze_quantizer=True (official default): forward calls
        # quantizer.eval() each step -> VQ self.training branches (commit,
        # EMA) never run -> kl_ssl is EXACTLY 0.0 (official tensor([0.]));
        # loss_gen_all still adds kl_ssl * 1 verbatim.
        if quantized_in is None:
            quantized, kl_ssl = self._quantize_ssl(
                ssl, with_commit=not self.freeze_quantizer)
            if kl_ssl is None:
                kl_ssl = mx.zeros((), mx.float32)
        elif isinstance(quantized_in, tuple):
            quantized, kl_ssl = quantized_in
        else:
            quantized, kl_ssl = quantized_in, mx.zeros((), mx.float32)
        if self.semantic_frame_rate == "25hz":
            quantized = _nearest_interp(quantized, quantized.shape[-1] * 2)

        x, m_p, logs_p, y_mask2 = self.enc_p(
            quantized.astype(y.dtype), y_lengths, text, text_lengths,
            ge512 if self.is_v2pro else ge)
        z, m_q, logs_q, x_mask = self.enc_q(y, y_lengths, g=ge, key=key)
        z_p = self.flow(z, x_mask, g=ge)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size,
                                                 key=key)
        o = self.dec(z_slice, g=ge)
        return (o, kl_ssl, ids_slice, y_mask2,
                (z, z_p, m_p, logs_p, m_q, logs_q), quantized)

    # -- quantizer block (frozen fp32; official autocast-disabled) ------------
    def _quantize_ssl(self, ssl: mx.array, with_commit: bool = False):
        """ssl_proj + RVQ layer-0 nearest-code assignment.

        Returns (quantized (B,768,T'), commit_or_None). commit =
        mse(quantized, proj(ssl)); only computed when with_commit — the
        official only produces it when the quantizer stays in TRAIN mode
        (freeze_quantizer=False), since SynthesizerTrn.forward eval()s the
        quantizer each step when frozen, disabling the self.training
        commit/EMA branches (kl_ssl is then exactly 0.0).
        """
        ssl32 = ssl.astype(mx.float32)
        proj = self.ssl_proj(ssl32)
        xt = mx.transpose(proj, (0, 2, 1))
        embed = self.quantizer_embed.astype(mx.float32)
        d = (mx.sum(xt * xt, axis=-1, keepdims=True)
             + mx.sum(embed * embed, axis=-1)[None, None, :]
             - 2.0 * xt @ mx.transpose(embed))
        codes = mx.argmin(d, axis=-1)
        q = mx.transpose(embed[codes], (0, 2, 1))
        commit = mx.mean((q - proj) ** 2) if with_commit else None
        return mx.stop_gradient(q), commit

    def quantize_ssl(self, ssl: mx.array) -> tuple:
        """Returns (quantized, kl_ssl) — pass as forward(quantized_in=).

        kl_ssl follows official freeze_quantizer semantics: exactly 0.0
        when frozen (quantizer.eval() each forward), mse commit otherwise.
        """
        q, commit = self._quantize_ssl(ssl, with_commit=not self.freeze_quantizer)
        return q, commit if commit is not None else mx.zeros((), mx.float32)

    # -- parameter names -------------------------------------------------------
    def parameter_names(self, include_frozen: bool = False) -> list[str]:
        """Torch-checkpoint-style names of every trainable parameter.

        With freeze_quantizer=True (official default), ssl_proj.* and
        quantizer.* are EXCLUDED (official wraps them in no_grad; the 'not
        requires_grad' printout in s2_train.py lists exactly these). Pass
        include_frozen=True to enumerate all.
        """
        names = []
        ep = "enc_p."
        names += [f"{ep}ssl_proj.weight", f"{ep}ssl_proj.bias"]
        names += [f"{ep}text_embedding.weight"]
        names += [f"{ep}proj.weight", f"{ep}proj.bias"]
        for pre, n in ((f"{ep}encoder_ssl", len(self.enc_p.encoder_ssl.attn_layers)),
                       (f"{ep}encoder_text", len(self.enc_p.encoder_text.attn_layers)),
                       (f"{ep}encoder2", len(self.enc_p.encoder2.attn_layers))):
            names += _encoder_param_names(pre, n)
        names += _mrte_param_names(f"{ep}mrte")
        # dec (ups + resblocks are weight-norm: (weight_g, weight_v) pairs)
        names += ["dec.conv_pre.weight", "dec.conv_pre.bias"]
        for i in range(len(self.dec.ups)):
            names += [f"dec.ups.{i}.weight_g", f"dec.ups.{i}.weight_v",
                      f"dec.ups.{i}.bias"]
        for i in range(len(self.dec.resblocks)):
            for cn in ("convs1", "convs2"):
                for j in range(3):
                    names += [f"dec.resblocks.{i}.{cn}.{j}.weight_g",
                              f"dec.resblocks.{i}.{cn}.{j}.weight_v",
                              f"dec.resblocks.{i}.{cn}.{j}.bias"]
        names += ["dec.cond.weight", "dec.cond.bias",
                  "dec.conv_post.weight"]
        # enc_q (WN convs weight-norm)
        names += ["enc_q.pre.weight", "enc_q.pre.bias",
                  "enc_q.proj.weight", "enc_q.proj.bias"]
        names += _wn_param_names("enc_q.enc", len(self.enc_q.enc.in_layers),
                                 with_cond=True)
        # flow
        wn_flows = [f for f in self.flow.flows if hasattr(f, "enc")]
        for fi in range(len(wn_flows)):
            names += [f"flow.flows.{fi*2}.pre.weight", f"flow.flows.{fi*2}.pre.bias",
                      f"flow.flows.{fi*2}.post.weight", f"flow.flows.{fi*2}.post.bias"]
            names += _wn_param_names(f"flow.flows.{fi*2}.enc",
                                     len(wn_flows[fi].enc.in_layers), with_cond=True)
        # ref_enc
        names += ["ref_enc.spectral.0.fc.weight", "ref_enc.spectral.0.fc.bias",
                  "ref_enc.spectral.3.fc.weight", "ref_enc.spectral.3.fc.bias"]
        for i in (0, 1):
            names += [f"ref_enc.temporal.{i}.conv1.conv.weight",
                      f"ref_enc.temporal.{i}.conv1.conv.bias"]
        for nm in ("w_qs", "w_ks", "w_vs", "fc"):
            names += [f"ref_enc.slf_attn.{nm}.weight", f"ref_enc.slf_attn.{nm}.bias"]
        names += ["ref_enc.fc.fc.weight", "ref_enc.fc.fc.bias"]
        # v2Pro extras
        if self.is_v2pro:
            names += ["sv_emb.weight", "sv_emb.bias", "ge_to512.weight",
                      "ge_to512.bias", "prelu.weight"]
        if include_frozen or not self.freeze_quantizer:
            names += ["ssl_proj.weight", "ssl_proj.bias",
                      "quantizer.vq.layers.0._codebook.embed"]
        return names

    # names grouped for the official low-lr param groups
    def text_low_lr_names(self) -> list[str]:
        out = []
        out += ["enc_p.text_embedding.weight"]
        out += _encoder_param_names("enc_p.encoder_text",
                                    len(self.enc_p.encoder_text.attn_layers))
        out += _mrte_param_names("enc_p.mrte")
        return out


def _encoder_param_names(prefix: str, n_layers: int) -> list[str]:
    names = []
    for i in range(n_layers):
        for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
            names += [f"{prefix}.attn_layers.{i}.{nm}.weight",
                      f"{prefix}.attn_layers.{i}.{nm}.bias"]
        names += [f"{prefix}.attn_layers.{i}.emb_rel_k",
                  f"{prefix}.attn_layers.{i}.emb_rel_v"]
        names += [f"{prefix}.norm_layers_1.{i}.gamma",
                  f"{prefix}.norm_layers_1.{i}.beta",
                  f"{prefix}.norm_layers_2.{i}.gamma",
                  f"{prefix}.norm_layers_2.{i}.beta"]
        names += [f"{prefix}.ffn_layers.{i}.conv_1.weight",
                  f"{prefix}.ffn_layers.{i}.conv_1.bias",
                  f"{prefix}.ffn_layers.{i}.conv_2.weight",
                  f"{prefix}.ffn_layers.{i}.conv_2.bias"]
    return names


def _mrte_param_names(prefix: str) -> list[str]:
    names = []
    for nm in ("c_pre", "text_pre", "c_post"):
        names += [f"{prefix}.{nm}.weight", f"{prefix}.{nm}.bias"]
    for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
        names += [f"{prefix}.cross_attention.{nm}.weight",
                  f"{prefix}.cross_attention.{nm}.bias"]
    return names


def _wn_param_names(prefix: str, n_layers: int, with_cond: bool) -> list[str]:
    """WN conv param names with live weight-norm (g,v) pairs."""
    names = []
    if with_cond:
        names += [f"{prefix}.cond_layer.weight_g", f"{prefix}.cond_layer.weight_v",
                  f"{prefix}.cond_layer.bias"]
    for i in range(n_layers):
        for layer in ("in_layers", "res_skip_layers"):
            names += [f"{prefix}.{layer}.{i}.weight_g",
                      f"{prefix}.{layer}.{i}.weight_v",
                      f"{prefix}.{layer}.{i}.bias"]
    return names


# ---------------------------------------------------------------------------
# ExponentialLR (torch semantics, per-epoch)
# ---------------------------------------------------------------------------

class ExponentialLR:
    """torch ExponentialLR: lr = base_lr * gamma^last_epoch (epoch-stepped)."""

    def __init__(self, optimizer, gamma: float = 0.999875, last_epoch: int = -1):
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.step()  # torch calls step() once at construction (last_epoch=-1)

    def step(self):
        self.last_epoch += 1
        factor = self.gamma ** self.last_epoch
        for g in self.optimizer.param_groups:
            g["lr"] = g.get("base_lr", g["lr"]) * factor
