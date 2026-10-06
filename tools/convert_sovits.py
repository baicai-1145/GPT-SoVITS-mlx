"""PyTorch checkpoint -> MLX safetensors converters.

torch is imported lazily here ONLY; the runtime (gsovits_mlx.*) never touches it.
Weight-norm conv weights are pre-fused: w = weight_g * (weight_v / ||weight_v||_per_out).
"""

from __future__ import annotations

import json
import os
import re
import sys
import types
from typing import Callable

import numpy as np



def to_mlx_conv1d(a):
    """torch Conv1d weight (out,in,k) -> MLX (out,k,in)."""
    return np.ascontiguousarray(np.transpose(a, (0, 2, 1)))


def to_mlx_conv1d_t(a):
    """torch ConvTranspose1d weight (in,out,k) -> MLX (out,k,in)."""
    return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))


def _fuse_wn(t):
    """t has weight_g (out, 1, *rest) and weight_v (out, in, *rest) -> fused (out, in, *rest)."""
    g = t["weight_g"]
    v = t["weight_v"]
    norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1, keepdims=True)
    fused = g.reshape(-1, *([1] * (v.ndim - 1))) * v / norm.reshape(-1, *([1] * (v.ndim - 1)))
    return fused.astype(np.float32)


def convert_state_dict(sd: dict, mapping: dict[str, str | Callable]) -> dict:
    """mapping: dst_name -> src_name or callable(sd)->array."""
    out = {}
    for dst, src in mapping.items():
        if callable(src):
            out[dst] = src(sd)
        else:
            out[dst] = sd[src]
    return out


def save_safetensors(path: str, arrays: dict[str, "np.ndarray"], metadata: dict | None = None):
    from safetensors.numpy import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in arrays.items()}, path,
              metadata=metadata or None)


# ---------------------------------------------------------------------------
# GPT s1 (AR t2s)
# ---------------------------------------------------------------------------

def convert_s1(ckpt_path: str, out_dir: str, fp16: bool = True):
    """s1v3.ckpt / s1bert25hz*.ckpt -> gpt.safetensors + gpt.json."""
    import torch

    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = d["weight"]
    cfg = d["config"]
    arrays: dict[str, np.ndarray] = {}

    def t(k):
        return sd[k].float().numpy()

    arrays["ar_text_embedding"] = t("model.ar_text_embedding.word_embeddings.weight")
    arrays["ar_audio_embedding"] = t("model.ar_audio_embedding.word_embeddings.weight")
    arrays["bert_proj.weight"] = t("model.bert_proj.weight")
    arrays["bert_proj.bias"] = t("model.bert_proj.bias")
    arrays["ar_text_position.alpha"] = t("model.ar_text_position.alpha")
    arrays["ar_audio_position.alpha"] = t("model.ar_audio_position.alpha")
    arrays["ar_predict_layer.weight"] = t("model.ar_predict_layer.weight")

    n_layer = cfg["model"]["n_layer"]
    for i in range(n_layer):
        pre = f"model.h.layers.{i}."
        dst = f"block.{i}."
        qkv_w = sd[pre + "self_attn.in_proj_weight"].float().numpy()
        qkv_b = sd[pre + "self_attn.in_proj_bias"].float().numpy()
        arrays[dst + "qkv_w"] = qkv_w
        arrays[dst + "qkv_b"] = qkv_b
        arrays[dst + "out_w"] = t(pre + "self_attn.out_proj.weight")
        arrays[dst + "out_b"] = t(pre + "self_attn.out_proj.bias")
        if pre + "norm1.project_layer.weight" in sd:
            arrays[dst + "norm1.w"] = t(pre + "norm1.project_layer.weight")
            arrays[dst + "norm1.b"] = t(pre + "norm1.project_layer.bias")
            arrays[dst + "norm2.w"] = t(pre + "norm2.project_layer.weight")
            arrays[dst + "norm2.b"] = t(pre + "norm2.project_layer.bias")
        else:
            arrays[dst + "norm1.g"] = t(pre + "norm1.weight")
            arrays[dst + "norm1.b"] = t(pre + "norm1.bias")
            arrays[dst + "norm2.g"] = t(pre + "norm2.weight")
            arrays[dst + "norm2.b"] = t(pre + "norm2.bias")
        arrays[dst + "mlp1.w"] = t(pre + "linear1.weight")
        arrays[dst + "mlp1.b"] = t(pre + "linear1.bias")
        arrays[dst + "mlp2.w"] = t(pre + "linear2.weight")
        arrays[dst + "mlp2.b"] = t(pre + "linear2.bias")

    if fp16:
        arrays = {k: v.astype(np.float16) for k, v in arrays.items()}
    save_safetensors(os.path.join(out_dir, "gpt.safetensors"), arrays)
    with open(os.path.join(out_dir, "gpt.json"), "w") as f:
        json.dump({"config": cfg, "n_layer": n_layer}, f)
    return len(arrays)


# ---------------------------------------------------------------------------
# SoVITS v1 / v2 / v2Pro / v2ProPlus
# ---------------------------------------------------------------------------

def _ensure_hparams_shim():
    """Legacy pretrained ckpts (s2G488k.pth) unpickle a top-level `utils.HParams`.

    Inside the official repo GPT_SoVITS/utils.py provides that class; standalone
    runs register a minimal same-named attribute container so torch.load can
    resolve the pickle reference. Mirrors GPT_SoVITS/utils.py HParams (data in
    instance attrs, getitem->getattr) plus a dict-like .get used by converters.
    """
    try:
        import importlib
        importlib.import_module("utils")
        return
    except ImportError:
        pass
    if "utils" in sys.modules:
        return

    class HParams:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                self[k] = v

        def keys(self):
            return self.__dict__.keys()

        def items(self):
            return self.__dict__.items()

        def values(self):
            return self.__dict__.values()

        def __len__(self):
            return len(self.__dict__)

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __contains__(self, key):
            return key in self.__dict__

        def get(self, key, default=None):
            return getattr(self, key, default)

        def __repr__(self):
            return self.__dict__.__repr__()

    shim = types.ModuleType("utils")
    shim.HParams = HParams
    sys.modules["utils"] = shim


def _load_sovits_ckpt(path: str):
    """Handles the zip-mislabelled checkpoints (PK header check like process_ckpt)."""
    import torch
    _ensure_hparams_shim()
    from io import BytesIO
    with open(path, "rb") as f:
        meta = f.read(2)
        if meta != b"PK":
            data = b"PK" + f.read()
            d = torch.load(BytesIO(data), map_location="cpu", weights_only=False)
        else:
            d = torch.load(path, map_location="cpu", weights_only=False)
    return d


def convert_sovits_v1v2(path: str, out_dir: str, version: str, sv_path: str | None = None,
                        fp16: bool = True):
    """s2G488k / s2G2333k / s2Gv2Pro(Plus) -> sovits.safetensors + sovits.json."""
    import torch

    d = _load_sovits_ckpt(path)
    sd = d["weight"]
    hps = d["config"]
    arrays: dict[str, np.ndarray] = {}

    def t(k):
        return sd[k].float().numpy()

    def wn(k):
        return _fuse_wn({"weight_g": sd[k + ".weight_g"].float().numpy(),
                         "weight_v": sd[k + ".weight_v"].float().numpy()})

    # ---- enc_p (TextEncoder) ----
    arrays["enc_p.ssl_proj.weight"] = to_mlx_conv1d(t("enc_p.ssl_proj.weight"))
    arrays["enc_p.ssl_proj.bias"] = t("enc_p.ssl_proj.bias")
    arrays["enc_p.text_embedding"] = t("enc_p.text_embedding.weight")
    arrays["enc_p.proj.weight"] = to_mlx_conv1d(t("enc_p.proj.weight"))
    arrays["enc_p.proj.bias"] = t("enc_p.proj.bias")
    n = lambda pre, name, cnt: [f"{pre}{name}.{i}." for i in range(cnt)]
    # count layers from keys
    ssl_layers = len({k.split(".")[2] for k in sd if k.startswith("enc_p.encoder_ssl")}) or 0
    # simpler: find max index
    def max_idx(prefix):
        idxs = [int(m.group(1)) for k in sd for m in [re.match(rf"{prefix}\.(\d+)\.", k)] if m]
        return (max(idxs) + 1) if idxs else 0

    def conv_attn(dst_prefix, src_prefix, n_layers, window=4):
        p = f"{src_prefix}."
        for i in range(n_layers):
            dst = f"{dst_prefix}.{i}."
            for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
                arrays[dst + f"attn.{nm}.weight"] = to_mlx_conv1d(t(p + f"attn_layers.{i}.{nm}.weight"))
                arrays[dst + f"attn.{nm}.bias"] = t(p + f"attn_layers.{i}.{nm}.bias")
            if p + f"attn_layers.{i}.emb_rel_k" in sd:
                arrays[dst + "attn.emb_rel_k"] = t(p + f"attn_layers.{i}.emb_rel_k")
                arrays[dst + "attn.emb_rel_v"] = t(p + f"attn_layers.{i}.emb_rel_v")
            arrays[dst + "norm1"] = t(p + f"norm_layers_1.{i}.gamma")
            arrays[dst + "norm1.b"] = t(p + f"norm_layers_1.{i}.beta")
            arrays[dst + "norm2"] = t(p + f"norm_layers_2.{i}.gamma")
            arrays[dst + "norm2.b"] = t(p + f"norm_layers_2.{i}.beta")
            arrays[dst + "ffn.conv1.weight"] = to_mlx_conv1d(t(p + f"ffn_layers.{i}.conv_1.weight"))
            arrays[dst + "ffn.conv1.bias"] = t(p + f"ffn_layers.{i}.conv_1.bias")
            arrays[dst + "ffn.conv2.weight"] = to_mlx_conv1d(t(p + f"ffn_layers.{i}.conv_2.weight"))
            arrays[dst + "ffn.conv2.bias"] = t(p + f"ffn_layers.{i}.conv_2.bias")

    conv_attn("enc_p.enc_ssl", "enc_p.encoder_ssl", max_idx("enc_p.encoder_ssl.attn_layers"))
    conv_attn("enc_p.enc_text", "enc_p.encoder_text", max_idx("enc_p.encoder_text.attn_layers"))
    conv_attn("enc_p.enc2", "enc_p.encoder2", max_idx("enc_p.encoder2.attn_layers"))

    arrays["enc_p.mrte.c_pre.weight"] = to_mlx_conv1d(t("enc_p.mrte.c_pre.weight"))
    arrays["enc_p.mrte.c_pre.bias"] = t("enc_p.mrte.c_pre.bias")
    arrays["enc_p.mrte.text_pre.weight"] = to_mlx_conv1d(t("enc_p.mrte.text_pre.weight"))
    arrays["enc_p.mrte.text_pre.bias"] = t("enc_p.mrte.text_pre.bias")
    arrays["enc_p.mrte.c_post.weight"] = to_mlx_conv1d(t("enc_p.mrte.c_post.weight"))
    arrays["enc_p.mrte.c_post.bias"] = t("enc_p.mrte.c_post.bias")
    arrays["enc_p.mrte.cross_attn.conv_q.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_q.weight"))
    arrays["enc_p.mrte.cross_attn.conv_q.bias"] = t("enc_p.mrte.cross_attention.conv_q.bias")
    arrays["enc_p.mrte.cross_attn.conv_k.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_k.weight"))
    arrays["enc_p.mrte.cross_attn.conv_k.bias"] = t("enc_p.mrte.cross_attention.conv_k.bias")
    arrays["enc_p.mrte.cross_attn.conv_v.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_v.weight"))
    arrays["enc_p.mrte.cross_attn.conv_v.bias"] = t("enc_p.mrte.cross_attention.conv_v.bias")
    arrays["enc_p.mrte.cross_attn.conv_o.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_o.weight"))
    arrays["enc_p.mrte.cross_attn.conv_o.bias"] = t("enc_p.mrte.cross_attention.conv_o.bias")
    if "enc_p.mrte.cross_attention.emb_rel_k" in sd:
        arrays["enc_p.mrte.cross_attn.emb_rel_k"] = t("enc_p.mrte.cross_attention.emb_rel_k")
        arrays["enc_p.mrte.cross_attn.emb_rel_v"] = t("enc_p.mrte.cross_attention.emb_rel_v")

    # ---- ref_enc (MelStyleEncoder) ----
    arrays["ref_enc.spectral.0.weight"] = t("ref_enc.spectral.0.fc.weight")
    arrays["ref_enc.spectral.0.bias"] = t("ref_enc.spectral.0.fc.bias")
    arrays["ref_enc.spectral.3.weight"] = t("ref_enc.spectral.3.fc.weight")
    arrays["ref_enc.spectral.3.bias"] = t("ref_enc.spectral.3.fc.bias")
    for i in (0, 1):
        arrays[f"ref_enc.temporal.{i}.w1"] = np.transpose(t(f"ref_enc.temporal.{i}.conv1.conv.weight"), (0, 2, 1))
        arrays[f"ref_enc.temporal.{i}.b1"] = t(f"ref_enc.temporal.{i}.conv1.conv.bias")
    arrays["ref_enc.slf_attn.w_qs"] = t("ref_enc.slf_attn.w_qs.weight")
    arrays["ref_enc.slf_attn.w_qs.b"] = t("ref_enc.slf_attn.w_qs.bias")
    arrays["ref_enc.slf_attn.w_ks"] = t("ref_enc.slf_attn.w_ks.weight")
    arrays["ref_enc.slf_attn.w_ks.b"] = t("ref_enc.slf_attn.w_ks.bias")
    arrays["ref_enc.slf_attn.w_vs"] = t("ref_enc.slf_attn.w_vs.weight")
    arrays["ref_enc.slf_attn.w_vs.b"] = t("ref_enc.slf_attn.w_vs.bias")
    arrays["ref_enc.slf_attn.fc"] = t("ref_enc.slf_attn.fc.weight")
    arrays["ref_enc.slf_attn.fc.b"] = t("ref_enc.slf_attn.fc.bias")
    arrays["ref_enc.fc"] = t("ref_enc.fc.fc.weight")
    arrays["ref_enc.fc.b"] = t("ref_enc.fc.fc.bias")

    # ---- dec (Generator) ----
    arrays["dec.conv_pre.weight"] = to_mlx_conv1d(t("dec.conv_pre.weight"))
    arrays["dec.conv_pre.bias"] = t("dec.conv_pre.bias")
    _up_idxs = [int(re.match(r"dec\.ups\.(\d+)\.", k).group(1))
                for k in sd if re.match(r"dec\.ups\.(\d+)\.", k)]
    n_up = (max(_up_idxs) + 1) if _up_idxs else 0
    for i in range(n_up):
        if f"dec.ups.{i}.weight" in sd:
            arrays[f"dec.ups.{i}.weight"] = to_mlx_conv1d_t(t(f"dec.ups.{i}.weight"))
            arrays[f"dec.ups.{i}.bias"] = t(f"dec.ups.{i}.bias")
        else:
            arrays[f"dec.ups.{i}.weight"] = to_mlx_conv1d_t(wn(f"dec.ups.{i}"))
            arrays[f"dec.ups.{i}.bias"] = t(f"dec.ups.{i}.bias")
    n_rb = max_idx("dec.resblocks") + 1
    for i in range(n_rb):
        for cn in ("convs1", "convs2"):
            idxs = [int(re.match(rf"dec\.resblocks\.{i}\.{cn}\.(\d+)\.", k).group(1))
                    for k in sd
                    if re.match(rf"dec\.resblocks\.{i}\.{cn}\.(\d+)\.", k)]
            nc = (max(idxs) + 1) if idxs else 0
            for j in range(nc):
                key = f"dec.resblocks.{i}.{cn}.{j}"
                if f"{key}.weight" in sd:
                    arrays[f"{key}.weight"] = to_mlx_conv1d(t(f"{key}.weight"))
                else:
                    arrays[f"{key}.weight"] = to_mlx_conv1d(wn(key))
                arrays[f"{key}.bias"] = t(f"{key}.bias")
    arrays["dec.cond.weight"] = to_mlx_conv1d(t("dec.cond.weight"))
    arrays["dec.cond.bias"] = t("dec.cond.bias")
    arrays["dec.conv_post.weight"] = to_mlx_conv1d(t("dec.conv_post.weight"))
    if "dec.conv_post.bias" in sd:
        arrays["dec.conv_post.bias"] = t("dec.conv_post.bias")
    else:  # fused as no-bias conv or via weight_norm
        arrays["dec.conv_post.bias"] = np.zeros(arrays["dec.conv_post.weight"].shape[0], np.float32)

    # ---- flow (weight-norm fused); top-level flow.pre/post only exist in v1 ----
    if "flow.pre.weight" in sd:
        arrays["flow.pre.weight"] = to_mlx_conv1d(t("flow.pre.weight"))
        arrays["flow.pre.bias"] = t("flow.pre.bias")
        arrays["flow.post.weight"] = to_mlx_conv1d(t("flow.post.weight"))
        arrays["flow.post.bias"] = t("flow.post.bias")
    n_flow = len([k for k in sd if re.match(r"flow\.flows\.\d+\.enc$", k) or False])
    n_flow = max(int(k.split(".")[2]) for k in sd if k.startswith("flow.flows.")) // 2 + 1
    for fi in range(n_flow):
        base = f"flow.flows.{fi * 2}.enc."
        dst = f"flow.{fi}.enc."
        if base + "pre.weight" in sd:
            arrays[dst + "pre.weight"] = to_mlx_conv1d(t(base + "pre.weight"))
            arrays[dst + "pre.bias"] = t(base + "pre.bias")
            arrays[dst + "post.weight"] = to_mlx_conv1d(t(base + "post.weight"))
            arrays[dst + "post.bias"] = t(base + "post.bias")
        # v2/v3 ckpts: layer-level pre/post outside enc
        elif f"flow.flows.{fi * 2}.pre.weight" in sd:
            arrays[f"flow.{fi}.pre.weight"] = to_mlx_conv1d(t(f"flow.flows.{fi * 2}.pre.weight"))
            arrays[f"flow.{fi}.pre.bias"] = t(f"flow.flows.{fi * 2}.pre.bias")
            arrays[f"flow.{fi}.post.weight"] = to_mlx_conv1d(t(f"flow.flows.{fi * 2}.post.weight"))
            arrays[f"flow.{fi}.post.bias"] = t(f"flow.flows.{fi * 2}.post.bias")
        arrays[dst + "cond"] = np.ascontiguousarray(np.transpose(wn(base + "cond_layer"), (0, 2, 1)))
        arrays[dst + "cond.b"] = t(base + "cond_layer.bias")
        n_wn = max_idx(base + "in_layers")
        for wi in range(n_wn):
            arrays[dst + f"in.{wi}"] = np.ascontiguousarray(np.transpose(wn(base + f"in_layers.{wi}"), (0, 2, 1)))
            arrays[dst + f"in.{wi}.b"] = t(base + f"in_layers.{wi}.bias")
            arrays[dst + f"skip.{wi}"] = np.ascontiguousarray(np.transpose(wn(base + f"res_skip_layers.{wi}"), (0, 2, 1)))
            arrays[dst + f"skip.{wi}.b"] = t(base + f"res_skip_layers.{wi}.bias")

    # ---- quantizer ----
    arrays["quantizer.codebook"] = t("quantizer.vq.layers.0._codebook.embed")

    # ---- ssl_proj (25hz) & enc_q are dropped (inference) ----
    if hps["model"].get("semantic_frame_rate") == "25hz":
        arrays["ssl_proj.weight"] = to_mlx_conv1d(t("ssl_proj.weight"))
        arrays["ssl_proj.bias"] = t("ssl_proj.bias")

    # ---- v2Pro sv projection ----
    meta = {"version": version}
    if version in ("v2Pro", "v2ProPlus"):
        arrays["sv_emb.weight"] = t("sv_emb.weight")
        arrays["sv_emb.bias"] = t("sv_emb.bias")
        arrays["ge_to512.weight"] = t("ge_to512.weight")
        arrays["ge_to512.bias"] = t("ge_to512.bias")
        arrays["prelu.weight"] = t("prelu.weight")

    if fp16:
        arrays = {k: v.astype(np.float16) for k, v in arrays.items()}
    save_safetensors(os.path.join(out_dir, "sovits.safetensors"), arrays)
    meta["model_hps"] = {k: hps["model"].get(k) for k in (
        "inter_channels", "hidden_channels", "filter_channels", "n_heads", "n_layers",
        "kernel_size", "p_dropout", "resblock", "resblock_kernel_sizes",
        "resblock_dilation_sizes", "upsample_rates", "upsample_initial_channel",
        "upsample_kernel_sizes", "gin_channels", "semantic_frame_rate")}
    meta["data"] = {"sampling_rate": hps["data"]["sampling_rate"]}
    with open(os.path.join(out_dir, "sovits.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return len(arrays)


# ---------------------------------------------------------------------------
# SoVITS v3 / v4 / v5dev / v5turbo (SynthesizerTrnV3)
# ---------------------------------------------------------------------------

def convert_sovits_v3v5(path: str, out_dir: str, version: str, fp16: bool = True):
    import torch

    d = _load_sovits_ckpt(path)
    sd = d["weight"]
    hps = d["config"]
    arrays: dict[str, np.ndarray] = {}

    def t(k):
        return sd[k].float().numpy()

    def wn(k):
        return _fuse_wn({"weight_g": sd[k + ".weight_g"].float().numpy(),
                         "weight_v": sd[k + ".weight_v"].float().numpy()})

    # enc_p identical layout to v2
    arrays["enc_p.ssl_proj.weight"] = to_mlx_conv1d(t("enc_p.ssl_proj.weight"))
    arrays["enc_p.ssl_proj.bias"] = t("enc_p.ssl_proj.bias")
    arrays["enc_p.text_embedding"] = t("enc_p.text_embedding.weight")
    arrays["enc_p.proj.weight"] = to_mlx_conv1d(t("enc_p.proj.weight"))
    arrays["enc_p.proj.bias"] = t("enc_p.proj.bias")

    def max_idx(prefix):
        idxs = [int(m.group(1)) for k in sd for m in [re.match(rf"{prefix}\.(\d+)\.", k)] if m]
        return (max(idxs) + 1) if idxs else 0

    def conv_attn(dst_prefix, src_prefix, n_layers):
        for i in range(n_layers):
            p = f"{src_prefix}."
            dst = f"{dst_prefix}.{i}."
            for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
                arrays[dst + f"attn.{nm}.weight"] = to_mlx_conv1d(t(p + f"attn_layers.{i}.{nm}.weight"))
                arrays[dst + f"attn.{nm}.bias"] = t(p + f"attn_layers.{i}.{nm}.bias")
            if p + f"attn_layers.{i}.emb_rel_k" in sd:
                arrays[dst + "attn.emb_rel_k"] = t(p + f"attn_layers.{i}.emb_rel_k")
                arrays[dst + "attn.emb_rel_v"] = t(p + f"attn_layers.{i}.emb_rel_v")
            arrays[dst + "norm1"] = t(p + f"norm_layers_1.{i}.gamma")
            arrays[dst + "norm1.b"] = t(p + f"norm_layers_1.{i}.beta")
            arrays[dst + "norm2"] = t(p + f"norm_layers_2.{i}.gamma")
            arrays[dst + "norm2.b"] = t(p + f"norm_layers_2.{i}.beta")
            arrays[dst + "ffn.conv1.weight"] = to_mlx_conv1d(t(p + f"ffn_layers.{i}.conv_1.weight"))
            arrays[dst + "ffn.conv1.bias"] = t(p + f"ffn_layers.{i}.conv_1.bias")
            arrays[dst + "ffn.conv2.weight"] = to_mlx_conv1d(t(p + f"ffn_layers.{i}.conv_2.weight"))
            arrays[dst + "ffn.conv2.bias"] = t(p + f"ffn_layers.{i}.conv_2.bias")

    conv_attn("enc_p.enc_ssl", "enc_p.encoder_ssl", max_idx("enc_p.encoder_ssl.attn_layers"))
    conv_attn("enc_p.enc_text", "enc_p.encoder_text", max_idx("enc_p.encoder_text.attn_layers"))
    conv_attn("enc_p.enc2", "enc_p.encoder2", max_idx("enc_p.encoder2.attn_layers"))
    for nm in ("c_pre", "text_pre", "c_post"):
        arrays[f"enc_p.mrte.{nm}.weight"] = to_mlx_conv1d(t(f"enc_p.mrte.{nm}.weight"))
        arrays[f"enc_p.mrte.{nm}.bias"] = t(f"enc_p.mrte.{nm}.bias")
        arrays[f"enc_p.mrte.{nm}.bias"] = t(f"enc_p.mrte.{nm}.bias")
    arrays["enc_p.mrte.cross_attn.conv_q.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_q.weight"))
    arrays["enc_p.mrte.cross_attn.conv_q.bias"] = t("enc_p.mrte.cross_attention.conv_q.bias")
    arrays["enc_p.mrte.cross_attn.conv_k.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_k.weight"))
    arrays["enc_p.mrte.cross_attn.conv_k.bias"] = t("enc_p.mrte.cross_attention.conv_k.bias")
    arrays["enc_p.mrte.cross_attn.conv_v.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_v.weight"))
    arrays["enc_p.mrte.cross_attn.conv_v.bias"] = t("enc_p.mrte.cross_attention.conv_v.bias")
    arrays["enc_p.mrte.cross_attn.conv_o.weight"] = to_mlx_conv1d(t("enc_p.mrte.cross_attention.conv_o.weight"))
    arrays["enc_p.mrte.cross_attn.conv_o.bias"] = t("enc_p.mrte.cross_attention.conv_o.bias")
    arrays["ref_enc.spectral.0.weight"] = t("ref_enc.spectral.0.fc.weight")
    arrays["ref_enc.spectral.0.bias"] = t("ref_enc.spectral.0.fc.bias")
    arrays["ref_enc.spectral.3.weight"] = t("ref_enc.spectral.3.fc.weight")
    arrays["ref_enc.spectral.3.bias"] = t("ref_enc.spectral.3.fc.bias")
    for i in (0, 1):
        arrays[f"ref_enc.temporal.{i}.w1"] = np.transpose(t(f"ref_enc.temporal.{i}.conv1.conv.weight"), (0, 2, 1))
        arrays[f"ref_enc.temporal.{i}.b1"] = t(f"ref_enc.temporal.{i}.conv1.conv.bias")
    arrays["ref_enc.slf_attn.w_qs"] = t("ref_enc.slf_attn.w_qs.weight")
    arrays["ref_enc.slf_attn.w_qs.b"] = t("ref_enc.slf_attn.w_qs.bias")
    arrays["ref_enc.slf_attn.w_ks"] = t("ref_enc.slf_attn.w_ks.weight")
    arrays["ref_enc.slf_attn.w_ks.b"] = t("ref_enc.slf_attn.w_ks.bias")
    arrays["ref_enc.slf_attn.w_vs"] = t("ref_enc.slf_attn.w_vs.weight")
    arrays["ref_enc.slf_attn.w_vs.b"] = t("ref_enc.slf_attn.w_vs.bias")
    arrays["ref_enc.slf_attn.fc"] = t("ref_enc.slf_attn.fc.weight")
    arrays["ref_enc.slf_attn.fc.b"] = t("ref_enc.slf_attn.fc.bias")
    arrays["ref_enc.fc"] = t("ref_enc.fc.fc.weight")
    arrays["ref_enc.fc.b"] = t("ref_enc.fc.fc.bias")

    if hps["model"].get("semantic_frame_rate") == "25hz":
        arrays["ssl_proj.weight"] = to_mlx_conv1d(t("ssl_proj.weight"))
        arrays["ssl_proj.bias"] = t("ssl_proj.bias")

    arrays["quantizer.codebook"] = t("quantizer.vq.layers.0._codebook.embed")

    # bridge / wns1 / linear_mel
    arrays["bridge.weight"] = to_mlx_conv1d(t("bridge.0.weight"))
    arrays["bridge.bias"] = t("bridge.0.bias")
    arrays["wns1.pre.weight"] = to_mlx_conv1d(t("wns1.pre.weight"))
    arrays["wns1.pre.bias"] = t("wns1.pre.bias")
    arrays["wns1.proj.weight"] = to_mlx_conv1d(t("wns1.proj.weight"))
    arrays["wns1.proj.bias"] = t("wns1.proj.bias")
    arrays["wns1.cond"] = np.ascontiguousarray(np.transpose(wn("wns1.enc.cond_layer"), (0, 2, 1)))
    arrays["wns1.cond.b"] = t("wns1.enc.cond_layer.bias")
    n_wn = max_idx("wns1.enc.in_layers")
    for wi in range(n_wn):
        arrays[f"wns1.in.{wi}"] = np.ascontiguousarray(np.transpose(wn(f"wns1.enc.in_layers.{wi}"), (0, 2, 1)))
        arrays[f"wns1.in.{wi}.b"] = t(f"wns1.enc.in_layers.{wi}.bias")
        arrays[f"wns1.skip.{wi}"] = np.ascontiguousarray(np.transpose(wn(f"wns1.enc.res_skip_layers.{wi}"), (0, 2, 1)))
        arrays[f"wns1.skip.{wi}.b"] = t(f"wns1.enc.res_skip_layers.{wi}.bias")
    # wns1 LayerNorm is affine (pre-fused into runtime default init when absent)
    arrays["linear_mel.weight"] = to_mlx_conv1d(t("linear_mel.weight"))
    arrays["linear_mel.bias"] = t("linear_mel.bias")

    # DiT estimator
    arrays["dit.time_embed.0.w"] = t("cfm.estimator.time_embed.time_mlp.0.weight")
    arrays["dit.time_embed.0.b"] = t("cfm.estimator.time_embed.time_mlp.0.bias")
    arrays["dit.time_embed.2.w"] = t("cfm.estimator.time_embed.time_mlp.2.weight")
    arrays["dit.time_embed.2.b"] = t("cfm.estimator.time_embed.time_mlp.2.bias")
    is_v5 = version in ("v5", "v5dev", "v5turbo")
    if not is_v5:
        arrays["dit.d_embed.0.w"] = t("cfm.estimator.d_embed.time_mlp.0.weight")
        arrays["dit.d_embed.0.b"] = t("cfm.estimator.d_embed.time_mlp.0.bias")
        arrays["dit.d_embed.2.w"] = t("cfm.estimator.d_embed.time_mlp.2.weight")
        arrays[("dit.d_embed.2.b")] = t("cfm.estimator.d_embed.time_mlp.2.bias")

    # text_embed convnext blocks (dwconv has bias in torch; folded into dw_b)
    n_tb = max_idx("cfm.estimator.text_embed.text_blocks")
    for i in range(n_tb):
        base = f"cfm.estimator.text_embed.text_blocks.{i}."
        dst = f"dit.text.{i}."
        dw = sd[base + "dwconv.weight"].float().numpy()  # (C, 1, k)
        arrays[dst + "dw"] = np.transpose(dw, (0, 2, 1))  # (C, k, 1)
        arrays[dst + "dw_b"] = t(base + "dwconv.bias")
        arrays[dst + "norm_w"] = t(base + "norm.weight")
        arrays[dst + "norm_b"] = t(base + "norm.bias")
        arrays[dst + "pw1.w"] = t(base + "pwconv1.weight")
        arrays[dst + "pw1.b"] = t(base + "pwconv1.bias")
        arrays[dst + "pw2.w"] = t(base + "pwconv2.weight")
        arrays[dst + "pw2.b"] = t(base + "pwconv2.bias")
        g = sd[base + "grn.gamma"].float().numpy().reshape(-1)
        b = sd[base + "grn.beta"].float().numpy().reshape(-1)
        arrays[dst + "grn.gamma"] = g
        arrays[dst + "grn.beta"] = b

    ie = "cfm.estimator.input_embed."
    arrays["dit.in.proj.w"] = t(ie + "proj.weight")
    arrays["dit.in.proj.b"] = t(ie + "proj.bias")
    cp0 = sd[ie + "conv_pos_embed.conv1d.0.weight"].float().numpy()  # (C, C/g, k)
    cp2 = sd[ie + "conv_pos_embed.conv1d.2.weight"].float().numpy()
    arrays["dit.in.conv_pos_0.w"] = np.transpose(cp0, (0, 2, 1))  # (C, k, C/g)
    arrays["dit.in.conv_pos_0.b"] = t(ie + "conv_pos_embed.conv1d.0.bias")
    arrays["dit.in.conv_pos_1.w"] = np.transpose(cp2, (0, 2, 1))
    arrays["dit.in.conv_pos_1.b"] = t(ie + "conv_pos_embed.conv1d.2.bias")

    depth = max_idx("cfm.estimator.transformer_blocks")
    for i in range(depth):
        base = f"cfm.estimator.transformer_blocks.{i}."
        dst = f"dit.blocks.{i}."
        arrays[dst + "attn_norm.w"] = t(base + "attn_norm.linear.weight")
        arrays[dst + "attn_norm.b"] = t(base + "attn_norm.linear.bias")
        for nm in ("q", "k", "v"):
            arrays[dst + f"attn.to_{nm}.w"] = t(base + f"attn.to_{nm}.weight")
            arrays[dst + f"attn.to_{nm}.b"] = t(base + f"attn.to_{nm}.bias")
        arrays[dst + "attn.to_out.w"] = t(base + "attn.to_out.0.weight")
        arrays[dst + "attn.to_out.b"] = t(base + "attn.to_out.0.bias")
        arrays[dst + "ff.0.w"] = t(base + "ff.ff.0.0.weight")
        arrays[dst + "ff.0.b"] = t(base + "ff.ff.0.0.bias")
        arrays[dst + "ff.2.w"] = t(base + "ff.ff.2.weight")
        arrays[dst + "ff.2.b"] = t(base + "ff.ff.2.bias")
        # ff_norm is elementwise_affine=False LayerNorm — no params

    if hps["model"].get("long_skip_connection"):
        arrays["dit.long_skip"] = t("cfm.estimator.long_skip_projection.weight")
    arrays["dit.norm_out.w"] = t("cfm.estimator.norm_out.linear.weight")
    arrays["dit.norm_out.b"] = t("cfm.estimator.norm_out.linear.bias")
    arrays["dit.proj_out.w"] = t("cfm.estimator.proj_out.weight")
    arrays["dit.proj_out.b"] = t("cfm.estimator.proj_out.bias")

    if fp16:
        arrays = {k: v.astype(np.float16) for k, v in arrays.items()}
    save_safetensors(os.path.join(out_dir, "sovits.safetensors"), arrays)
    meta = {"version": version,
            "model_hps": {k: hps["model"].get(k) for k in (
                "inter_channels", "hidden_channels", "gin_channels", "semantic_frame_rate",
                "long_skip_connection")},
            "data": {"sampling_rate": hps["data"]["sampling_rate"]},
            "dit": {"depth": depth, "text_blocks": n_tb}}
    with open(os.path.join(out_dir, "sovits.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return len(arrays)


# ---------------------------------------------------------------------------
# v4/v5 HiFi-GAN vocoder (gsv-v4-pretrained/vocoder.pth)
# ---------------------------------------------------------------------------

def convert_generator_vocoder(ckpt_path: str, out_dir: str):
    """v4/v5 vocoder.pth (Generator, 48 kHz) -> vocoder.safetensors + vocoder.json.

    The checkpoint stores PLAIN conv weights (official calls remove_weight_norm
    before state_dict capture), so no weight-norm fusion is needed.
    """
    import torch

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert not any("weight_g" in k for k in sd), "unexpected weight_norm keys"
    arrays: dict[str, np.ndarray] = {}
    arrays["conv_pre.weight"] = to_mlx_conv1d(sd["conv_pre.weight"].float().numpy())
    arrays["conv_pre.bias"] = sd["conv_pre.bias"].float().numpy()
    n_ups = len([k for k in sd if re.fullmatch(r"ups\.\d+\.weight", k)])
    for i in range(n_ups):
        arrays[f"ups.{i}.weight"] = to_mlx_conv1d_t(sd[f"ups.{i}.weight"].float().numpy())
        arrays[f"ups.{i}.bias"] = sd[f"ups.{i}.bias"].float().numpy()
    n_rb = len([k for k in sd if re.fullmatch(r"resblocks\.\d+\.convs1\.0\.weight", k)])
    for b in range(n_rb):
        for cn in ("convs1", "convs2"):
            for j in range(3):
                arrays[f"resblocks.{b}.{cn}.{j}.weight"] = to_mlx_conv1d(sd[f"resblocks.{b}.{cn}.{j}.weight"].float().numpy())
                arrays[f"resblocks.{b}.{cn}.{j}.bias"] = sd[f"resblocks.{b}.{cn}.{j}.bias"].float().numpy()
    arrays["conv_post.weight"] = to_mlx_conv1d(sd["conv_post.weight"].float().numpy())
    arrays["conv_post.bias"] = sd["conv_post.bias"].float().numpy()
    save_safetensors(os.path.join(out_dir, "vocoder.safetensors"), arrays)
    meta = {"initial_channel": 100, "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 6, 2, 2, 2], "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [20, 12, 4, 4, 4], "gin_channels": 0,
            "is_bias": True, "sampling_rate": 48000}
    with open(os.path.join(out_dir, "vocoder.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return len(arrays)


def _fuse_wn_hf(g, v):
    """HF weight_norm(g, v, dim=last): norm over all dims except the last."""
    g = g.astype(np.float32)
    v = v.astype(np.float32)
    axes = tuple(range(v.ndim - 1))
    norm = np.sqrt(np.sum(v**2, axis=axes, keepdims=True))
    return g * v / norm


# ---------------------------------------------------------------------------
# BigVGAN v2 vocoder (v3; nvidia--bigvgan_v2_24khz_100band_256x)
# ---------------------------------------------------------------------------

def convert_bigvgan(gen_pt_path: str, config_path: str, out_dir: str, fp16: bool = False):
    """bigvgan_generator.pt (weights_only-safe dict['generator']) -> bigvgan.safetensors.

    Weight-norm convs are pre-fused (equivalent to official remove_weight_norm():
    fused weight = weight_g * weight_v / ||weight_v||_per_out_channel, which stays
    EXACTLY the module's effective weight). kaiser resample filters are buffers,
    recomputed at runtime (verified identical to the checkpoint buffers).

    NOTE: fp16 is available but defaults OFF -- fp16 rounding of the fused convs
    amplifies through the 256x stack (A/B max-abs 1.1e-2 on random mel vs 5e-5
    fp32 on the same input). v3 keeps the vocoder in fp32 like official CPU.
    """
    import torch

    sd = torch.load(gen_pt_path, map_location="cpu", weights_only=False)["generator"]
    with open(config_path) as f:
        h = json.load(f)

    def fuse(k):
        g = sd[k + ".weight_g"].float().numpy()
        v = sd[k + ".weight_v"].float().numpy()
        norm = np.linalg.norm(v.reshape(v.shape[0], -1), axis=1, keepdims=True)
        return (g.reshape(-1, *([1] * (v.ndim - 1))) * v / norm.reshape(-1, *([1] * (v.ndim - 1))))

    arrays: dict[str, np.ndarray] = {}
    arrays["conv_pre.weight"] = to_mlx_conv1d(fuse("conv_pre"))
    arrays["conv_pre.bias"] = sd["conv_pre.bias"].float().numpy()

    n_ups = len(h["upsample_rates"])
    for i in range(n_ups):
        arrays[f"ups.{i}.weight"] = to_mlx_conv1d_t(fuse(f"ups.{i}.0"))
        arrays[f"ups.{i}.bias"] = sd[f"ups.{i}.0.bias"].float().numpy()

    n_k = len(h["resblock_kernel_sizes"])
    n_rb = n_ups * n_k
    for b in range(n_rb):
        for cn in ("convs1", "convs2"):
            for j in range(3):
                arrays[f"resblocks.{b}.{cn}.{j}.weight"] = to_mlx_conv1d(fuse(f"resblocks.{b}.{cn}.{j}"))
                arrays[f"resblocks.{b}.{cn}.{j}.bias"] = sd[f"resblocks.{b}.{cn}.{j}.bias"].float().numpy()
        for a in range(6):
            arrays[f"resblocks.{b}.act.{a}.alpha"] = sd[f"resblocks.{b}.activations.{a}.act.alpha"].float().numpy().reshape(-1)
            arrays[f"resblocks.{b}.act.{a}.beta"] = sd[f"resblocks.{b}.activations.{a}.act.beta"].float().numpy().reshape(-1)

    arrays["post.alpha"] = sd["activation_post.act.alpha"].float().numpy().reshape(-1)
    arrays["post.beta"] = sd["activation_post.act.beta"].float().numpy().reshape(-1)
    arrays["conv_post.weight"] = to_mlx_conv1d(fuse("conv_post"))
    if h.get("use_bias_at_final", True) and "conv_post.bias" in sd:
        arrays["conv_post.bias"] = sd["conv_post.bias"].float().numpy()

    if fp16:
        arrays = {k: v.astype(np.float16) for k, v in arrays.items()}
    save_safetensors(os.path.join(out_dir, "bigvgan.safetensors"), arrays)
    meta = {k: h.get(k) for k in (
        "upsample_rates", "upsample_kernel_sizes", "upsample_initial_channel",
        "resblock_kernel_sizes", "resblock_dilation_sizes", "num_mels",
        "activation", "snake_logscale", "use_bias_at_final", "use_tanh_at_final",
        "sampling_rate")}
    with open(os.path.join(out_dir, "bigvgan.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return len(arrays)
