"""Export trained s2 weights (training flat names) -> pipeline inference dir.

Produces <out>/sovits.safetensors + sovits.json loadable by
gsovits_mlx.pipeline._load_sovits_v1v2 (same layout as
tools/convert_sovits.py convert_sovits_v1v2):

  - weight-norm (g,v) pairs fused: w = g * v / ||v||_per-out-channel
  - conv layouts stay MLX (training npz already transposed), EXCEPT:
      flow WN in/skip/cond  fused -> (0,2,1) transposed to torch layout
      (the pipeline loader applies g(f"flow.{fi}.enc.in.{wi}")) directly
      into Conv1d.weight, and its WN stores torch layout (out,in,k)?? no —
      see notes: pipeline loads flow in.{i} into wn.in_layers[i].weight and
      the WN Conv1d uses MLX layout (out,k,in); convert_sovits transposes
      the FUSED wn weight (0,2,1) from torch (out,in,k) to MLX (out,k,in).
  - enc_q dropped entirely (inference doesn't use it)
  - ssl_proj + quantizer.codebook copied from the loaded (frozen) values
  - fp16 cast, matching models_local exports (sv family stays F32 for v2Pro)
"""

from __future__ import annotations

import json
import os

import mlx.core as mx
import numpy as np


def _fuse(g: mx.array, v: mx.array) -> mx.array:
    v32 = v.astype(mx.float32)
    norm = mx.sqrt(mx.sum(v32 * v32, axis=tuple(range(1, v32.ndim)), keepdims=True))
    return (g.reshape(-1)[: , None, None][:, : v32.ndim - 1 or 1].reshape(
        [v32.shape[0]] + [1] * (v32.ndim - 1)) * v32 / norm)


def fuse_wn(g: mx.array, v: mx.array) -> mx.array:
    """torch weight_norm fuse: w[out] = g[out] * v[out] / ||v[out]||."""
    v32 = v.astype(mx.float32)
    norm = mx.sqrt(mx.sum(v32.reshape(v32.shape[0], -1) ** 2, axis=1,
                          keepdims=True))
    return (g.reshape(-1)[:, None, None][:, :1, :1].reshape(
        [v32.shape[0]] + [1] * (v32.ndim - 1)) if v32.ndim == 3 else
        g.reshape(-1)[:, None][:, :1] if v32.ndim == 2 else g.reshape(-1)) \
        * v32 / norm.reshape([v32.shape[0]] + [1] * (v32.ndim - 1))


def export_inference(net_g, masters: dict, params32: dict, hps: dict,
                     version: str, out_dir: str) -> str:
    """masters: {training_name: fp32 mx.array} (from optimizer groups)."""
    p = {**masters}
    a: dict[str, np.ndarray] = {}

    def np16(x):
        return np.asarray(x.astype(mx.float16))

    def np32(x):
        return np.asarray(x.astype(mx.float32))

    def fused_np(base: str) -> np.ndarray:
        w = fuse_wn(p[base + ".weight_g"], p[base + ".weight_v"])
        return np32(w)

    # enc_p
    a["enc_p.ssl_proj.weight"] = np32(p["enc_p.ssl_proj.weight"])
    a["enc_p.ssl_proj.bias"] = np32(p["enc_p.ssl_proj.bias"])
    a["enc_p.text_embedding"] = np32(p["enc_p.text_embedding"])
    a["enc_p.proj.weight"] = np32(p["enc_p.proj.weight"])
    a["enc_p.proj.bias"] = np32(p["enc_p.proj.bias"])
    for src_pre, dst_pre, n in (
            ("enc_p.encoder_ssl", "enc_p.enc_ssl",
             len(net_g.enc_p.encoder_ssl.attn_layers)),
            ("enc_p.encoder_text", "enc_p.enc_text",
             len(net_g.enc_p.encoder_text.attn_layers)),
            ("enc_p.encoder2", "enc_p.enc2", len(net_g.enc_p.encoder2.attn_layers))):
        for i in range(n):
            d = f"{dst_pre}.{i}."
            s = f"{src_pre}.attn_layers.{i}."
            for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
                a[d + f"attn.{nm}.weight"] = np32(p[s + nm + ".weight"])
                a[d + f"attn.{nm}.bias"] = np32(p[s + nm + ".bias"])
            if s + "emb_rel_k" in p:
                a[d + "attn.emb_rel_k"] = np32(p[s + "emb_rel_k"])
                a[d + "attn.emb_rel_v"] = np32(p[s + "emb_rel_v"])
            a[d + "norm1"] = np32(p[f"{src_pre}.norm_layers_1.{i}.gamma"])
            a[d + "norm1.b"] = np32(p[f"{src_pre}.norm_layers_1.{i}.beta"])
            a[d + "norm2"] = np32(p[f"{src_pre}.norm_layers_2.{i}.gamma"])
            a[d + "norm2.b"] = np32(p[f"{src_pre}.norm_layers_2.{i}.beta"])
            a[d + "ffn.conv1.weight"] = np32(
                p[f"{src_pre}.ffn_layers.{i}.conv_1.weight"])
            a[d + "ffn.conv1.bias"] = np32(
                p[f"{src_pre}.ffn_layers.{i}.conv_1.bias"])
            a[d + "ffn.conv2.weight"] = np32(
                p[f"{src_pre}.ffn_layers.{i}.conv_2.weight"])
            a[d + "ffn.conv2.bias"] = np32(
                p[f"{src_pre}.ffn_layers.{i}.conv_2.bias"])
    for nm in ("c_pre", "text_pre", "c_post"):
        a[f"enc_p.mrte.{nm}.weight"] = np32(p[f"enc_p.mrte.{nm}.weight"])
        a[f"enc_p.mrte.{nm}.bias"] = np32(p[f"enc_p.mrte.{nm}.bias"])
    for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
        a[f"enc_p.mrte.cross_attn.{nm}.weight"] = np32(
            p[f"enc_p.mrte.cross_attention.{nm}.weight"])
        a[f"enc_p.mrte.cross_attn.{nm}.bias"] = np32(
            p[f"enc_p.mrte.cross_attention.{nm}.bias"])
    if "enc_p.mrte.cross_attention.emb_rel_k" in p:
        a["enc_p.mrte.cross_attn.emb_rel_k"] = np32(
            p["enc_p.mrte.cross_attention.emb_rel_k"])
        a["enc_p.mrte.cross_attn.emb_rel_v"] = np32(
            p["enc_p.mrte.cross_attention.emb_rel_v"])

    # ref_enc
    a["ref_enc.spectral.0.weight"] = np32(p["ref_enc.spectral.0.fc.weight"])
    a["ref_enc.spectral.0.bias"] = np32(p["ref_enc.spectral.0.fc.bias"])
    a["ref_enc.spectral.3.weight"] = np32(p["ref_enc.spectral.3.fc.weight"])
    a["ref_enc.spectral.3.bias"] = np32(p["ref_enc.spectral.3.fc.bias"])
    for i in (0, 1):
        # training layout is MLX (2C,k,C); loader's w_1 expects the SAME
        # MLX layout (pipeline loads temporal w1 directly into t.w_1)
        a[f"ref_enc.temporal.{i}.w1"] = np32(
            p[f"ref_enc.temporal.{i}.conv1.conv.weight"])
        a[f"ref_enc.temporal.{i}.b1"] = np32(
            p[f"ref_enc.temporal.{i}.conv1.conv.bias"])
    a["ref_enc.slf_attn.w_qs"] = np32(p["ref_enc.slf_attn.w_qs.weight"])
    a["ref_enc.slf_attn.w_qs.b"] = np32(p["ref_enc.slf_attn.w_qs.bias"])
    a["ref_enc.slf_attn.w_ks"] = np32(p["ref_enc.slf_attn.w_ks.weight"])
    a["ref_enc.slf_attn.w_ks.b"] = np32(p["ref_enc.slf_attn.w_ks.bias"])
    a["ref_enc.slf_attn.w_vs"] = np32(p["ref_enc.slf_attn.w_vs.weight"])
    a["ref_enc.slf_attn.w_vs.b"] = np32(p["ref_enc.slf_attn.w_vs.bias"])
    a["ref_enc.slf_attn.fc"] = np32(p["ref_enc.slf_attn.fc.weight"])
    a["ref_enc.slf_attn.fc.b"] = np32(p["ref_enc.slf_attn.fc.bias"])
    a["ref_enc.fc"] = np32(p["ref_enc.fc.fc.weight"])
    a["ref_enc.fc.b"] = np32(p["ref_enc.fc.fc.bias"])

    # dec
    a["dec.conv_pre.weight"] = np32(p["dec.conv_pre.weight"])
    a["dec.conv_pre.bias"] = np32(p["dec.conv_pre.bias"])
    for i in range(len(net_g.dec.ups)):
        # fused weight is (out,k,in) — same as convert's to_mlx_conv1d_t
        a[f"dec.ups.{i}.weight"] = fused_np(f"dec.ups.{i}")
        a[f"dec.ups.{i}.bias"] = np32(p[f"dec.ups.{i}.bias"])
    for i in range(len(net_g.dec.resblocks)):
        for cn in ("convs1", "convs2"):
            for j in range(3):
                base = f"dec.resblocks.{i}.{cn}.{j}"
                a[base + ".weight"] = fused_np(base)
                a[base + ".bias"] = np32(p[base + ".bias"])
    a["dec.cond.weight"] = np32(p["dec.cond.weight"])
    a["dec.cond.bias"] = np32(p["dec.cond.bias"])
    a["dec.conv_post.weight"] = np32(p["dec.conv_post.weight"])
    a["dec.conv_post.bias"] = np.zeros((1,), np.float32)  # is_bias=False

    # flow (WN fused, layout back to torch (out,in,k) for the loader's WN)
    wn_flows = [f for f in net_g.flow.flows if hasattr(f, "enc")]
    for fi in range(len(wn_flows)):
        src = f"flow.flows.{fi*2}"
        for w in ("pre", "post"):
            a[f"flow.{fi}.{w}.weight"] = np32(p[f"{src}.{w}.weight"])
            a[f"flow.{fi}.{w}.bias"] = np32(p[f"{src}.{w}.bias"])
        # cond: training v is MLX (out,k,in) with k=1; loader wants
        # cond_layer.weight as (out,1,in)?? pipeline: enc.cond_layer.weight =
        # g("flow.{fi}.enc.cond") and convert transposed (0,2,1) from torch
        # (out,in,1) -> (out,1,in). Our fused MLX (out,1,in) IS the target.
        a[f"flow.{fi}.enc.cond"] = fused_np(f"{src}.enc.cond_layer")
        a[f"flow.{fi}.enc.cond.b"] = np32(p[f"{src}.enc.cond_layer.bias"])
        n_wn = len(wn_flows[fi].enc.in_layers)
        for wi in range(n_wn):
            for layer, dst_l in (("in_layers", "in"), ("res_skip_layers", "skip")):
                fused = fuse_wn(p[f"{src}.enc.{layer}.{wi}.weight_g"],
                                p[f"{src}.enc.{layer}.{wi}.weight_v"])
                # training MLX (out,k,in) -> loader wants torch (out,in,k):
                # convert transposed torch fused by (0,2,1). Undo it.
                a[f"flow.{fi}.enc.{dst_l}.{wi}"] = np32(
                    mx.transpose(fused, (0, 2, 1)))
                a[f"flow.{fi}.enc.{dst_l}.{wi}.b"] = np32(
                    p[f"{src}.enc.{layer}.{wi}.bias"])

    # quantizer + ssl_proj (frozen; from the loaded checkpoint)
    a["quantizer.codebook"] = np32(params32["quantizer.vq.layers.0._codebook.embed"])
    if hps.get("semantic_frame_rate") == "25hz":
        a["ssl_proj.weight"] = np32(params32["ssl_proj.weight"])
        a["ssl_proj.bias"] = np32(params32["ssl_proj.bias"])

    # v2Pro sv family (F32 in models_local exports)
    if version in ("v2Pro", "v2ProPlus"):
        for nm in ("sv_emb", "ge_to512"):
            a[f"{nm}.weight"] = np32(p[f"{nm}.weight"])
            a[f"{nm}.bias"] = np32(p[f"{nm}.bias"])
        a["prelu.weight"] = np32(p["prelu.weight"])

    # fp16 cast except sv family
    out = {}
    for k, v in a.items():
        out[k] = v.astype(np.float16) if not k.startswith(
            ("sv_emb.", "ge_to512.", "prelu.")) else v.astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    # save via the same writer as convert_sovits
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from tools.convert_sovits import save_safetensors
    save_safetensors(os.path.join(out_dir, "sovits.safetensors"), out)

    meta = {"version": version,
            "model_hps": {k: hps[k] for k in (
                "inter_channels", "hidden_channels", "filter_channels",
                "n_heads", "n_layers", "kernel_size",
                "resblock", "resblock_kernel_sizes", "resblock_dilation_sizes",
                "upsample_rates", "upsample_initial_channel",
                "upsample_kernel_sizes", "gin_channels", "semantic_frame_rate")},
            "data": {"sampling_rate": 32000}}
    meta["model_hps"]["p_dropout"] = 0.1
    with open(os.path.join(out_dir, "sovits.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return out_dir
