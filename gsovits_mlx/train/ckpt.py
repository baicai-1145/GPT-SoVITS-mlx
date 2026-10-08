"""Checkpoint save/load for MLX training + inference-weight exporters.

* save_resume / load_resume -- .safetensors training checkpoints carrying
  fp32 params + optimizer states + step/epoch (one file per param group
  shard plus a JSON manifest; optimizer states go in as fp32 arrays).
* save_s1_inference -- writes gpt.safetensors (fp16) + gpt.json exactly
  loadable by ``gsovits_mlx.pipeline.load_gpt``: meta schema
  ``{"config": <full training config dict>, "n_layer": N}`` where config
  carries ``model.n_layer`` etc. (see models_local/s1/gpt.json).
* save_s2_inference -- writes sovits.safetensors + sovits.json loadable by
  ``pipeline._load_sovits_v1v2`` / ``load_sovits_v3``. The key names mirror
  tools/convert_sovits.py outputs; exporters take the TRAINING state dict
  (flat torch-style keys) and perform the same transposes/fusions concept —
  but since MLX trainers already keep MLX-layout tensors, the expected input
  here is a {name: mx.array} dict ALREADY in MLX layout (training-side
  modules build them that way); the exporter only handles dtype policy:
  fp16 everywhere except the v2Pro sv-projection family (sv_emb / ge_to512 /
  prelu) which stays fp32 to match models_local/v2pro F32.

Version strings follow the inference engine: "v1","v2","v2Pro","v2ProPlus"
(v1v2 loader) and "v3","v4","v5dev","v5turbo" (v3 loader).
"""

from __future__ import annotations

import json
import os

import mlx.core as mx

__all__ = ["save_resume", "load_resume", "save_s1_inference",
           "save_s2_inference"]

# v2Pro-family weights that stay fp32 in the official exports (ERes2NetV2
# sv projection is fp32-sensitive; models_local/v2pro stores these F32).
_SV_FP32_KEYS_PREFIXES = ("sv_emb.", "ge_to512.", "prelu")


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def save_resume(path: str, model, optimizers, step: int, epoch: int,
                extra: dict | None = None) -> str:
    """Persist fp32 params + optimizer states + progress.

    ``model``: object with ``.parameters() -> {name: mx.array}`` or a plain
    dict. Optimizers expose ``.state_dict()`` (AdamW/ScaledAdam here).
    Layout: ``path`` (a directory) with manifest.json, params.safetensors,
    opt<i>.safetensors.
    """
    params = model.parameters() if hasattr(model, "parameters") else model
    os.makedirs(path, exist_ok=True)
    mx.save_safetensors(os.path.join(path, "params.safetensors"),
                        {k: v.astype(mx.float32) for k, v in params.items()})
    manifest = {"step": step, "epoch": epoch, "extra": extra or {},
                "optimizers": []}
    for i, o in enumerate(optimizers if isinstance(optimizers, (list, tuple))
                          else [optimizers]):
        sd = o.state_dict()
        flat = {}
        for gi, g in enumerate(sd.get("groups", [])):
            for name, st in g.get("states", {}).items():
                for sk in ("m", "v", "delta", "exp_avg_sq", "param_rms",
                           "scale_exp_avg_sq", "scale_grads"):
                    if sk in st:
                        flat[f"opt{i}.group{gi}.{name}.{sk}"] = st[sk].astype(mx.float32)
                flat[f"opt{i}.group{gi}.{name}.step"] = mx.array(st["step"], mx.uint64)
        manifest["optimizers"].append({
            "step": sd.get("step", step),
            "groups": [{k: g[k] for k in g if k != "states"}
                       for g in sd.get("groups", [])],
        })
        if flat:
            mx.save_safetensors(os.path.join(path, f"opt{i}.safetensors"), flat)
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(_json_safe(manifest), f, indent=1)
    return path


def load_resume(path: str, model, optimizers) -> dict:
    """Inverse of save_resume: restores params, optimizer states, step/epoch.

    Returns the manifest ({"step","epoch","extra"}). ``model`` is updated in
    place if it exposes ``.update(dict)``; otherwise the caller reads
    ``manifest`` and the loaded params themselves via the returned dict key
    "params" (also written into model when possible).
    """
    with open(os.path.join(path, "manifest.json")) as f:
        manifest = json.load(f)
    params = mx.load(os.path.join(path, "params.safetensors"))
    if hasattr(model, "update"):
        model.update(params)
    for i, o in enumerate(optimizers if isinstance(optimizers, (list, tuple))
                          else [optimizers]):
        p = os.path.join(path, f"opt{i}.safetensors")
        if not os.path.isfile(p):
            continue
        flat = mx.load(p)
        sd_groups = manifest["optimizers"][i]["groups"]
        states = [dict() for _ in o.param_groups]
        for key, arr in flat.items():
            # opt{i}.group{gi}.{name}.{state_key}
            rest = key.split(".", 2)
            gi = int(rest[1].removeprefix("group"))
            name, sk = rest[2].rsplit(".", 1)
            if sk == "step":
                states[gi].setdefault(name, {})["step"] = int(arr)
            else:
                states[gi].setdefault(name, {})[sk] = arr
        sd = {"step": manifest["optimizers"][i].get("step", 0),
              "groups": [dict(g, states=states[gi])
                         for gi, g in enumerate(sd_groups)]}
        o.load_state_dict(sd)
    return {"step": manifest["step"], "epoch": manifest["epoch"],
            "extra": manifest.get("extra", {}), "params": params}


# ---------------------------------------------------------------------------
# s1 (GPT AR) inference export
# ---------------------------------------------------------------------------

def save_s1_inference(weights_dir: str, model, config: dict, epoch: int) -> str:
    """Write gpt.safetensors (fp16) + gpt.json loadable by pipeline.load_gpt.

    ``model``: the training Text2Semantic-style module exposing
    ``.parameters()`` with the SAME key names the inference engine expects
    (ar_text_embedding, block.<i>.qkv_w, ... — the trainer builds MLX-layout
    tensors directly, matching convert_s1 output names).
    ``config``: the full training config dict (stored under "config"; the
    loader reads config["model"] etc.).
    """
    os.makedirs(weights_dir, exist_ok=True)
    params = model.parameters() if hasattr(model, "parameters") else model
    arrays = {k: v.astype(mx.float16) for k, v in params.items()}
    mx.save_safetensors(os.path.join(weights_dir, "gpt.safetensors"), arrays)
    n_layer = arrays and max(
        (int(k.split(".")[1]) for k in arrays if k.startswith("block.")),
        default=0) + 1
    meta = {"config": _json_safe(config), "n_layer": n_layer,
            "epoch": epoch}
    with open(os.path.join(weights_dir, "gpt.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    return weights_dir


# ---------------------------------------------------------------------------
# s2 (SoVITS) inference export
# ---------------------------------------------------------------------------

_S2_MODEL_HPS_KEYS_V1V2 = (
    "inter_channels", "hidden_channels", "filter_channels", "n_heads",
    "n_layers", "kernel_size", "p_dropout", "resblock",
    "resblock_kernel_sizes", "resblock_dilation_sizes", "upsample_rates",
    "upsample_initial_channel", "upsample_kernel_sizes", "gin_channels",
    "semantic_frame_rate")

_S2_MODEL_HPS_KEYS_V3 = (
    "inter_channels", "hidden_channels", "gin_channels", "semantic_frame_rate",
    "long_skip_connection")


def s2v3_model_to_arrays(model, merged_dit: dict | None = None) -> dict:
    """SynthesizerTrnV3 MODULE -> convert_sovits_v3v5 name/layout dict.

    Inverse of gsovits_mlx.pipeline.load_sovits_v3 + tools/
    convert_sovits.py::convert_sovits_v3v5: emits the exact safetensors
    names the inference loader consumes. Module weights keep the converted
    (safetensors) layout — load_sovits_v3 assigns arrays into module attrs
    unchanged — so this is a pure renaming pass. ``merged_dit`` optionally
    overrides DiT weights with MERGED LoRA values keyed by convert-style
    names (``dit.blocks.<i>.attn.to_q.w`` etc., from
    lora.merged_training_weights). The frozen quantizer codebook is
    embedded so the export is self-contained (bitwise the base value).
    """
    out: dict = {}

    def conv(weight):
        # pipeline.load_sovits_v3 assigns converted arrays UNCHANGED into the
        # module attrs (the MLX layers consume torch (out, in, k) via internal
        # transposes), so the module weight layout EQUALS the safetensors
        # layout: pass through untouched.
        return weight

    # quantizer codebook (frozen; embed for a self-contained export)
    out["quantizer.codebook"] = model.quantizer.embed

    ep = model.enc_p
    out["enc_p.ssl_proj.weight"] = conv(ep.ssl_proj.weight)
    out["enc_p.ssl_proj.bias"] = ep.ssl_proj.bias
    out["enc_p.text_embedding"] = ep.text_embedding
    out["enc_p.proj.weight"] = conv(ep.proj.weight)
    out["enc_p.proj.bias"] = ep.proj.bias
    for enc, prefix in ((ep.encoder_ssl, "enc_ssl"),
                        (ep.encoder_text, "enc_text"),
                        (ep.encoder2, "enc2")):
        for i, at in enumerate(enc.attn_layers):
            d = f"enc_p.{prefix}.{i}."
            for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
                out[f"{d}attn.{nm}.weight"] = conv(getattr(at, nm).weight)
                out[f"{d}attn.{nm}.bias"] = getattr(at, nm).bias
            if getattr(at, "emb_rel_k", None) is not None:
                out[f"{d}attn.emb_rel_k"] = at.emb_rel_k
                out[f"{d}attn.emb_rel_v"] = at.emb_rel_v
            n1, n2 = enc.norm_layers_1[i], enc.norm_layers_2[i]
            out[f"{d}norm1"] = n1.gamma
            out[f"{d}norm1.b"] = n1.beta
            out[f"{d}norm2"] = n2.gamma
            out[f"{d}norm2.b"] = n2.beta
            ffn = enc.ffn_layers[i]
            out[f"{d}ffn.conv1.weight"] = conv(ffn.conv_1.weight)
            out[f"{d}ffn.conv1.bias"] = ffn.conv_1.bias
            out[f"{d}ffn.conv2.weight"] = conv(ffn.conv_2.weight)
            out[f"{d}ffn.conv2.bias"] = ffn.conv_2.bias
    mr = ep.mrte
    for nm in ("c_pre", "text_pre", "c_post"):
        out[f"enc_p.mrte.{nm}.weight"] = conv(getattr(mr, nm).weight)
        out[f"enc_p.mrte.{nm}.bias"] = getattr(mr, nm).bias
    ca = mr.cross_attention
    for nm in ("conv_q", "conv_k", "conv_v", "conv_o"):
        out[f"enc_p.mrte.cross_attn.{nm}.weight"] = conv(getattr(ca, nm).weight)
        out[f"enc_p.mrte.cross_attn.{nm}.bias"] = getattr(ca, nm).bias

    re_ = model.ref_enc
    out["ref_enc.spectral.0.weight"] = re_.spectral_0.weight
    out["ref_enc.spectral.0.bias"] = re_.spectral_0.bias
    out["ref_enc.spectral.3.weight"] = re_.spectral_1.weight
    out["ref_enc.spectral.3.bias"] = re_.spectral_1.bias
    for i, t in enumerate((re_.temporal_0, re_.temporal_1)):
        out[f"ref_enc.temporal.{i}.w1"] = t.w_1
        out[f"ref_enc.temporal.{i}.b1"] = t.b_1
    sa = re_.slf_attn
    out["ref_enc.slf_attn.w_qs"] = sa.w_qs.weight
    out["ref_enc.slf_attn.w_qs.b"] = sa.w_qs.bias
    out["ref_enc.slf_attn.w_ks"] = sa.w_ks.weight
    out["ref_enc.slf_attn.w_ks.b"] = sa.w_ks.bias
    out["ref_enc.slf_attn.w_vs"] = sa.w_vs.weight
    out["ref_enc.slf_attn.w_vs.b"] = sa.w_vs.bias
    out["ref_enc.slf_attn.fc"] = sa.fc.weight
    out["ref_enc.slf_attn.fc.b"] = sa.fc.bias
    out["ref_enc.fc"] = re_.fc.weight
    out["ref_enc.fc.b"] = re_.fc.bias

    if getattr(model, "ssl_proj", None) is not None:
        out["ssl_proj.weight"] = conv(model.ssl_proj.weight)
        out["ssl_proj.bias"] = model.ssl_proj.bias

    out["bridge.weight"] = conv(model.bridge_0.weight)
    out["bridge.bias"] = model.bridge_0.bias
    wn = model.wns1
    out["wns1.pre.weight"] = conv(wn.pre.weight)
    out["wns1.pre.bias"] = wn.pre.bias
    out["wns1.proj.weight"] = conv(wn.proj.weight)
    out["wns1.proj.bias"] = wn.proj.bias
    out["wns1.cond"] = wn.enc.cond_layer.weight
    out["wns1.cond.b"] = wn.enc.cond_layer.bias
    for wi in range(len(wn.enc.in_layers)):
        out[f"wns1.in.{wi}"] = wn.enc.in_layers[wi].weight
        out[f"wns1.in.{wi}.b"] = wn.enc.in_layers[wi].bias
        out[f"wns1.skip.{wi}"] = wn.enc.res_skip_layers[wi].weight
        out[f"wns1.skip.{wi}.b"] = wn.enc.res_skip_layers[wi].bias
    out["linear_mel.weight"] = conv(model.linear_mel.weight)
    out["linear_mel.bias"] = model.linear_mel.bias

    dit = model.cfm.estimator
    d = "dit."
    out[d + "time_embed.0.w"] = dit.time_embed.time_mlp_0_w
    out[d + "time_embed.0.b"] = dit.time_embed.time_mlp_0_b
    out[d + "time_embed.2.w"] = dit.time_embed.time_mlp_2_w
    out[d + "time_embed.2.b"] = dit.time_embed.time_mlp_2_b
    if dit.use_step_embedding:
        out[d + "d_embed.0.w"] = dit.d_embed.time_mlp_0_w
        out[d + "d_embed.0.b"] = dit.d_embed.time_mlp_0_b
        out[d + "d_embed.2.w"] = dit.d_embed.time_mlp_2_w
        out[d + "d_embed.2.b"] = dit.d_embed.time_mlp_2_b
    for i, blk in enumerate(dit.text_embed.text_blocks):
        b = f"dit.text.{i}."
        out[b + "dw"] = blk.dw
        out[b + "dw_b"] = blk.dw_b
        out[b + "norm_w"] = blk.norm_w
        out[b + "norm_b"] = blk.norm_b
        out[b + "pw1.w"] = blk.pwconv1_w
        out[b + "pw1.b"] = blk.pwconv1_b
        out[b + "pw2.w"] = blk.pwconv2_w
        out[b + "pw2.b"] = blk.pwconv2_b
        out[b + "grn.gamma"] = blk.grn_gamma
        out[b + "grn.beta"] = blk.grn_beta
    ie = dit.input_embed
    out[d + "in.proj.w"] = ie.proj_w
    out[d + "in.proj.b"] = ie.proj_b
    out[d + "in.conv_pos_0.w"] = ie.conv_pos_0_w
    out[d + "in.conv_pos_0.b"] = ie.conv_pos_0_b
    out[d + "in.conv_pos_1.w"] = ie.conv_pos_1_w
    out[d + "in.conv_pos_1.b"] = ie.conv_pos_1_b
    for i, blk in enumerate(dit.transformer_blocks):
        b = f"dit.blocks.{i}."
        out[b + "attn_norm.w"] = blk.attn_norm.linear_w
        out[b + "attn_norm.b"] = blk.attn_norm.linear_b
        out[b + "attn.to_q.w"] = blk.attn.to_q_w
        out[b + "attn.to_q.b"] = blk.attn.to_q_b
        out[b + "attn.to_k.w"] = blk.attn.to_k_w
        out[b + "attn.to_k.b"] = blk.attn.to_k_b
        out[b + "attn.to_v.w"] = blk.attn.to_v_w
        out[b + "attn.to_v.b"] = blk.attn.to_v_b
        out[b + "attn.to_out.w"] = blk.attn.to_out_0_w
        out[b + "attn.to_out.b"] = blk.attn.to_out_0_b
        out[b + "ff.0.w"] = blk.ff.ff_0_0_w
        out[b + "ff.0.b"] = blk.ff.ff_0_0_b
        out[b + "ff.2.w"] = blk.ff.ff_2_w
        out[b + "ff.2.b"] = blk.ff.ff_2_b
    if getattr(dit, "long_skip_w", None) is not None:
        out[d + "long_skip"] = dit.long_skip_w
    out[d + "norm_out.w"] = dit.norm_out.linear_w
    out[d + "norm_out.b"] = dit.norm_out.linear_b
    out[d + "proj_out.w"] = dit.proj_out_w
    out[d + "proj_out.b"] = dit.proj_out_b

    if merged_dit:
        out.update(merged_dit)
    return out


def save_s2v3_merged(weights_dir: str, model, merged_dit: dict | None,
                     model_hps: dict, version: str) -> str:
    """Convenience: s2v3_model_to_arrays + save_s2_inference (fp16 export)."""
    arrays = s2v3_model_to_arrays(model, merged_dit)
    return save_s2_inference(weights_dir, arrays, {"model_hps_source": {
        "model": model_hps, "data": {"sampling_rate": 32000}}}, version,
        dit_depth=len(model.cfm.estimator.transformer_blocks),
        dit_text_blocks=len(model.cfm.estimator.text_embed.text_blocks))


def save_s2_inference(weights_dir: str, model_class_weights: dict, meta: dict,
                      version: str, dit_depth: int | None = None,
                      dit_text_blocks: int | None = None) -> str:
    """Write sovits.safetensors + sovits.json loadable by the pipeline.

    ``model_class_weights``: {name: mx.array} in MLX layout with convert_sovits
    key names (training modules keep these names natively).
    ``meta``: must carry ``model_hps`` source dict with hps["model"]/["data"]
    values; for v3-family also "dit" depth/text_blocks (or pass explicitly).
    Dtype: fp16, except v2Pro/Plus sv keys (sv_emb/ge_to512/prelu) -> fp32.
    """
    os.makedirs(weights_dir, exist_ok=True)
    arrays = {}
    for k, v in model_class_weights.items():
        if version in ("v2Pro", "v2ProPlus") and \
                k.startswith(_SV_FP32_KEYS_PREFIXES):
            arrays[k] = v.astype(mx.float32)
        else:
            arrays[k] = v.astype(mx.float16)
    mx.save_safetensors(os.path.join(weights_dir, "sovits.safetensors"), arrays)
    hps = meta.get("model_hps_source", meta)
    out_meta = {"version": version}
    if version in ("v1", "v2", "v2Pro", "v2ProPlus"):
        src = hps.get("model", hps)
        out_meta["model_hps"] = {k: _json_safe(src.get(k))
                                 for k in _S2_MODEL_HPS_KEYS_V1V2}
    else:
        src = hps.get("model", hps)
        out_meta["model_hps"] = {k: _json_safe(src.get(k))
                                 for k in _S2_MODEL_HPS_KEYS_V3}
        out_meta["model_hps"]["long_skip_connection"] = src.get("long_skip_connection")
        dit = meta.get("dit", {})
        out_meta["dit"] = {
            "depth": dit_depth if dit_depth is not None else dit.get("depth", 0),
            "text_blocks": dit_text_blocks if dit_text_blocks is not None
            else dit.get("text_blocks", 0)}
    data = hps.get("data", {})
    out_meta["data"] = {"sampling_rate": data.get("sampling_rate", 32000)}
    with open(os.path.join(weights_dir, "sovits.json"), "w") as f:
        json.dump(out_meta, f, indent=1)
    return weights_dir
