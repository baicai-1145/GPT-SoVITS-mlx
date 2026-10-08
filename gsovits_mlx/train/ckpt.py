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
