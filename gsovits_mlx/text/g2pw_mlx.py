"""MLX port of the CPUFast G2PW polyphonic-disambiguation model.

Replaces ``text.g2pw.torch_api`` (the only torch-dependent piece of the
official zh G2P chain). Everything around the model -- data prep
(``prepare_onnx_input``), polyphonic/promise dictionaries, static assets,
pypinyin prefill, sentence dedup, bopomofo->pinyin conversion -- comes from
the official torch-free modules (text/g2pw/base_api.py + dataset.py +
pronunciation.py) executed by gsovits_mlx.text.vendored_cpufront; this file
only re-implements ``G2PWTorchConverter``: model load + one forward.

Model (CPUFast torch_api.G2PWModel, bert-base-chinese sized): embeddings ->
12 transformer layers (post-LN, gelu, 12 heads) -> gather per-query
positions -> pos head (argmax over 11 positions) -> descriptor-sigmoid mask
-> masked softmax over phoneme labels -> argmax. The float checkpoint
carries QuantStub/DeQuantStub residue from an fx quantization export; in
eval they are identity ops, so the float math here is the reference math.

Weights: g2pw.safetensors (fp32 export of G2PWModel/g2pw.pth, created by
tools/export_g2pw_mlx.py; kept fp32 because the argmax decides zh phones
and fp16 logits can flip near-ties). Lookup order:
    $GSOVITS_G2PW_SAFETENSORS
    <official G2PWModel dir>/g2pw.safetensors
    <models_root>/g2pw/g2pw.safetensors   (default /Volumes/2T/.../mlx/g2pw)

Numerical gate: label-exact vs the torch G2PW on the parity corpus (see
tests/test_g2pw_mlx.py).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_NUM_POS = 11
_NUM_HEADS = 12
_MODEL = None  # {"arrays": ..., "n_layers": int}, built lazily

DEFAULT_MODELS_ROOT = "/Volumes/2T/gpt-sovits-models/mlx"


def _abs_model_dir(model_dir: str) -> str:
    """Absolutize the official repo-relative default ("GPT_SoVITS/text/
    G2PWModel") against the vendored text package's checkout root."""
    if os.path.isabs(model_dir):
        return model_dir
    tmod = sys.modules.get("text")
    if tmod is not None and getattr(tmod, "__path__", None):
        # <repo>/GPT_SoVITS/text -> <repo>
        repo_root = os.path.dirname(os.path.dirname(tmod.__path__[0]))
        return os.path.join(repo_root, model_dir)
    return os.path.abspath(model_dir)


def _resolve_weights(model_dir: str) -> str:
    model_dir = _abs_model_dir(model_dir)
    explicit = os.environ.get("GSOVITS_G2PW_SAFETENSORS", "")
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.append(os.path.join(model_dir, "g2pw.safetensors"))
    candidates.append(os.path.join(DEFAULT_MODELS_ROOT, "g2pw", "g2pw.safetensors"))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "g2pw.safetensors not found (tried: "
        + ", ".join(c for c in candidates if c)
        + "); export it with tools/export_g2pw_mlx.py or set "
        "GSOVITS_G2PW_SAFETENSORS")


def _load_model(model_dir: str) -> dict:
    import mlx.core as mx

    arrays = mx.load(_resolve_weights(model_dir))
    n_layers = 1 + max(int(k.split(".")[3]) for k in arrays
                       if k.startswith("bert.encoder.layer."))
    return {"arrays": arrays, "n_layers": n_layers}


def get_model(model_dir: str) -> dict:
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model(model_dir)
    return _MODEL


def _gelu(x):
    import mlx.core as mx

    return nn_gelu(x)


def nn_gelu(x):
    """erf-based gelu (transformers F.gelu default), on MLX."""
    import mlx.core as mx

    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def _forward(w: dict, input_ids, token_type_ids, attention_mask,
             phoneme_mask, char_ids, position_ids):
    """G2PWModel.forward in MLX. Index arrays int32; masks float32.

    Shapes: input_ids/token_type_ids/attention_mask (B,S); phoneme_mask/
    char_ids/position_ids (B,N). Returns probs (B,N,L).
    """
    import mlx.core as mx

    arrays = w["arrays"]
    B, S = input_ids.shape
    x = (arrays["bert.embeddings.word_embeddings.weight"][input_ids]
         + arrays["bert.embeddings.position_embeddings.weight"][:S][None]
         + arrays["bert.embeddings.token_type_embeddings.weight"][token_type_ids])
    x = mx.fast.layer_norm(
        x, arrays["bert.embeddings.LayerNorm.weight"],
        arrays["bert.embeddings.LayerNorm.bias"], 1e-12)
    # additive key mask, official (1 - am) * -10000, broadcast over heads/queries
    m = (1.0 - attention_mask[:, None, None, :]) * -10000.0
    dim = x.shape[-1]
    hd = dim // _NUM_HEADS
    scale = 1.0 / math.sqrt(hd)
    for i in range(w["n_layers"]):
        p = f"bert.encoder.layer.{i}."
        q = (x @ arrays[p + "attention.self.query.weight"].T
             + arrays[p + "attention.self.query.bias"])
        k = (x @ arrays[p + "attention.self.key.weight"].T
             + arrays[p + "attention.self.key.bias"])
        v = (x @ arrays[p + "attention.self.value.weight"].T
             + arrays[p + "attention.self.value.bias"])
        q = q.reshape(B, S, _NUM_HEADS, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, _NUM_HEADS, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, _NUM_HEADS, hd).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=m)
        a = a.transpose(0, 2, 1, 3).reshape(B, S, dim)
        a = mx.fast.layer_norm(
            a @ arrays[p + "attention.output.dense.weight"].T
            + arrays[p + "attention.output.dense.bias"] + x,
            arrays[p + "attention.output.LayerNorm.weight"],
            arrays[p + "attention.output.LayerNorm.bias"], 1e-12)
        h = _gelu(a @ arrays[p + "intermediate.dense.weight"].T
                  + arrays[p + "intermediate.dense.bias"]) if False else _gelu(a @ arrays[p + "intermediate.dense.weight"].T + arrays[p + "intermediate.dense.bias"])
        x = mx.fast.layer_norm(
            h @ arrays[p + "output.dense.weight"].T
            + arrays[p + "output.dense.bias"] + a,
            arrays[p + "output.LayerNorm.weight"],
            arrays[p + "output.LayerNorm.bias"], 1e-12)
    # gather per-query positions: flat[(arange(B)*S + position_ids)]
    flat = x.reshape(B * S, dim)
    idx = mx.arange(B)[:, None] * S + position_ids
    hq = flat[idx]  # (B, N, dim)
    pos_logits = hq @ arrays["pos_classifier.weight"].T + arrays["pos_classifier.bias"]
    pos_pred = mx.argmax(pos_logits, axis=-1)  # (B, N)
    # descriptor mask: bias (1,L) + char (B,N,L) + second (B,N,L) -> (B,N,L)
    mask_w = (mx.sigmoid(arrays["descriptor_bias.weight"]
                         + arrays["char_descriptor.weight"][char_ids]
                         + arrays["second_order_descriptor.weight"][char_ids * _NUM_POS + pos_pred])
              * phoneme_mask)
    logits = hq @ arrays["classifier.weight"].T + arrays["classifier.bias"]
    logits_max = mx.max(logits, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits - logits_max) * mask_w
    return exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)


def _predict_numpy(w: dict, model_input: dict):
    import mlx.core as mx

    probs = np.array(_forward(
        w,
        mx.array(np.asarray(model_input["input_ids"], np.int32)),
        mx.array(np.asarray(model_input["token_type_ids"], np.int32)),
        mx.array(np.asarray(model_input["attention_masks"], np.float32)),
        mx.array(np.asarray(model_input["phoneme_masks"], np.float32)),
        mx.array(np.asarray(model_input["char_ids"], np.int32)),
        mx.array(np.asarray(model_input["position_ids"], np.int32)),
    ))
    probs = probs.reshape(-1, probs.shape[-1])  # (B=1, N, L) -> (N, L)
    preds = np.argmax(probs, axis=1).tolist()
    confidences = [float(probs[i, p]) for i, p in enumerate(preds)]
    return preds, confidences


def register_as_torch_api() -> None:
    """Insert this module as text.g2pw.torch_api in sys.modules."""
    sys.modules["text.g2pw.torch_api"] = sys.modules[__name__]


def _make_converter_class():
    from text.g2pw import base_api  # vendored module, loaded by vendored_cpufront

    class G2PWTorchConverter(base_api._G2PWBaseConverter):
        """Drop-in for torch_api.G2PWTorchConverter (base_api does the rest)."""

        def __init__(self, model_dir: str = "GPT_SoVITS/text/G2PWModel/",
                     style: str = "bopomofo", model_source: str = None,
                     enable_non_tradional_chinese: bool = False):
            base_api._G2PWBaseConverter.__init__(
                self, model_dir=model_dir, style=style,
                model_source=model_source,
                enable_non_tradional_chinese=enable_non_tradional_chinese)
            get_model(self.model_dir)

        def _predict(self, model_input: dict):
            preds, confidences = _predict_numpy(get_model(self.model_dir),
                                                model_input)
            return [self.labels[p] for p in preds], confidences

    return G2PWTorchConverter


# Built lazily so importing this module never requires the vendored package;
# vendored_cpufront loads base_api first, then registers us as torch_api.
G2PWTorchConverter = None


def ensure_converter_class():
    global G2PWTorchConverter
    if G2PWTorchConverter is None:
        G2PWTorchConverter = _make_converter_class()
    return G2PWTorchConverter
