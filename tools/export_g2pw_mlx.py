#!/usr/bin/env python3
"""Export the CPUFast G2PW float checkpoint (G2PWModel/g2pw.pth) to
g2pw.safetensors for the torch-free MLX front-end.

One-off conversion, run in a torch env (e.g. .tmp/refenv):
    python tools/export_g2pw_mlx.py \
        --ckpt /path/to/GPT-SoVITS-CPUFast/GPT_SoVITS/text/G2PWModel/g2pw.pth \
        --out /Volumes/2T/gpt-sovits-models/mlx/g2pw/g2pw.safetensors

What is kept: every float tensor of G2PWModel except bert.pooler.* and the
bert.embeddings.position_ids buffer (not used by forward). QuantStub /
DeQuantStub are parameterless nn modules — they store nothing. The
"Int8"-style wrapper classes only apply to g2pw_int8.pth, which CPUFast
loads only when it exists (CPUFast's G2PWModel has no g2pw_int8.pth; the
float g2pw.pth path is the reference).

The safetensors is fp32 (matches the checkpoint); G2PW stays fp32 at
runtime because its argmax decides zh phones (fp16 logits can flip
near-ties).
"""

from __future__ import annotations

import argparse
import os

EXCLUDE_PREFIXES = ("bert.pooler.",)
EXCLUDE_KEYS = {"bert.embeddings.position_ids"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="path to G2PWModel/g2pw.pth")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    args = ap.parse_args()

    import torch
    from safetensors.numpy import save_file

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    out = {}
    for k, v in sd.items():
        if k in EXCLUDE_KEYS or k.startswith(EXCLUDE_PREFIXES):
            continue
        if not isinstance(v, torch.Tensor):
            print(f"skip non-tensor {k}: {type(v)}")
            continue
        if v.dtype not in (torch.float32, torch.float64):
            print(f"skip non-float {k}: {v.dtype}")
            continue
        out[k] = v.to(torch.float32).contiguous().numpy()

    # sanity: architecture fingerprint (bert-base-chinese sized, 12 layers)
    n_layers = 1 + max(int(k.split(".")[3]) for k in out
                       if k.startswith("bert.encoder.layer."))
    checks = {
        "vocab": out["bert.embeddings.word_embeddings.weight"].shape,
        "n_layers": n_layers,
        "num_chars": out["char_descriptor.weight"].shape[0],
        "num_labels": out["classifier.weight"].shape[0],
        "second_order": out["second_order_descriptor.weight"].shape,
    }
    print("architecture:", checks)
    # CPUFast G2PWModel: bert-base-chinese arch trained w/ roberta-large-zh vocab
    assert checks["vocab"] == (21128, 768), checks
    assert n_layers == 12, n_layers
    assert checks["num_labels"] == 1305, checks          # bopomofo label set
    assert checks["second_order"][0] == checks["num_chars"] * 11, checks

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(out, args.out)
    n_bytes = sum(v.nbytes for v in out.values())
    print(f"wrote {args.out}: {len(out)} arrays, {n_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
