"""s2 (SoVITS) GAN training driver — MLX port of official s2_train.py.

Usage (single process; the official multi-GPU DDP path collapses to
num_replicas=1 here):

  # one-time checkpoint extraction (base venv, torch):
  /Users/baicai1145/.venvs/base/bin/python tools/train_s2_extract.py \
      --version v2 --out .tmp/train_s2

  # training run (repo venv, GPU behind the lock):
  python tools/train_s2.py --version v2 \
      --exp-dir <exp_dir from tools/prepare_data.py> \
      --train-npz .tmp/train_s2/s2G_v2.npz \
      [--disc-npz .tmp/train_s2/s2D_v2.npz] \
      --out .tmp/train_s2/run_v2 --steps 200 --batch-size 6

Semantics (official s2_train.py, verified line-by-line):
  * AdamW G: 4 param groups — base lr 1e-4; text_embedding / encoder_text /
    mrte each at lr*0.4 (text_low_lr_rate); betas (0.8, 0.99), eps 1e-9,
    weight_decay 0.01 (torch default).
  * AdamW D: single group, same hyperparams.
  * SHARED GradScaler: D step scales+steps on the pre-update scale;
    scaler.update() runs ONLY after the G step.
  * ExponentialLR gamma=0.999875 stepped once per epoch.
  * clip_grad_value_(None): NO-OP on grads (official commons only clamps
    when clip_value is not None); we compute the grad norm for logging.
  * freeze_quantizer=True: ssl_proj + quantizer excluded from optimizers
    (official no_grad+eval); quantizer runs fp32 outside the low-precision
    graph.
  * fp16 autocast for the G forward; fp32 losses (mel/kl/fm/gen).

G-forward modes (--g-forward):
  * single (default): ONE G forward per step inside ONE traced
    value_and_grad over (d16, p16) jointly — official torch structure
    (official runs net_g() once and reuses y_hat: net_d(y, y_hat.detach())
    for D, net_d(y, y_hat) for G). The D branch sees y_hat through
    stop_gradient (grad-free, like .detach()); the G branch sees D weights
    through stop_gradient (D as constants). Both losses + the 5 loss parts
    come back as the aux output of the single trace. Official RNG
    semantics: ONE rand_slice_segments draw + ONE posterior-noise draw per
    step, shared by both branches.
  * double (legacy): the pre-optimization structure — separate D-step and
    G-step traces, each with its OWN G forward. Matches the old behavior
    except the RNG: both forwards now use the SAME key (official
    single-draw semantics; the legacy code drew kd/kg separately, an
    undocumented deviation). --double-split-draw restores the exact legacy
    RNG (different draws per forward) for A/B.

Deviations (documented, unavoidable or measured):
  * G-step D forward uses the PRE-update D weights (official torch
    sequential-overlap artifact: net_d.step() has updated the .data of the
    same tensors the G-step forward reads; MLX traced grads need frozen
    inputs, so the G step sees the pre-step D). Measured immaterial on
    200-step smokes; kept identical across all modes/precisions here.
  * DDP gradient averaging absent (single process).
  * Bucket sampler randperm: numpy PCG64, not torch Philox (s2_data.py).
  * v1/v2 have NO pretrained s2D on this machine (checked /Volumes/2T
    .../pretrained_models and the whole repo tree): D inits fresh for
    v1/v2; v2Pro/ProPlus load s2Dv2Pro*.pth.

Precision (--precision):
  * fp32: no autocast, no GradScaler (official s2 config default,
    fp16_run=false). fp16 is proven unstable here: KL exp(-2*logs_p)
    overflows fp16 range and the scaler collapses (scale 1e-56 in the v2
    smoke); use fp32.
  * fp16: autocast+GradScaler (official fp16_run=True semantics).
    Retained for reproducibility; unstable (see above).
  * bf16: bf16 G/D compute graph, fp32 loss heads (mel/kl/fm/gen cast to
    fp32 at entry — KL's exp(-2*logs_p) is evaluated in fp32; bf16 shares
    fp32's exponent range so the fp16 overflow mode cannot occur), fp32
    masters/optimizer, GradScaler bypassed (CFM precedent: bf16 needs no
    scaler). Frozen fp32 exemptions kept: ssl_proj/quantizer (frozen32) and
    sv_emb/ge_to512/prelu (v2Pro export policy). OPT-IN; fp32 stays the
    default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from gsovits_mlx.gpu_lock import resolve_device  # noqa: E402

_HPS_V1V2 = dict(
    spec_channels=1025, inter_channels=192, hidden_channels=192,
    filter_channels=768, n_heads=2, n_layers=6, kernel_size=3,
    resblock="1", resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5]] * 3,
    upsample_rates=[10, 8, 2, 2, 2], upsample_initial_channel=512,
    upsample_kernel_sizes=[16, 16, 8, 2, 2], gin_channels=512,
    semantic_frame_rate="25hz", freeze_quantizer=True)
VERSION_MODEL_HPS = {
    "v1": dict(_HPS_V1V2),
    "v2": dict(_HPS_V1V2),
    "v2Pro": dict(_HPS_V1V2, gin_channels=1024),
    "v2ProPlus": dict(_HPS_V1V2, gin_channels=1024,
                      upsample_initial_channel=768,
                      upsample_kernel_sizes=[20, 16, 8, 2, 2]),
}
TRAIN = dict(
    learning_rate=1e-4, betas=(0.8, 0.99), eps=1e-9, fp16_run=True,
    lr_decay=0.999875, segment_size=20480, hop_length=640,
    filter_length=2048, win_length=2048, n_mel_channels=128,
    mel_fmin=0.0, mel_fmax=None, c_mel=45.0, c_kl=1.0,
    text_low_lr_rate=0.4, sampling_rate=32000,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", required=True,
                   choices=["v1", "v2", "v2Pro", "v2ProPlus"])
    p.add_argument("--exp-dir", required=True,
                   help="exp_dir from tools/prepare_data.py (2-name2text.txt, "
                        "4-cnhubert/, 5-wav32k/ [, 7-sv_cn/])")
    p.add_argument("--train-npz", required=True,
                   help="s2G training npz from tools/train_s2_extract.py")
    p.add_argument("--disc-npz", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=TRAIN["learning_rate"])
    p.add_argument("--lr-scale", type=float, default=1.0,
                   help="extra multiplier on --lr (e.g. batch_size/official "
                        "32 for smoke stability; default 1.0 = official)")
    p.add_argument("--warmup-steps", type=int, default=0,
                   help="linear lr warmup steps (official has none; smoke "
                        "uses a few to avoid a first-batch D spike)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--export-inference", action="store_true")
    p.add_argument("--init-scale", type=float, default=65536.0)
    p.add_argument("--precision", choices=("fp16", "fp32", "bf16"),
                   default="fp16",
                   help="fp16: autocast+GradScaler (official fp16_run=True "
                        "semantics); fp32: no autocast/scaler — official "
                        "s2 config default (fp16_run=false); bf16: bf16 G/D "
                        "compute + fp32 loss heads + fp32 masters/optimizer, "
                        "no GradScaler (CFM precedent). fp16 proven "
                        "unstable here: KL exp(-2*logs_p) overflows fp16 "
                        "range and the scaler collapses (scale 1e-56 in "
                        "the v2 smoke); use fp32 (default for real runs) "
                        "or bf16 (opt-in speed path).")
    p.add_argument("--g-forward", choices=("single", "double"), default="single",
                   help="single: ONE G forward per step inside one joint "
                        "(D,G) grad trace — official torch structure (net_g "
                        "runs once; y_hat detached-reused). double: legacy "
                        "two-trace structure (G forward in the D trace AND "
                        "in the G trace) for A/B.")
    p.add_argument("--double-split-draw", action="store_true",
                   help="(double mode only) use DIFFERENT RNG keys for the "
                        "two G forwards — exact legacy RNG behavior "
                        "(pre-official-parity fix). Default: same key for "
                        "both forwards (official single-draw semantics).")
    p.add_argument("--bf16-strict-noise", action="store_true", default=True)
    p.add_argument("--no-bf16-strict-noise", dest="bf16_strict_noise",
                   action="store_false",
                   help="posterior noise cast to the compute dtype in "
                        "low-precision modes (bf16/fp16) so the decoder "
                        "input z_slice actually computes in low precision "
                        "(the fp32 randn silently promotes flow+dec). "
                        "Default on; disable for exact-legacy fp16 numerics.")
    p.add_argument("--pad-multiple", type=int, default=None,
                   help="quantize collated ssl/spec/text widths (and wav, in "
                        "hop units) up to multiples of N frames — CFM "
                        "trainer's proven pattern; recycles Metal buffer "
                        "sizes; OFF by default (official collate semantics).")
    p.add_argument("--fixed-shape", action="store_true",
                   help="force ALL batches to one global padded shape (dataset "
                        "max widths quantized to 64). Extra pad is masked "
                        "from losses exactly like official padding (zero "
                        "columns beyond every length; masks/slices length-"
                        "bounded). Recycles ONE working set of buffer sizes; "
                        "mx.compile prerequisite.")
    p.add_argument("--compile-mode", choices=("none", "step", "fwd"),
                   default="none",
                   help="mx.compile experiment (needs --fixed-shape for stable "
                        "shapes): none=default eager; step=wrap the phase-"
                        "split eval'd step helper in mx.compile; fwd=wrap the "
                        "inner G/D forward functions (finer granularity). "
                        "Exploratory — expect MLX 0.32.2 failures on custom "
                        "ops/dynamic shapes; results recorded in TRAINING.md.")
    p.add_argument("--memory-limit-mb", type=int,
                   default=int(os.environ.get("GSOVITS_METAL_LIMIT_MB", "8192")),
                   help="Metal wired limit; 8GB default for training (lead "
                        "advisory after a 13GB runaway kill; env override)")
    p.add_argument("--cpu", action="store_true",
                   help="debug-only CPU run (tiny steps; the full model "
                        "does NOT fit CPU time budgets)")
    return p.parse_args(argv)


_lock_held = False


def main(argv=None) -> int:
    global _lock_held
    args = parse_args(argv)
    # GPU LAW (task-6): canonical main-repo lock + ps guard, mirroring
    # tools/train_s2_v3.py — acquire BEFORE resolve_device (the resolver
    # grants gpu only when a fresh lock exists). One GPU process machine-wide.
    if not args.cpu:
        import subprocess as _sp
        procs = _sp.run(["ps", "ax", "-o", "command"], capture_output=True,
                        text=True).stdout.splitlines()
        gpu_procs = [ln.strip() for ln in procs
                     if ("train" in ln or "smoke" in ln or "prepare" in ln
                         or "tools/e2e" in ln)
                     and "python" in ln and "train_s2.py" not in ln
                     and "grep" not in ln]
        if gpu_procs:
            raise SystemExit("[gpu.lock] other GPU python processes running:\n  "
                             + "\n  ".join(gpu_procs[:5]))
        from gsovits_mlx.gpu_lock import acquire_lock
        acquire_lock(f"tools/train_s2.py s2-GAN {args.version} "
                     f"gfw={args.g_forward} prec={args.precision} "
                     f"gan-speed-opt")
        _lock_held = True
    import mlx.core as mx
    device = resolve_device(flag_gpu=not args.cpu, verbose=True)
    try:
        return _run(args, device, mx)
    finally:
        if _lock_held:
            from gsovits_mlx.gpu_lock import release_lock
            release_lock()


def _run(args, device, mx):
    if device == "gpu":
        try:
            mx.set_memory_limit(args.memory_limit_mb * 1024 * 1024)
        except AttributeError:
            mx.metal.set_memory_limit(args.memory_limit_mb * 1024 * 1024)
        except Exception as e:  # noqa: BLE001
            print(f"[mem] limit not set: {e}", file=sys.stderr)

    from gsovits_mlx.train.s2_data import (
        BucketSampler, TextAudioSpeakerCollate, TextAudioSpeakerLoader)
    from gsovits_mlx.train import s2_gan as G
    from gsovits_mlx.train.s2_discriminator import (
        MultiPeriodDiscriminator, load_mpd_weights)
    from gsovits_mlx.train.s2_torchio import load_train_params_npz
    from gsovits_mlx.train.optim import AdamW
    from gsovits_mlx.train.mixed_precision import GradScaler
    from gsovits_mlx.train import ckpt as ckpt_mod
    from gsovits_mlx.text.mel_frontend import _librosa_mel

    os.makedirs(args.out, exist_ok=True)
    hps = VERSION_MODEL_HPS[args.version]
    seg_frames = TRAIN["segment_size"] // TRAIN["hop_length"]  # 32
    is_pro = args.version in ("v2Pro", "v2ProPlus")

    # ---- data ----
    dataset = TextAudioSpeakerLoader(
        args.exp_dir, version=args.version,
        sampling_rate=TRAIN["sampling_rate"],
        filter_length=TRAIN["filter_length"],
        hop_length=TRAIN["hop_length"], win_length=TRAIN["win_length"])
    print(f"[data] {dataset.stats}")
    fixed_widths = None
    if args.fixed_shape:
        # one global padded shape: dataset maxima quantized to 64
        # (spec/ssl in frames, wav in samples = spec_frames*640, text rows)
        q = lambda x: ((x + 63) // 64) * 64
        mx_spec = max(int(l) for l in dataset.lengths)
        # wav samples = frames*hop exactly (stft center=False, hop 640)
        fixed_widths = dict(ssl=q(mx_spec), spec=q(mx_spec),
                            wav=q(mx_spec) * 640)
        # text width: max phoneme-id count over the dataset (cheap scan)
        mx_text = max(len(ds_t[1]) for ds_t in dataset.audiopaths_sid_text)
        fixed_widths["text"] = q(mx_text)
        print(f"[fixed-shape] spec/ssl {fixed_widths['spec']} frames, wav "
              f"{fixed_widths['wav']} samples, text {fixed_widths['text']}")
    collate = TextAudioSpeakerCollate(version=args.version,
                                      pad_multiple=args.pad_multiple,
                                      fixed_widths=fixed_widths)
    sampler = BucketSampler(dataset.lengths, args.batch_size, shuffle=True)

    # ---- models ----
    # Seed the GLOBAL mx RNG before model construction (official
    # torch.manual_seed(hps.train.seed)): the fresh-D init draws from the
    # global stream; without this the SAME config is not reproducible
    # run-to-run (measured: 3 runs, 3 different step-0 loss_mel values —
    # loss_mel is D-independent, so the divergence was upstream: D-init
    # consumed different global-stream offsets, shifting the split-derived
    # step keys). Step RNG is unaffected (key = mx.random.key(seed)).
    mx.random.seed(args.seed)
    net_g = G.SynthesizerTrnTrain(version=args.version, segment_size=seg_frames,
                                  **hps)
    params32 = load_train_params_npz(args.train_npz)
    missing = [n for n in net_g.parameter_names(include_frozen=True)
               if n not in params32]
    if missing:
        raise RuntimeError(f"s2G npz missing {len(missing)} params: {missing[:8]}")
    net_g.bind(params32)

    periods = [2, 3, 5, 7, 11, 17, 23] if is_pro else [2, 3, 5, 7, 11]
    net_d = MultiPeriodDiscriminator(periods=periods)
    d_fresh = True
    if args.disc_npz and os.path.exists(args.disc_npz):
        d_src = load_train_params_npz(args.disc_npz)
        miss, unexp = load_mpd_weights(net_d, d_src)
        d_fresh = False
        print(f"[init] D loaded ({len(miss)} missing / {len(unexp)} unexpected)")
    else:
        print(f"[init] no pretrained s2D for {args.version}: D init fresh "
              "(documented; official expects s2D for v1/v2 but it is absent "
              "on this machine)")

    # ---- optimizers ----
    all_names = net_g.parameter_names()
    text_emb = [n for n in all_names if n == "enc_p.text_embedding.weight"]
    enc_text = [n for n in all_names if n.startswith("enc_p.encoder_text")]
    mrte = [n for n in all_names if n.startswith("enc_p.mrte")]
    low = set(text_emb + enc_text + mrte)
    base = [n for n in all_names if n not in low]
    assert set(base) | low == set(all_names) and not (set(base) & low)

    def gdict(names):
        return {n: params32[n] for n in names}

    lr_eff = args.lr * args.lr_scale
    optim_g = AdamW(
        [{"params": gdict(base), "lr": lr_eff},
         {"params": gdict(text_emb), "lr": lr_eff * TRAIN["text_low_lr_rate"]},
         {"params": gdict(enc_text), "lr": lr_eff * TRAIN["text_low_lr_rate"]},
         {"params": gdict(mrte), "lr": lr_eff * TRAIN["text_low_lr_rate"]}],
        lr=lr_eff, betas=TRAIN["betas"], eps=TRAIN["eps"], weight_decay=0.01)
    for g_ in optim_g.param_groups:
        g_["base_lr"] = g_["lr"]

    d32 = net_d.parameters()  # {name: fp32 array}
    optim_d = AdamW([d32], lr=lr_eff, betas=TRAIN["betas"],
                    eps=TRAIN["eps"], weight_decay=0.01)
    optim_d.param_groups[0]["base_lr"] = lr_eff

    sched_g = G.ExponentialLR(optim_g, gamma=TRAIN["lr_decay"])
    sched_d = G.ExponentialLR(optim_d, gamma=TRAIN["lr_decay"])
    scaler = GradScaler(init_scale=args.init_scale)

    mel_basis = mx.array(_librosa_mel(
        TRAIN["sampling_rate"], TRAIN["filter_length"],
        TRAIN["n_mel_channels"], TRAIN["mel_fmin"], TRAIN["mel_fmax"]))

    # frozen fp32 block (official autocast-disabled): ssl_proj + quantizer
    # stay fp32 and are never optimized; carried inside p16 so bind() sees
    # them and the quantizer math matches official exactly.
    frozen32 = {n: params32[n] for n in net_g.parameter_names(include_frozen=True)
                if n not in set(all_names)}

    compute_dtype = {"fp16": mx.float16, "fp32": mx.float32,
                     "bf16": mx.bfloat16}[args.precision]

    def make_compute(params_f32: dict) -> dict:
        """Working copy of G params in the compute dtype (or fp32).

        fp32 exemptions (unchanged across modes):
          * sv_emb./ge_to512./prelu — v2Pro export policy (F32 exports);
          * the frozen32 block (ssl_proj/quantizer) is merged back fp32.
        """
        out = {}
        for k, v in params_f32.items():
            if args.precision == "fp32":
                out[k] = v
            elif k.startswith(("sv_emb.", "ge_to512.", "prelu")):
                out[k] = v  # sv family stays fp32 (v2Pro export policy)
            else:
                out[k] = v.astype(compute_dtype)
        out.update(frozen32)  # fp32 constants (ssl_proj/quantizer)
        return out

    # ---- traced losses --------------------------------------------------------
    def prep_batch(batch):
        arrays = [mx.array(b) for b in batch[:8]]
        sv = mx.array(batch[8]) if is_pro else None
        return arrays, sv

    def g_forward(p16, arrays, sv, key):
        (ssl, ssl_lengths, spec, spec_lengths, wav, wav_lengths, text,
         text_lengths) = arrays
        net_g.bind(p16)
        if args.precision == "fp32":
            spec_in = spec
        else:
            spec_in = spec.astype(compute_dtype)
        y_lengths = spec_lengths  # y frames == spec frames (official)
        # frozen fp32 quantizer block (outside the low-precision graph);
        # commit value feeds loss_gen_all (official kl_ssl*1) and is logged
        quant, commit = net_g.quantize_ssl(ssl)
        o, kl_ssl, ids_slice, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), _ = \
            net_g.forward(ssl, spec_in, y_lengths, text.astype(mx.int32),
                          text_lengths, sv_emb=sv, key=key,
                          quantized_in=(quant, commit),
                          strict_noise=args.bf16_strict_noise)
        mel = G.spec_to_mel(spec.astype(mx.float32), mel_basis)
        if net_g.use_gather_slice:
            # compile-safe path (fwd arm): loop slice is a graph break
            y_mel = G.slice_segments_gather(mel, ids_slice, seg_frames)
            y = G.slice_segments_gather(
                wav, ids_slice * TRAIN["hop_length"],
                TRAIN["segment_size"])
        else:
            y_mel = G.slice_segments(mel, ids_slice, seg_frames)
            y = G.slice_segments(wav, ids_slice * TRAIN["hop_length"],
                                 TRAIN["segment_size"])
        y_hat_mel = G.mel_spectrogram_train(
            o.astype(mx.float32).squeeze(1), mel_basis,
            TRAIN["filter_length"], TRAIN["hop_length"], TRAIN["win_length"])
        return o, y, y_mel, y_hat_mel, kl_ssl, commit, \
            (z, z_p, m_p, logs_p, m_q, logs_q), y_mask

    if args.compile_mode == "fwd":
        # fwd arm: compile the driver's whole g_forward helper (weights as
        # args — p16 rebinds inside; the gather slice keeps it traceable)
        g_forward = mx.compile(g_forward)

    # legacy helper retained for double mode
    def d_loss_fn(d16, p16, arrays, sv, key):
        """D loss wrt d16; G forward under captured p16 (constants)."""
        if d_call_c is not None:
            o, y, *_ = g_forward(p16, arrays, sv, key)
            y32 = y.astype(mx.float32)
            y_hat32 = mx.stop_gradient(o).astype(mx.float32)
            y_d_hat_r, y_d_hat_g, _, _ = d_call_c(d16, y32, y_hat32)
        else:
            load_mpd_weights(net_d, d16)
            o, y, *_ = g_forward(p16, arrays, sv, key)
            y32 = y.astype(mx.float32)
            y_hat32 = mx.stop_gradient(o).astype(mx.float32)
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y32, y_hat32)
        loss, _, _ = G.discriminator_loss(y_d_hat_r, y_d_hat_g)
        return loss

    def g_loss_fn(p16, d16, arrays, sv, key):
        """G loss wrt p16; D under captured d16 (constants)."""
        o, y, y_mel, y_hat_mel, kl_ssl, commit, ql, y_mask = g_forward(
            p16, arrays, sv, key)
        z, z_p, m_p, logs_p, m_q, logs_q = ql
        loss_mel = mx.mean(mx.abs(y_mel - y_hat_mel)) * TRAIN["c_mel"]
        loss_kl = G.kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * TRAIN["c_kl"]
        y32 = y.astype(mx.float32)
        if d_call_c is not None:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = d_call_c(
                d16, y32, o.astype(mx.float32))
        else:
            load_mpd_weights(net_d, d16)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(
                y32, o.astype(mx.float32))
        loss_fm = G.feature_loss(fmap_r, fmap_g)
        loss_gen, _ = G.generator_loss(y_d_hat_g)
        # official: loss_gen + loss_fm + loss_mel + kl_ssl * 1 + loss_kl
        return (loss_gen + loss_fm + loss_mel + kl_ssl * 1.0 + loss_kl,
                (loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl))

    # -- mx.compile experiment, fwd arm (ITEM 2) ---------------------------
    # ONE compiled wrapper over the driver's g_forward helper (below —
    # defined before this point and referenced lazily via closure), plus
    # one over the D call. Weights are ARGUMENTS (never captured): a
    # captured mx array becomes a compile-time constant and the
    # optimizer's updates would be silently ignored (stale-weight hazard).
    # Under fwd, g_forward's mel/wav segment slices use the gather variant
    # (the loop slice's int(ids_str[i]) is a compile graph break that
    # bakes stale offsets). None = eager (official semantics).
    d_call_c = None
    if args.compile_mode == "fwd":
        def _d_call_c(d16_, y32_, yhat32_):
            load_mpd_weights(net_d, d16_)
            return net_d(y32_, yhat32_)
        d_call_c = mx.compile(_d_call_c)

    # compiled G-forward (fwd arm) must use the gather slice: the loop's
    # int(ids_str[i]) is a graph break under mx.compile (bakes stale offsets)
    net_g.use_gather_slice = (args.compile_mode == "fwd")

    def fused_loss_fn(d16, p16, arrays, sv, key):
        """Single-trace joint (D, G) loss — official one-forward structure.

        Returns (loss_d + loss_g, (loss_d, gen_parts...)) with gradient
        paths partitioned by stop_gradient:
          * D branch: y_hat detached (== official net_d(y, y_hat.detach()));
            grads flow ONLY into d16.
          * G branch: D weights detached (D is a constant function — our
            documented pre-update-D deviation); grads flow ONLY into p16.
        One rand_slice + one posterior-noise draw (key) shared by both
        branches — official single-draw RNG.
        d_call_c (the --compile-mode fwd experiment) is a compiled D call;
        the G forward here reuses g_forward's body inline (eager) — the
        fwd arm compiles g_forward itself where the double traces use it.
        """
        # ---- shared single G forward (official net_g call #1) ----
        (ssl, ssl_lengths, spec, spec_lengths, wav, wav_lengths, text,
         text_lengths) = arrays
        net_g.bind(p16)
        if args.precision == "fp32":
            spec_in = spec
        else:
            spec_in = spec.astype(compute_dtype)
        y_lengths = spec_lengths
        quant, commit = net_g.quantize_ssl(ssl)
        o, y, y_mel, y_hat_mel, kl_ssl, _commit, ql, y_mask = g_forward(
            p16, arrays, sv, key)
        z, z_p, m_p, logs_p, m_q, logs_q = ql
        y32 = y.astype(mx.float32)

        # ---- D branch (official: net_d(y, y_hat.detach())) ----
        if d_call_c is not None:
            y_d_hat_r_d, y_d_hat_g_d, _, _ = d_call_c(
                d16, y32, mx.stop_gradient(o).astype(mx.float32))
        else:
            load_mpd_weights(net_d, d16)
            y_d_hat_r_d, y_d_hat_g_d, _, _ = net_d(
                y32, mx.stop_gradient(o).astype(mx.float32))
        loss_d, _, _ = G.discriminator_loss(y_d_hat_r_d, y_d_hat_g_d)

        # ---- G branch (official: net_d(y, y_hat) on updated D; ours: pre-update D) ----
        d_const = {k: mx.stop_gradient(v) for k, v in d16.items()}
        if d_call_c is not None:
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = d_call_c(
                d_const, y32, o.astype(mx.float32))
        else:
            load_mpd_weights(net_d, d_const)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y32, o.astype(mx.float32))
        loss_mel = mx.mean(mx.abs(y_mel - y_hat_mel)) * TRAIN["c_mel"]
        loss_kl = G.kl_loss(z_p, logs_q, m_p, logs_p, y_mask) * TRAIN["c_kl"]
        loss_fm = G.feature_loss(fmap_r, fmap_g)
        loss_gen, _ = G.generator_loss(y_d_hat_g)
        loss_g_total = loss_gen + loss_fm + loss_mel + kl_ssl * 1.0 + loss_kl
        return loss_d + loss_g_total, (loss_d, loss_gen, loss_fm, loss_mel,
                                       kl_ssl, loss_kl)

    # ---- loop -----------------------------------------------------------------
    # step arm: wrap the joint value_and_grad helper in mx.compile. Outputs
    # stay lazy; the loop below still evals phase-1 (D branch + dgrads)
    # then phase-2 (G parts + ggrads) exactly like eager — compile cannot
    # fuse both backwards into one peak (the 8GB-gate lesson from the
    # first fused attempt).
    if args.compile_mode == "step":
        def _joint_step(d16_, p16_, ssl_, ssl_len_, spec_, spec_len_, wav_,
                        wav_len_, text_, text_len_, sv_, key_):
            arrays_ = (ssl_, ssl_len_, spec_, spec_len_, wav_, wav_len_,
                       text_, text_len_)
            (total, parts), (dgrads, ggrads) = mx.value_and_grad(
                fused_loss_fn, argnums=(0, 1))(d16_, p16_, arrays_, sv_, key_)
            return parts, dgrads, ggrads
        joint_step_c = mx.compile(_joint_step)
    else:
        joint_step_c = None
    if args.compile_mode and args.compile_mode != "none" and not args.fixed_shape:
        print("[compile] WARNING: compile without --fixed-shape re-specializes "
              "on every distinct width (expect slowness)")
    logf = open(os.path.join(args.out, "loss.jsonl"), "a")
    fp_log = open(os.path.join(args.out, "footprint_steps.jsonl"), "a",
                  buffering=1)
    step, epoch = 0, 1
    key = mx.random.key(args.seed)
    t0 = time.perf_counter()
    stop = False
    first_rec = None

    def grad_norm_sq(gr):
        return sum(float(mx.sum(g_.astype(mx.float32) ** 2)) for g_ in gr.values())

    def _step_mem(step_i, extra):
        try:
            fp = __import__("gsovits_mlx.train.loop", fromlist=["x"]) \
                .FootprintSampler.read_phys_footprint(os.getpid())
        except Exception:
            fp = 0
        try:
            act = int(mx.get_active_memory())
            cache = int(mx.get_cache_memory())
        except Exception:
            try:
                act = int(mx.metal.get_active_memory())
                cache = int(mx.metal.get_cache_memory())
            except Exception:
                act = cache = 0
        rec = {"step": step_i, "footprint": fp, "metal_active": act,
               "metal_cache": cache, "metal_active_plus_cache": act + cache}
        rec.update(extra)
        return rec

    while not stop:
        sampler.set_epoch(epoch)
        batches = [collate([dataset[i] for i in b]) for b in iter(sampler)]
        for bi, batch in enumerate(batches):
            if step >= args.steps:
                stop = True
                break
            key, kd, kg = mx.random.split(key, 3)
            arrays, sv = prep_batch(batch)

            masters = {}
            for g_ in optim_g.param_groups:
                masters.update(g_["params"])
            p16 = make_compute(masters)
            d_masters = dict(optim_d.param_groups[0]["params"])
            if args.precision == "fp32":
                d16 = dict(d_masters)
            else:
                d16 = {k: v.astype(compute_dtype) for k, v in d_masters.items()}

            # ---- lr warmup (smoke aid; official has none) ----
            if args.warmup_steps > 0:
                w = min(1.0, (step + 1) / args.warmup_steps)
                for g_ in optim_g.param_groups:
                    g_["lr"] = g_["base_lr"] * w
                optim_d.param_groups[0]["lr"] = \
                    optim_d.param_groups[0]["base_lr"] * w

            ts = time.perf_counter()
            if args.g_forward == "single":
                # ---- fused single-trace step, evaluated in TWO phases ----
                # One trace = one G forward; the D-branch and G-branch
                # outputs are separate lazy subgraphs. Evaluating them in
                # ONE mx.eval would materialize both backwards
                # simultaneously (measured 8.81GB phys at b3 fp32, over the
                # 8GB gate). Phase-split: eval D branch -> step D -> free ->
                # eval G branch (reuses the SAME G-forward primals — the
                # compute win stays, the peak matches double mode).
                if joint_step_c is not None:
                    parts, dgrads, ggrads = joint_step_c(
                        d16, p16, *arrays, sv, kd)
                else:
                    (total, parts), (dgrads, ggrads) = mx.value_and_grad(
                        fused_loss_fn, argnums=(0, 1))(d16, p16, arrays, sv, kd)
                # fp16 keeps the legacy GradScaler machinery; bf16 bypasses
                # (CFM precedent — no scaler needed, fp32 grads/heads).
                if args.precision == "fp16":
                    scale = scaler.get_scale()
                    dgrads = {k: v * scale for k, v in dgrads.items()}
                    ggrads = {k: v * scale for k, v in ggrads.items()}
                # -- phase 1: D branch only (official D-step-first order) --
                mx.eval(parts[0], *dgrads.values())
                optim_d.param_groups[0]["grads"] = {
                    k: v.astype(mx.float32) for k, v in dgrads.items()}
                if args.precision == "fp16":
                    scaler.unscale_(optim_d)
                gn_d = grad_norm_sq(optim_d.param_groups[0]["grads"]) ** 0.5
                if args.precision == "fp16":
                    scaler.step(optim_d)
                else:
                    optim_d.step()
                try:
                    mx.clear_cache()
                except Exception:
                    pass
                # -- phase 2: G branch (same G-forward primals, D as consts) --
                mx.eval(*parts[1:], *ggrads.values())
                for g_ in optim_g.param_groups:
                    g_["grads"] = {k: ggrads[k].astype(mx.float32)
                                   for k in g_["params"] if k in ggrads}
                if args.precision == "fp16":
                    scaler.unscale_(optim_g)
                gn_g = sum(grad_norm_sq(g_["grads"])
                           for g_ in optim_g.param_groups) ** 0.5
                if args.precision == "fp16":
                    scaler.step(optim_g)
                    scaler.update()
                else:
                    optim_g.step()
                (loss_d, loss_gen, loss_fm, loss_mel, kl_ssl,
                 loss_kl) = [float(x) for x in parts]
            else:
                # ---- legacy double-forward step ----
                key_d = kd
                key_g = kd if not args.double_split_draw else kg
                (loss_d, dgrads) = mx.value_and_grad(d_loss_fn)(
                    d16, p16, arrays, sv, key_d)
                if args.precision == "fp16":
                    dgrads = {k: v * scaler.get_scale() for k, v in dgrads.items()}
                mx.eval(loss_d, *dgrads.values())
                optim_d.param_groups[0]["grads"] = {
                    k: v.astype(mx.float32) for k, v in dgrads.items()}
                if args.precision == "fp16":
                    scaler.unscale_(optim_d)
                gn_d = grad_norm_sq(optim_d.param_groups[0]["grads"]) ** 0.5
                if args.precision == "fp16":
                    scaler.step(optim_d)
                else:
                    optim_d.step()

                (loss_g, gparts), ggrads = mx.value_and_grad(g_loss_fn)(
                    p16, d16, arrays, sv, key_g)
                if args.precision == "fp16":
                    ggrads = {k: v * scaler.get_scale() for k, v in ggrads.items()}
                mx.eval(loss_g, *ggrads.values())
                for g_ in optim_g.param_groups:
                    g_["grads"] = {k: ggrads[k].astype(mx.float32)
                                   for k in g_["params"] if k in ggrads}
                if args.precision == "fp16":
                    scaler.unscale_(optim_g)
                gn_g = sum(grad_norm_sq(g_["grads"])
                           for g_ in optim_g.param_groups) ** 0.5
                if args.precision == "fp16":
                    scaler.step(optim_g)
                    scaler.update()
                else:
                    optim_g.step()
                loss_gen, loss_fm, loss_mel, kl_ssl, loss_kl = [
                    float(x) for x in gparts]
                loss_d = float(loss_d)
            t_step = time.perf_counter() - ts

            # ---- logging ----
            if step % args.log_interval == 0 or step == args.steps - 1:
                rec = dict(step=step, epoch=epoch, batch=bi,
                           loss_disc=loss_d,
                           loss_gen=loss_gen, loss_fm=loss_fm,
                           loss_mel=loss_mel, loss_kl_ssl=kl_ssl,
                           loss_kl=loss_kl,
                           grad_norm_d=round(gn_d, 3) if gn_d == gn_d else None,
                           grad_norm_g=round(gn_g, 3) if gn_g == gn_g else None,
                           scale=scaler.get_scale() if args.precision == "fp16" else 1.0,
                           lr=optim_g.param_groups[0]["lr"],
                           elapsed=round(time.perf_counter() - t0, 1))
                print(json.dumps(rec))
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if first_rec is None:
                    first_rec = rec
            fp_log.write(json.dumps(_step_mem(
                step, {"t_step": round(t_step, 3),
                       "loss_disc": loss_d, "loss_mel": loss_mel})) + "\n")
            if step == 1:
                # silent no-train regression net: compare the JUST-FINISHED
                # step-1 losses against the logged step-0 record
                assert (loss_d != first_rec["loss_disc"]
                        or loss_mel != first_rec["loss_mel"]), \
                    "silent no-train: step-1 losses identical to step-0 " \
                    "(optimizer update path dead?)"
            step += 1
            if step % 40 == 0 and _lock_held:
                from gsovits_mlx.gpu_lock import refresh_lock
                refresh_lock()
            try:
                mx.clear_cache()
            except Exception:
                pass
            from gsovits_mlx.train.loop import FootprintSampler
            _fp = FootprintSampler.read_phys_footprint(os.getpid())
            if _fp > 8 * 1024 * 1024 * 1024:
                raise SystemExit(
                    f"[abort] step {step} footprint {_fp/1e9:.2f} GB > 8GB "
                    "gate — aborting before machine risk")
            if args.save_every and step % args.save_every == 0:
                _save(ckpt_mod, args, optim_g, optim_d, step, epoch, d_fresh)
        epoch += 1
        sched_g.step()
        sched_d.step()
        if args.epochs is not None and epoch - 1 >= args.epochs:
            stop = True
        if not batches:
            stop = True

    logf.close()
    fp_log.close()
    _save(ckpt_mod, args, optim_g, optim_d, step, epoch - 1, d_fresh)

    peak = 0
    try:
        if device == "gpu":
            peak = int(mx.metal.get_peak_memory())
    except Exception:
        pass
    summary = dict(steps=step, epochs=epoch - 1, peak_metal_bytes=peak,
                   elapsed_s=round(time.perf_counter() - t0, 1),
                   disc_init="fresh" if d_fresh else "pretrained",
                   g_forward=args.g_forward, precision=args.precision,
                   device=device)
    print("[done]", json.dumps(summary))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    if args.export_inference:
        from tools.train_s2_export import export_inference
        masters = {}
        for g_ in optim_g.param_groups:
            masters.update(g_["params"])
        out_dir = export_inference(net_g, masters, params32, hps, args.version,
                                   os.path.join(args.out, "sovits_export"))
        print(f"[export] {out_dir}")
    return 0


def _save(ckpt_mod, args, optim_g, optim_d, step, epoch, d_fresh):
    masters = {}
    for g_ in optim_g.param_groups:
        masters.update(g_["params"])
    ckpt_mod.save_resume(
        os.path.join(args.out, "resume"), masters, [optim_g, optim_d],
        step=step, epoch=epoch,
        extra={"version": args.version, "disc_fresh": d_fresh})


if __name__ == "__main__":
    sys.exit(main())
