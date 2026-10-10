"""Fused Metal kernels for the s2 GAN trainer (mx.fast.metal_kernel).

STEP-1 profile census (bc_fix_d, v2 b3, 5 steps, CPU-side dispatch count,
.py instrumentation; binary `+`/`*` operator dispatches NOT countable from
Python and are folded in analytically at the site level):

  per step (counts/5):  transpose 1682 | sum 1367 | sqrt 1359 | pad 592 |
  conv1d 540 | conv2d 120 | zeros_like 370 | leaky_relu 316 | maximum 346 |
  tanh 70 + sigmoid 68 (WN gating) | mean 56 | astype ~large (uncountable) |

Ranked fusable patterns (dispatch + bytes moved):

  #1 bias+LeakyReLU tails — every D conv (S:6+1, P:5+1 x5 periods) and the
     HiFi-GAN dec resblock chain. Conv output is (B,T,C) there, so the
     chain is transpose + astype + add(bias) + leaky_relu = 4 dispatches,
     4 full-tensor memory passes (activations up to (3,1024,20480) fp32 =
     251 MB) -> ONE elementwise kernel, 1 read + 1 write. Lead-validated
     4.4x on the 6144x2048 shape, bit-exact.

  #2 weight-norm effective weight w = g*v/||v|| (per-out-channel): every
     WN conv in enc_q (17) + flow (4x7) + HiFi-GAN dec (34) + every D conv
     (S 7 + P 36) ~= ~120 convs per forward; the chain is astype+mul+mul+
     sum+sqrt+div (+astype) ~= 6 dispatches on the WEIGHT tensors, recomputed
     in forward AND backward (mx.grad re-runs it) -> ONE kernel per conv.
     Weight tensors are small (384x5x192=369k elem max) so this is a
     dispatch-count kill, not a bandwidth kill.

  #3 WN gating tanh*sigmoid (fused_add_tanh_sigmoid_multiply): in_act =
     a+b then tanh(half)*sigmoid(half) over (B,2H,T) fp32 at the trunk
     width (960): 340 tanh + 328 sigmoid + ~700 binary-op dispatches/step
     across enc_q/flow; chain = add + tanh + sigmoid + multiply = 4
     dispatches, 4 passes over big activations -> ONE elementwise kernel.

Numerics policy per mission: fp32 bit-exact vs the replaced MLX op chain
where the op ORDER is preserved (lrelu chain); documented <=1e-6 rel where
the GPU reduction order cannot match MLX's tree sum (weight-norm row sum
for R>~1k: measured max rel 2.0e-6 on dec.ups k16; documented in tests).
All three primals run under mx.custom_function with Metal/MLX vjps so
mx.grad flows (a raw metal_kernel call is opaque to the autodiff graph).

Kernel law (lead's lesson, mlx #4534): grid= is TOTAL THREAD COUNT. Use
grid=(N,1,1), threadgroup=(256,1,1), guard elem<N. Every kernel is
probed with a full-coverage write test AND parity vs the op chain before
any benchmark (tests/test_metal_kernels.py).

mx 0.32.2 metal_kernel conventions (probed, see tests):
  - source is the function BODY only; the wrapper emits the kernel shell
    with one `const constant T* <input_name> [[buffer(i)]]` per input and
    `device T* <output_name> [[buffer(j)]]` per output, plus
    uint3 thread_position_in_grid. Use thread_position_in_grid.x.
  - inputs are flattened ROW-MAJOR contiguous views (ensure_row_contiguous
    default); a (B,C,T) tensor's element e maps to channel c = (e // T) % C.
  - template=[("T", dtype)] instantiates one typename for ALL T-typed
    params; template names must not collide with input names.
  - scalars: pass python ints/floats in inputs= (they specialize the
    kernel hash) or bake via string substitution (compile-time constant).
  - mx.custom_function vjp args convention (probed): for a 1-input primal,
    vjp(x, cot, out); for >1 inputs, vjp((x1, x2, ...), cot, out).
"""
from __future__ import annotations

import mlx.core as mx

__all__ = [
    "available",
    "fused_bias_lrelu",
    "fused_wn_scale",
    "fused_gate",
    "USE_METAL_KERNELS",
]

USE_METAL_KERNELS = False  # set by tools/train_s2.py --metal-kernels


def available() -> bool:
    """True when custom Metal kernels can run (GPU + mx.fast.metal_kernel)."""
    return (hasattr(mx.fast, "metal_kernel")
            and mx.metal.is_available())


# ---------------------------------------------------------------------------
# kernel sources (bodies; wrapped by mx.fast.metal_kernel shells)
# ---------------------------------------------------------------------------

_LRELU_BODY = """
uint e = thread_position_in_grid.x;
if (e >= (uint)@NEL@) return;
T v = x_[e] + b_[(e / (uint)@T_) % (uint)@C@];
y_[e] = v < T(0) ? T(@SLOPE@) * v : v;
"""

# per-out-channel row sum-of-squares + scale; one thread per output channel.
# R = product of the norm axes (row-major contiguous rows).
_WN_BODY = """
uint oc = thread_position_in_grid.x;
if (oc >= (uint)@OUT@) return;
auto vrow = v_ + (size_t)oc * (size_t)@R@;
float acc = 0.0f;
for (size_t i = 0; i < (size_t)@R@; ++i) {
    float v = static_cast<float>(vrow[i]);
    acc += v * v;
}
float inv = 1.0f / metal::precise::sqrt(acc);
auto wrow = w_ + (size_t)oc * (size_t)@R@;
float g = static_cast<float>(g_[oc]);
for (size_t i = 0; i < (size_t)@R@; ++i) {
    wrow[i] = static_cast<T>(g * static_cast<float>(vrow[i]) * inv);
}
"""

# in = a + b over (B, 2H, T); y[:H] halves tanh, [H:] sigmoid, multiplied.
_GATE_BODY = """
uint e = thread_position_in_grid.x;
if (e >= (uint)@NEL@) return;      // NEL = B*H*T (halves only)
uint h_len = @H@;
uint t_len = @T@;
uint idx = e % (h_len * t_len);     // position within (H, T)
uint h = idx / t_len;               // channel within half
uint bt = e / (h_len * t_len);      // batch*T block
uint base_a = bt * (2 * h_len * t_len) + h * t_len + (idx % t_len);
uint base_b = base_a + h_len * t_len;
float u = static_cast<float>(a_[base_a]) + static_cast<float>(b_[base_a]);
float u2 = static_cast<float>(a_[base_b]) + static_cast<float>(b_[base_b]);
float t_v = metal::precise::tanh(u);
float s = 1.0f / (1.0f + metal::fast::exp(-u2));
y_[e] = static_cast<T>(t_v * s);
"""

_KERNEL_CACHE: dict[tuple, object] = {}


def _get_kernel(name: str, body: str):
    key = (name, body)
    k = _KERNEL_CACHE.get(key)
    if k is None:
        k = mx.fast.metal_kernel(name=name, source=body,
                                 input_names=_INPUT_NAMES[name],
                                 output_names=_OUTPUT_NAMES[name])
        _KERNEL_CACHE[key] = k
    return k


_INPUT_NAMES = {
    "bias_lrelu": ["x_", "b_"],
    "wn_scale": ["g_", "v_"],
    "gate": ["a_", "b_"],
}
_OUTPUT_NAMES = {
    "bias_lrelu": ["y_"],
    "wn_scale": ["w_"],
    "gate": ["y_"],
}


# ---------------------------------------------------------------------------
# 1. fused bias + LeakyReLU
# ---------------------------------------------------------------------------

def _lrelu_body(nel: int, t_len: int, c: int, slope: str, dtype: mx.Dtype):
    src = (_LRELU_BODY.replace("@NEL@", str(nel))
           .replace("@T_", str(t_len)).replace("@C@", str(c))
           .replace("@SLOPE@", slope))
    return src


@mx.custom_function
def fused_bias_lrelu(x: mx.array, bias: mx.array, slope: float = 0.1):
    """y = leaky_relu(x + bias) on (B, C, T); bit-exact vs the op chain.

    Replaces transpose+astype+add+leaky_relu (4 dispatches, 4 passes) with
    one elementwise kernel (1 read + 1 write). x must be row-contiguous
    (any (B,C,T); ensured by ensure_row_contiguous on the kernel inputs).
    """
    nel = x.size
    b, c, t = x.shape
    body = _lrelu_body(nel, t, c, repr(float(slope)), x.dtype)
    k = _get_kernel("bias_lrelu", body)
    return k(inputs=[x, bias], template=[("T", x.dtype)],
             grid=(nel, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[x.shape], output_dtypes=[x.dtype])[0]


@fused_bias_lrelu.vjp
def _bias_lrelu_vjp(*args):
    # convention (probed on 0.32.2): args = (primals_tuple, cot, out) for
    # multi-input primals; a python-float primal (slope) is dropped.
    primals, cot = args[0], args[-2]
    x, bias = primals[0], primals[1]
    slope = _BIAS_LRELU_SLOPE[0]
    # d/dx leaky_relu(x+b) = slope if x+b<0 else 1; bit-exact vs chain.
    mask = (x + bias[None, :, None]) < 0
    grad_in = mx.where(mask, slope, 1.0).astype(cot.dtype) * cot
    bgrad = mx.sum(grad_in, axis=(0, 2))
    return (grad_in, bgrad)


_BIAS_LRELU_SLOPE = [0.1]  # set per call (single-threaded trainer loop)


def bias_lrelu(x: mx.array, bias: mx.array, slope: float = 0.1) -> mx.array:
    """Public entry: fused_bias_lrelu with the slope threaded through.

    The Metal VJP is elementwise (d/dx = slope or 1), so the analytic vjp
    needs the slope constant; custom_function drops python-float primals
    from the vjp signature, so we stash it in a module-level slot. The
    trainer is single-threaded (GPU law: one process) — safe.
    """
    _BIAS_LRELU_SLOPE[0] = slope
    return fused_bias_lrelu(x, bias, slope)


# ---------------------------------------------------------------------------
# 2. fused weight-norm effective weight
# ---------------------------------------------------------------------------

@mx.custom_function
def fused_wn_scale(g: mx.array, v: mx.array):
    """w = g * v / ||v||_2 per-out-channel; v is (OUT, ...) row-contiguous.

    Replaces astype+mul+mul+sum+sqrt+div (+astype) = ~6 dispatches per WN
    conv per graph. fp32: max rel <= ~2e-6 vs the MLX chain (GPU serial
    row-sum vs MLX tree reduction order; documented, in-tolerance). fp16/
    bf16 compute: accumulate fp32, cast once at the write (BETTER than the
    chain, which accumulates in the input dtype after astype).
    """
    out = v.shape[0]
    r = v.size // out
    body = _WN_BODY.replace("@OUT@", str(out)).replace("@R@", str(r))
    k = _get_kernel("wn_scale", body)
    return k(inputs=[g, v], template=[("T", v.dtype)],
             grid=(out, 1, 1), threadgroup=(32, 1, 1),
             output_shapes=[v.shape], output_dtypes=[v.dtype])[0]


@fused_wn_scale.vjp
def _wn_vjp(*args):
    primals, cot, out = (args[0] if isinstance(args[0], tuple) else (args[0],), args[-2], args[-1])
    g, v = primals
    axes = tuple(range(1, v.ndim))
    v32 = v.astype(mx.float32)
    s = mx.sum(v32 * v32, axis=axes, keepdims=True)   # ||v||^2
    n = mx.sqrt(s)                                    # ||v||
    gw = cot.astype(mx.float32)
    g32 = g.astype(mx.float32).reshape(-1, *([1] * (v.ndim - 1)))
    # w = g * v / n  per out-channel:
    #   dL/dg_c = sum_r gw * v / n
    #   dL/dv    = g/n * (gw - v * (sum_r gw*v) / n^2)
    grad_g = mx.sum(gw * v32 / n, axis=axes)
    proj = mx.sum(v32 * gw, axis=axes, keepdims=True) / s
    grad_v = g32 / n * (gw - proj * v32)
    return (grad_g.astype(g.dtype), grad_v.astype(v.dtype))


# ---------------------------------------------------------------------------
# 3. fused WN gating: tanh(a+b) * sigmoid(a+b) over halves
# ---------------------------------------------------------------------------

@mx.custom_function
def fused_gate(a: mx.array, b: mx.array):
    """y = tanh((a+b)[:H]) * sigmoid((a+b)[H:]) on (B, 2H, T) halves.

    Replaces fused_add_tanh_sigmoid_multiply's add+tanh+sigmoid+mul chain
    (4 dispatches, 4 full passes) with one kernel; halves computed in fp32
    (chain: activations fp32; matches mx.tanh/mx.sigmoid fp32 math to
    <=1e-6 rel — precise::tanh + fast::exp vs MLX's implementations).
    """
    blks, two_h, t = a.shape
    h = two_h // 2
    nel = blks * h * t
    body = (_GATE_BODY.replace("@NEL@", str(nel))
            .replace("@H@", str(h)).replace("@T@", str(t)))
    k = _get_kernel("gate", body)
    return k(inputs=[a, b], template=[("T", a.dtype)],
             grid=(nel, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(blks, h, t)], output_dtypes=[a.dtype])[0]


@fused_gate.vjp
def _gate_vjp(*args):
    primals, cot = args[0], args[-2]
    a, b = primals
    h = a.shape[1] // 2
    v = a + b
    t_act = mx.tanh(v[:, :h, :])
    s_act = mx.sigmoid(v[:, h:, :])
    # dy/du for the tanh half: (1 - t^2) * s ; for the sigmoid half: t * s * (1-s)
    du1 = (1.0 - t_act * t_act) * s_act
    du2 = t_act * (s_act * (1.0 - s_act))
    ga = mx.concatenate([cot * du1, cot * du2], axis=1)
    return (ga, ga)
