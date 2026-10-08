"""BS=1 fp32-vector/fp16-weight GEMV using MLX's fp32 reduction order.

Derived from MLX 0.32.2 GEMVKernel (BM=4, BN=1, SM=1, SN=32, TM=4,
TN=4), https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/gemv.h

Copyright (c) 2023-2024 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import mlx.core as mx

_SOURCE = r"""
const uint lane = thread_index_in_simdgroup;
const uint simd = simdgroup_index_in_threadgroup;
const int row = int(threadgroup_position_in_grid.x) * BM * 4 + int(simd) * 4;
float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
for (int base = int(lane) * 4; base < K; base += 128) {
    float4 v = *((const device float4*)(vector + base));
    for (int m = 0; m < 4; ++m) {
        float4 w = float4(*((const device half4*)(matrix + (row + m) * K + base)));
        result[m] += w[0] * v[0];
        result[m] += w[1] * v[1];
        result[m] += w[2] * v[2];
        result[m] += w[3] * v[3];
    }
}
for (int m = 0; m < 4; ++m) {
    for (ushort s = 16; s >= 1; s >>= 1) {
        result[m] += simd_shuffle_down(result[m], s);
    }
}
if (lane == 0) {
    for (int m = 0; m < 4; ++m) {
        float value = result[m] + float(bias[row + m]);
        if (ADD_RESIDUAL) value += residual[row + m];
        output[row + m] = RELU ? metal::max(value, 0.0f) : value;
    }
}
"""
_KERNEL = None


def supports(x, w, b, residual=None):
    """Match the stock fp32 GEMV row tile; exclude prefill and tail shapes."""
    return (mx.default_device() == mx.gpu and x.dtype == mx.float32
            and w.dtype == mx.float16 and w.ndim == 2
            and x.size == w.shape[1] and b is not None and b.size == w.shape[0]
            and 16 <= w.shape[0] < 4096 and w.shape[0] % 16 == 0
            and w.shape[1] > 64 and w.shape[1] % 128 == 0
            and w.shape[1] < 16 * w.shape[0]
            and (residual is None or (residual.dtype == mx.float32
                                     and residual.size == w.shape[0])))


def mixed_gemv(x, w, b, *, relu=False, residual=None):
    if not supports(x, w, b, residual):
        raise ValueError("Mixed GEMV requires a supported GPU single-vector shape")
    return _launch(x, w, b, relu=relu, residual=residual)


def _launch(x, w, b, *, relu=False, residual=None, bm=4):
    """Internal hot path; caller has already checked the geometry and device."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="gsovits_mixed_gemv", input_names=["vector", "matrix", "bias", "residual"],
            output_names=["output"], source=_SOURCE, compile_options={"math_mode": "safe"})
    n, k = w.shape
    return _KERNEL(
        inputs=[x, w, b, residual if residual is not None else x],
        template=[("K", k), ("BM", bm), ("RELU", relu), ("ADD_RESIDUAL", residual is not None)],
        grid=(n // (4 * bm) * 32, 1, bm), threadgroup=(32, 1, bm),
        output_shapes=[x.shape[:-1] + (n,)], output_dtypes=[mx.float32],
    )[0]
