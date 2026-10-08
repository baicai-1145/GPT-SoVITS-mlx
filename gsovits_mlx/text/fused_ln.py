"""Half-rounded dense bias + residual + stock MLX LayerNorm epilogue.

Derived from MLX 0.32.2 layer_norm.metal, Copyright (c) 2024 Apple Inc.
https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/layer_norm.metal

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

from functools import lru_cache
import os

import mlx.core as mx

_HEADER = r"""
#include <metal_common>
#include <metal_simdgroup>
using namespace metal;

inline void initialize_buffer(threadgroup float* xs, uint lane, uint group) {
    if (group == 0) xs[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline void threadgroup_sum(thread float* x, threadgroup float* xs,
                            uint lane, uint group) {
    *x = simd_sum(*x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) xs[group] = *x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *x = simd_sum(xs[lane]);
}
"""
_SOURCE = r"""
const uint gid = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint group = simdgroup_index_in_threadgroup;
constexpr int N_READS = 8;
constexpr uint AXIS = {D};
constexpr float EPS = {EPS};
float values[N_READS] = {0};
threadgroup float buffer[32];
initialize_buffer(buffer, lane, group);
mm += gid * size_t(AXIS) + lid * N_READS;
bias += lid * N_READS;
residual += gid * size_t(AXIS) + lid * N_READS;
weight += lid * N_READS;
beta += lid * N_READS;
output += gid * size_t(AXIS) + lid * N_READS;
for (int i = 0; i < N_READS; ++i) {
    half biased = half(mm[i] + bias[i]);
    values[i] = half(biased + residual[i]);
}
float mean = 0;
for (int i = 0; i < N_READS; ++i) mean += values[i];
threadgroup_sum(&mean, buffer, lane, group);
mean /= AXIS;
float variance = 0;
for (int i = 0; i < N_READS; ++i) {
    values[i] -= mean;
    variance += values[i] * values[i];
}
threadgroup_sum(&variance, buffer, lane, group);
const float normalizer = metal::precise::rsqrt(variance / AXIS + EPS);
for (int i = 0; i < N_READS; ++i) {
    values[i] *= normalizer;
    output[i] = weight[i] * half(values[i]) + beta[i];
}
"""


@lru_cache(maxsize=4)
def _kernel(dim, eps):
    source = _SOURCE.replace("{D}", str(dim)).replace("{EPS}", repr(eps))
    kernel = mx.fast.metal_kernel(
        name="gsovits_half_residual_ln", input_names=["mm", "bias", "residual", "weight", "beta"],
        output_names=["output"], source=source, header=_HEADER,
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"})
    return kernel, dim // 8


def dense_residual_norm(mm, bias, residual, weight, beta, eps):
    if (os.environ.get("GSOVITS_BERT_FUSED_LN", "0") == "1"
            and mx.default_device() == mx.gpu and mm.shape[-1] in (768, 1024)
            and all(a.dtype == mx.float16 for a in (mm, bias, residual, weight, beta))):
        dim = mm.shape[-1]
        kernel, tg = _kernel(dim, eps)
        return kernel(
            inputs=[mm, bias, residual, weight, beta],
            grid=(mm.size // dim * tg, 1, 1), threadgroup=(tg, 1, 1),
            output_shapes=[mm.shape], output_dtypes=[mx.float16],
        )[0]
    return mx.fast.layer_norm(mm + bias + residual, weight, beta, eps)
