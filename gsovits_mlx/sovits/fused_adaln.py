"""Stock fp32 LayerNorm plus DiT modulation with separate fp32 rounding.

Derived from MLX 0.32.2 layer_norm.metal, Copyright (c) 2024 Apple Inc.

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

import mlx.core as mx

_HEADER = r"""
#include <metal_common>
#include <metal_simdgroup>
using namespace metal;
inline void sum_group(thread float* x, threadgroup float* buffer, uint lane, uint group) {
    *x = simd_sum(*x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) buffer[group] = *x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *x = simd_sum(buffer[lane]);
}
#pragma clang fp contract(off)
template <typename T>
inline T modulate(float norm, float scale, float shift) {
    float factor = 1.0f + scale;
    float scaled = norm * factor;
    return static_cast<T>(scaled + shift);
}
#pragma clang fp contract(on)
"""
_SOURCE = r"""
uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
threadgroup float buffer[32];
if (group == 0) buffer[lane] = 0;
threadgroup_barrier(mem_flags::mem_threadgroup);
x += row * 1024 + lid * 8;
scale += lid * 8;
shift += lid * 8;
output += row * 1024 + lid * 8;
float values[8];
float mean = 0;
for (int i = 0; i < 8; ++i) {
    values[i] = x[i];
    mean += values[i];
}
sum_group(&mean, buffer, lane, group);
mean /= 1024;
float variance = 0;
for (int i = 0; i < 8; ++i) {
    values[i] -= mean;
    variance += values[i] * values[i];
}
sum_group(&variance, buffer, lane, group);
float normalizer = metal::precise::rsqrt(variance / 1024 + 1e-6);
for (int i = 0; i < 8; ++i) {
    values[i] *= normalizer;
    output[i] = modulate<T>(values[i], scale[i], shift[i]);
}
"""


@lru_cache(maxsize=2)
def _kernel(dtype):
    return mx.fast.metal_kernel(
        name="gsovits_modulated_ln", input_names=["x", "scale", "shift"],
        output_names=["output"], source=_SOURCE, header=_HEADER,
        ensure_row_contiguous=True, compile_options={"math_mode": "safe"})


def fused_modulated_norm(x, scale, shift, dtype):
    return _kernel(dtype)(inputs=[x, scale, shift], template=[("T", dtype)],
                          grid=(x.size // 1024 * 128, 1, 1), threadgroup=(128, 1, 1),
                          output_shapes=[x.shape], output_dtypes=[dtype])[0]
