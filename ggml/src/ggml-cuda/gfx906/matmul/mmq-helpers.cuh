#pragma once

// gfx906-specific helpers for the legacy MMQ path used by mx-llama.cpp.
// These are intentionally small, surgical optimizations that can be toggled
// without replacing the entire MMQ subsystem.

#include "../gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// Load eight int32 quant values from two int arrays using 128-bit vector loads.
// Used in the legacy Q4_0/Q4_1 MMQ DP4A path.  On gfx906 this turns into
// global_load_dwordx4 / flat_load_dwordx4 when the base addresses are aligned.
//
// Layout after the call:
//   u[0..7] = { vec0.x, vec1.x, vec0.y, vec1.y, vec0.z, vec1.z, vec0.w, vec1.w }
static __device__ __forceinline__ void gfx906_load_q4_quants_vectorized_8(
        const int * __restrict__ src,
        const int base_addr0,
        const int base_addr1,
        int * __restrict__ u) {
    // Only use the vectorized path when both addresses are 16-byte aligned.
    const bool aligned = ((base_addr0 & 3) == 0) && ((base_addr1 & 3) == 0);
    if (aligned) {
        const int4 vec0 = *((const int4 *) &src[base_addr0]);
        const int4 vec1 = *((const int4 *) &src[base_addr1]);
        u[0] = vec0.x; u[1] = vec1.x;
        u[2] = vec0.y; u[3] = vec1.y;
        u[4] = vec0.z; u[5] = vec1.z;
        u[6] = vec0.w; u[7] = vec1.w;
    } else {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            u[2*l + 0] = src[base_addr0 + l];
            u[2*l + 1] = src[base_addr1 + l];
        }
    }
}

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
