#pragma once

// GFX906 optimized MXFP4 dequantization for vec_dot / MMQ.
// Uses __builtin_amdgcn_perm for an 8-entry table lookup instead of the
// generic table-based path.  This avoids extra ALU instructions on gfx906
// where v_perm_b32 is available.

#include "../gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(__gfx906__) && defined(GFX906_VEC_DOT_MXFP4_ENABLED)

// 8-entry magnitude table for MXFP4 (E2M1 with the zero value).
// Stored in __constant__ memory so it can be accessed via v_perm_b32.
__constant__ uint8_t gfx906_mxfp4_magnitudes[8] = { 0, 1, 2, 3, 4, 6, 8, 12 };

// Fast unaligned 4-byte load.  The compiler optimizes the memcpy to a
// flat_load_dword which handles misaligned addresses efficiently on gfx906.
static __device__ __forceinline__ int gfx906_get_int_b1_fast(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;
    int x32;
    memcpy(&x32, x8 + 4*i32, 4);
    return x32;
}

// Dequantize one 32-bit MXFP4 nibble-pair block using v_perm_b32.
// Input: q4 contains 8 MXFP4 values (4 bits each).
// Output: int2 with the dequantized int8 values in .x and .y.
static __device__ __forceinline__ int2 gfx906_get_int_from_mxfp4_table(const uint32_t q4) {
    const uint32_t * mags32 = (const uint32_t *)gfx906_mxfp4_magnitudes;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = q4 >> 4;

    const uint32_t sign_even = (q_even >> 3) & 0x01010101;
    const uint32_t sign_odd  = (q_odd  >> 3) & 0x01010101;

    const uint32_t sel_even = q_even & 0x07070707;
    const uint32_t sel_odd  = q_odd  & 0x07070707;

    // Use the hardware permute to look up the magnitude for each nibble.
    const uint32_t mag_even = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_even);
    const uint32_t mag_odd  = __builtin_amdgcn_perm(mags32[1], mags32[0], sel_odd);

    const uint32_t mask_even = sign_even * 0xFFu;
    const uint32_t mask_odd  = sign_odd  * 0xFFu;

    const uint32_t res_x = (mag_even ^ mask_even) + sign_even;
    const uint32_t res_y = (mag_odd  ^ mask_odd)  + sign_odd;

    return make_int2((int)res_x, (int)res_y);
}

// Optimized vec_dot for MXFP4 x Q8_1 on gfx906.
// This mirrors vec_dot_mxfp4_q8_1 from ggml/src/ggml-cuda/vecdotq.cuh but
// uses the hardware permute path for MXFP4 dequantization.
#define GFX906_VEC_DOT_MXFP4_Q8_1(bq4, bq8_1, iqs, sumi) \
    do { \
        const int * q8 = (const int *) (bq8_1)->qs + (iqs); \
        const int aux_q4_0 = gfx906_get_int_b1_fast((bq4)->qs, (iqs) + 0); \
        const int aux_q4_1 = gfx906_get_int_b1_fast((bq4)->qs, (iqs) + 1); \
        const int2 v0 = gfx906_get_int_from_mxfp4_table((uint32_t)aux_q4_0); \
        const int2 v1 = gfx906_get_int_from_mxfp4_table((uint32_t)aux_q4_1); \
        (sumi) = ggml_cuda_dp4a(v0.x, q8[0], (sumi)); \
        (sumi) = ggml_cuda_dp4a(v0.y, q8[4], (sumi)); \
        (sumi) = ggml_cuda_dp4a(v1.x, q8[1], (sumi)); \
        (sumi) = ggml_cuda_dp4a(v1.y, q8[5], (sumi)); \
    } while(0)

#endif // defined(GGML_USE_HIP) && defined(__gfx906__) && defined(GFX906_VEC_DOT_MXFP4_ENABLED)
