#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration
// This header is included by the gfx906-specific HIP kernels under
// ggml/src/ggml-cuda/gfx906/.

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// MMQ nwarps are tuned in ggml/src/ggml-cuda/mmq.cuh via
// GGML_MMQ_NWARPS_GFX906_Q8 / GGML_MMQ_NWARPS_GFX906_OTHER.
// Do not add a duplicate constant here; it will silently diverge.

// ============================================
// Q8 Cache Configuration (fork-specific, currently disabled)
// ============================================
#ifndef GFX906_KVQ_MOE_CACHE_ENABLED
#define GFX906_KVQ_MOE_CACHE_ENABLED 0
#endif

// Layer-cycling: N cycles, slot size = TOTAL / N
#ifndef GFX906_Q8_CACHE_TOTAL_SIZE
#define GFX906_Q8_CACHE_TOTAL_SIZE      (128 * 1024 * 1024)  // Total cache size: 128MB
#endif
#ifndef GFX906_Q8_CACHE_NUM_SLOTS
#define GFX906_Q8_CACHE_NUM_SLOTS       1                    // Number of cycles
#endif
#ifndef GFX906_Q8_CACHE_LAYERS_PER_SLOT
#define GFX906_Q8_CACHE_LAYERS_PER_SLOT 1                    // 1 layer per slot
#endif

// ============================================
// ROPE Optimization
// ============================================
#ifndef GFX906_ROPE_ENABLED
#define GFX906_ROPE_ENABLED 1
#endif

// ============================================
// Q4_0/Q4_1 vectorized loads in legacy MMQ DP4A path.
// Can regress large-batch (pp512) Q4_0/Q4_1 perf on gfx906 due to register pressure;
// toggle off to test.
// ============================================
#ifndef GFX906_MMQ_VEC_LOAD_Q4_ENABLED
#define GFX906_MMQ_VEC_LOAD_Q4_ENABLED 1
#endif

// ============================================
// skyne98 warp-cooperative MMVQ kernels for Q4_0/Q4_1/Q8_0.
// Currently disabled by default: on Ornith-1.0-9B-Q4_0 they were slower than
// the existing MMVQ path (tg128 63.3 vs 65.6 t/s). Keep for further tuning.
// ============================================
#ifndef GFX906_MMVQ_WARP_COOP_ENABLED
#define GFX906_MMVQ_WARP_COOP_ENABLED 0
#endif

// ============================================
// MXFP4 vec_dot optimization using __builtin_amdgcn_perm
// ============================================
#ifndef GFX906_VEC_DOT_MXFP4_ENABLED
#define GFX906_VEC_DOT_MXFP4_ENABLED 1
#endif

// ============================================
// skyne98 custom FP16 GEMM for medium batch sizes on gfx906.
// Default enabled for benchmarking; set to 0 to fall back to hipBLAS.
// ============================================
#ifndef GFX906_MMF_ENABLED
#define GFX906_MMF_ENABLED 1
#endif

// ============================================
// skyne98 custom Q8_0 flash-attention tile kernel for gfx906.
// Currently disabled by default: on TinyLlama-1.1B-Q8_0 it was not faster
// than upstream tile/vec path (pp512/pp32768/tg128 all within noise).
// Keep for further tuning and large-context experiments.
// ============================================
#ifndef GFX906_FATTN_Q8_ENABLED
#define GFX906_FATTN_Q8_ENABLED 0
#endif

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
