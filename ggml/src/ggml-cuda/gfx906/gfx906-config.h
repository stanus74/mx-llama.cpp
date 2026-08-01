#pragma once

// GFX906 (Vega 20 / MI50) kernel configuration
// This header is included by the gfx906-specific HIP kernels under
// ggml/src/ggml-cuda/gfx906/.

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// ============================================
// MMQ Kernel Configuration
// ============================================
#ifndef GFX906_MMQ_NWARPS
#define GFX906_MMQ_NWARPS 2
#endif

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
// MXFP4 vec_dot optimization using __builtin_amdgcn_perm
// ============================================
#ifndef GFX906_VEC_DOT_MXFP4_ENABLED
#define GFX906_VEC_DOT_MXFP4_ENABLED 1
#endif

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
