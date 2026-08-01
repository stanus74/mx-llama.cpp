#pragma once

// Common gfx906-specific helpers.
// This file is intentionally minimal for now and will be extended as more
// kernels from the skyne98 gfx906 fork are ported.

#include "gfx906-config.h"

#if defined(GGML_USE_HIP) && defined(__gfx906__)

// Placeholder for future DPP-based reductions and fast math intrinsics.
// Currently the only ported kernel (vecdotq) does not require additional
// helpers beyond the ones already provided by ggml-cuda/common.cuh.

#endif // defined(GGML_USE_HIP) && defined(__gfx906__)
