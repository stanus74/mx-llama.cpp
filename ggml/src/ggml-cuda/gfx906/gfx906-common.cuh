#pragma once

// Common gfx906-specific helpers for skyne98-derived kernels.
// Host-side definitions are provided as fallbacks so the header can be parsed
// in host compilation passes; the ASM paths are only active during device
// compilation.

#include "gfx906-config.h"

#if defined(GGML_USE_HIP)

#include <cmath>

#if defined(__HIP_DEVICE_COMPILE__)

static __device__ __forceinline__ float sgpr_broadcast_f32(float value) {
    int i = __float_as_int(value);
    i = __builtin_amdgcn_readfirstlane(i);
    return __int_as_float(i);
}

static __device__ __forceinline__ int sgpr_broadcast_i32(int value) {
    return __builtin_amdgcn_readfirstlane(value);
}

static __device__ __forceinline__ half sgpr_broadcast_f16(half value) {
    int i = *reinterpret_cast<const short*>(&value);
    i = __builtin_amdgcn_readfirstlane(i);
    short s = static_cast<short>(i);
    return *reinterpret_cast<half*>(&s);
}

static __device__ __forceinline__ float fast_exp_f32(float x) {
    constexpr float LOG2_E = 1.4426950408889634f;
    float result;
    asm volatile(
        "v_exp_f32 %0, %1"
        : "=v"(result)
        : "v"(x * LOG2_E)
    );
    return result;
}

static __device__ __forceinline__ float fast_exp2_f32(float x) {
    float result;
    asm volatile(
        "v_exp_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

static __device__ __forceinline__ float fast_log2_f32(float x) {
    float result;
    asm volatile(
        "v_log_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

static __device__ __forceinline__ float fast_tanh_f32(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;

    const float exp2x = fast_exp_f32(2.0f * x);
    return 1.0f - 2.0f / (exp2x + 1.0f);
}

static __device__ __forceinline__ float fast_rcp_f32(float x) {
    float result;
    asm volatile(
        "v_rcp_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

#else // Host fallback

static __inline__ float sgpr_broadcast_f32(float value) {
    return value;
}

static __inline__ int sgpr_broadcast_i32(int value) {
    return value;
}

static __inline__ half sgpr_broadcast_f16(half value) {
    return value;
}

static __inline__ float fast_exp_f32(float x) {
    return std::exp(x);
}

static __inline__ float fast_exp2_f32(float x) {
    return std::exp2(x);
}

static __inline__ float fast_log2_f32(float x) {
    return std::log2(x);
}

static __inline__ float fast_tanh_f32(float x) {
    return std::tanh(x);
}

static __inline__ float fast_rcp_f32(float x) {
    return 1.0f / x;
}

#endif // __HIP_DEVICE_COMPILE__

#endif // defined(GGML_USE_HIP)
