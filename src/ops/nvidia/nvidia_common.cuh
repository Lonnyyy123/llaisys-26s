#pragma once

#include "../../device/nvidia/nvidia_compat.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>

namespace llaisys::ops::nvidia {

inline void checkCuda(cudaError_t err, const char *message) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA op error: " << message << ": "
                  << cudaGetErrorString(err) << EXCEPTION_LOCATION_MSG
                  << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

inline int gridSizeFor(size_t numel, int block_size) {
    int device = 0;
    cudaDeviceProp prop{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    const size_t needed_blocks =
        (numel + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    const size_t occupancy_blocks = static_cast<size_t>(prop.multiProcessorCount) * 8;
    const size_t max_grid_x = static_cast<size_t>(prop.maxGridSize[0]);
    const size_t capped_blocks = std::min({needed_blocks, occupancy_blocks, max_grid_x});
    return static_cast<int>(std::max<size_t>(capped_blocks, 1));
}

__device__ inline float bf16ToFloat(uint16_t value) {
    union {
        uint32_t u32;
        float f32;
    } bits{};
    bits.u32 = static_cast<uint32_t>(value) << 16;
    return bits.f32;
}

__device__ inline uint16_t floatToBf16(float value) {
    union {
        float f32;
        uint32_t u32;
    } bits{};
    bits.f32 = value;
    const uint32_t rounding_bias = 0x00007FFFu + ((bits.u32 >> 16) & 1u);
    return static_cast<uint16_t>((bits.u32 + rounding_bias) >> 16);
}

template <typename T>
__device__ inline float loadAsFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float loadAsFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float loadAsFloat<uint16_t>(uint16_t value) {
    return bf16ToFloat(value);
}

template <typename T>
__device__ inline T storeFromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline __half storeFromFloat<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ inline uint16_t storeFromFloat<uint16_t>(float value) {
    return floatToBf16(value);
}

} // namespace llaisys::ops::nvidia
