#include "add_nvidia.cuh"

#include "../../nvidia/nvidia_common.cuh"

#include <cstdint>

namespace {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < numel;
         i += stride) {
        c[i] = llaisys::ops::nvidia::storeFromFloat<T>(
            llaisys::ops::nvidia::loadAsFloat(a[i])
            + llaisys::ops::nvidia::loadAsFloat(b[i]));
    }
}

template <typename T>
void launchAddKernel(T *c, const T *a, const T *b, size_t numel) {
    if (numel == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = llaisys::ops::nvidia::gridSizeFor(numel, block_size);
    addKernel<<<grid_size, block_size>>>(c, a, b, numel);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch addKernel");
}

} // namespace

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchAddKernel(
            reinterpret_cast<float *>(c),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            numel);
    case LLAISYS_DTYPE_F16:
        return launchAddKernel(
            reinterpret_cast<__half *>(c),
            reinterpret_cast<const __half *>(a),
            reinterpret_cast<const __half *>(b),
            numel);
    case LLAISYS_DTYPE_BF16:
        return launchAddKernel(
            reinterpret_cast<uint16_t *>(c),
            reinterpret_cast<const uint16_t *>(a),
            reinterpret_cast<const uint16_t *>(b),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
