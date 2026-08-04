#include "swiglu_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

#include <cstdint>

namespace {

template <typename T>
__global__ void swigluKernel(
    T *out,
    const T *gate,
    const T *up,
    size_t numel) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for(size_t i = idx; i < numel; i += stride) {
        const float gate_val = llaisys::ops::nvidia::loadAsFloat(gate[i]);
        const float up_val = llaisys::ops::nvidia::loadAsFloat(up[i]);
        const float sigmoid_gate = 1.0F / (1.0F + expf(-gate_val));
        out[i] = llaisys::ops::nvidia::storeFromFloat<T>(up_val * gate_val * sigmoid_gate);
    }
}

template <typename T>
void launchSwigluKernel(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel) {
    if (numel == 0) {
        return;
    }
    const int block_size = 256;
    const int grid_size = llaisys::ops::nvidia::gridSizeFor(numel, block_size);
    swigluKernel<<<grid_size, block_size>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch swigluKernel");
}

} // namespace

namespace llaisys::ops::nvidia {
void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSwigluKernel<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launchSwigluKernel<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launchSwigluKernel<uint16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
