#include "rms_norm_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

#include <cstdint>

namespace {

__device__ inline float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T>
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t cols,
    float eps) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    const size_t stride = static_cast<size_t>(blockDim.x);

    float local_sum = 0.0F;
    for (size_t col = threadIdx.x; col < cols; col += stride) {
        const float val = llaisys::ops::nvidia::loadAsFloat(in[row * cols + col]);
        local_sum += val * val;
    }

    local_sum = warpReduceSum(local_sum);

    __shared__ float warp_sums[8];
    __shared__ float inverse_rms;
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        float block_sum = lane < (blockDim.x + 31) / 32 ? warp_sums[lane] : 0.0F;
        block_sum = warpReduceSum(block_sum);
        if (lane == 0) {
            inverse_rms = rsqrtf(block_sum / static_cast<float>(cols) + eps);
        }
    }
    __syncthreads();

    for (size_t col = threadIdx.x; col < cols; col += stride) {
        const float val = llaisys::ops::nvidia::loadAsFloat(in[row * cols + col]);
        const float w = llaisys::ops::nvidia::loadAsFloat(weight[col]);

        out[row * cols + col] =
            llaisys::ops::nvidia::storeFromFloat<T>(val * w * inverse_rms);
    }
}

template <typename T>
void launchRmsNormKernel(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t cols,
    float eps) {
    if (rows == 0 || cols == 0) {
        return;
    }
    const int block_size = 256;
    const int grid_size = static_cast<int>(rows);
    rmsNormKernel<<<grid_size, block_size>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        rows,
        cols,
        eps);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch rmsNormKernel");
}

} // namespace

namespace llaisys::ops::nvidia {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t cols,
    float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNormKernel<float>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return launchRmsNormKernel<__half>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNormKernel<uint16_t>(out, in, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
