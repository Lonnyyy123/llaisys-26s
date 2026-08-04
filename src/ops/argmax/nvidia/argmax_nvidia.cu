#include "argmax_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

#include <cstdint>

namespace {

__device__ bool isBetterCandidate(
    float candidate_val,
    int64_t candidate_idx,
    float best_val,
    int64_t best_idx) {
    const bool candidate_nan = isnan(candidate_val);
    const bool best_nan = isnan(best_val);

    if (candidate_nan || best_nan) {
        return candidate_nan && (!best_nan || candidate_idx < best_idx);
    }
    return candidate_val > best_val
        || (candidate_val == best_val && candidate_idx < best_idx);
}

template <typename T>
__global__ void argmaxBlockKernel(
    int64_t *partial_idx,
    T *partial_val,
    const T *vals,
    size_t numel) {
    __shared__ float shared_val[256];
    __shared__ int64_t shared_idx[256];

    const size_t tid = threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    int64_t best_idx = 0;
    float best_val = llaisys::ops::nvidia::loadAsFloat(vals[0]);
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
         i < numel;
         i += stride) {
        const float value = llaisys::ops::nvidia::loadAsFloat(vals[i]);
        const auto idx = static_cast<int64_t>(i);
        if (isBetterCandidate(value, idx, best_val, best_idx)) {
            best_val = value;
            best_idx = idx;
        }
    }

    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset
            && isBetterCandidate(
                shared_val[tid + offset],
                shared_idx[tid + offset],
                shared_val[tid],
                shared_idx[tid])) {
            shared_val[tid] = shared_val[tid + offset];
            shared_idx[tid] = shared_idx[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_idx[blockIdx.x] = shared_idx[0];
        partial_val[blockIdx.x] = vals[shared_idx[0]];
    }
}

template <typename T>
__global__ void argmaxFinalKernel(
    int64_t *max_idx,
    T *max_val,
    const int64_t *partial_idx,
    const T *partial_val,
    size_t num_blocks) {
    __shared__ float shared_val[256];
    __shared__ int64_t shared_idx[256];
    __shared__ size_t shared_block[256];

    const size_t tid = threadIdx.x;

    size_t best_block = 0;
    int64_t best_idx = partial_idx[0];
    float best_val = llaisys::ops::nvidia::loadAsFloat(partial_val[0]);
    for (size_t block = tid; block < num_blocks; block += blockDim.x) {
        const float value = llaisys::ops::nvidia::loadAsFloat(partial_val[block]);
        const int64_t idx = partial_idx[block];
        if (isBetterCandidate(value, idx, best_val, best_idx)) {
            best_val = value;
            best_idx = idx;
            best_block = block;
        }
    }

    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    shared_block[tid] = best_block;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset
            && isBetterCandidate(
                shared_val[tid + offset],
                shared_idx[tid + offset],
                shared_val[tid],
                shared_idx[tid])) {
            shared_val[tid] = shared_val[tid + offset];
            shared_idx[tid] = shared_idx[tid + offset];
            shared_block[tid] = shared_block[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = shared_idx[0];
        *max_val = partial_val[shared_block[0]];
    }
}

template <typename T>
void launchArgmaxKernel(
    int64_t *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel) {
    if (numel == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size = llaisys::ops::nvidia::gridSizeFor(numel, block_size);
    int64_t *partial_idx = nullptr;
    T *partial_val = nullptr;
    llaisys::ops::nvidia::checkCuda(
        cudaMalloc(&partial_idx, static_cast<size_t>(grid_size) * sizeof(int64_t)),
        "cudaMalloc partial_idx");
    llaisys::ops::nvidia::checkCuda(
        cudaMalloc(&partial_val, static_cast<size_t>(grid_size) * sizeof(T)),
        "cudaMalloc partial_val");

    auto *typed_max_val = reinterpret_cast<T *>(max_val);
    const auto *typed_vals = reinterpret_cast<const T *>(vals);
    argmaxBlockKernel<<<grid_size, block_size>>>(
        partial_idx,
        partial_val,
        typed_vals,
        numel);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch argmaxBlockKernel");

    argmaxFinalKernel<<<1, block_size>>>(
        max_idx,
        typed_max_val,
        partial_idx,
        partial_val,
        static_cast<size_t>(grid_size));
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch argmaxFinalKernel");

    llaisys::ops::nvidia::checkCuda(cudaFree(partial_idx), "cudaFree partial_idx");
    llaisys::ops::nvidia::checkCuda(cudaFree(partial_val), "cudaFree partial_val");
}

} // namespace

namespace llaisys::ops::nvidia {
void argmax(
    int64_t *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchArgmaxKernel<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return launchArgmaxKernel<__half>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return launchArgmaxKernel<uint16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
