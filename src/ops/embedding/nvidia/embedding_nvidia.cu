#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

namespace {

template <typename T>
__global__ void embeddingKernel(T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {
    const size_t total = num_indices * embedding_dim;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride) {
        const size_t row = linear / embedding_dim;
        const size_t col = linear % embedding_dim;

        const int64_t idx = index[row];

        out[row * embedding_dim + col] =
            weight[static_cast<size_t>(idx) * embedding_dim + col];
    }
}

template <typename T>
void launchEmbeddingKernel(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {
    const size_t total = num_indices * embedding_dim;
    if (total == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size = llaisys::ops::nvidia::gridSizeFor(total, block_size);
    embeddingKernel<<<grid_size, block_size>>>(
        reinterpret_cast<T *>(out),
        index,
        reinterpret_cast<const T *>(weight),
        num_indices,
        num_embeddings,
        embedding_dim);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch embeddingKernel");

}

} // namespace

namespace llaisys::ops::nvidia {
void embedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchEmbeddingKernel<float>(out, index, weight, num_indices, num_embeddings, embedding_dim);
    case LLAISYS_DTYPE_F16:
        return launchEmbeddingKernel<uint16_t>(out, index, weight, num_indices, num_embeddings, embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return launchEmbeddingKernel<uint16_t>(out, index, weight, num_indices, num_embeddings, embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
