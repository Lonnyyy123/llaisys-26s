#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

namespace {

template <typename T>
void embedding_(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {
    for (size_t i = 0; i < num_indices; ++i) {
        const int64_t row = index[i];

        CHECK_ARGUMENT(
            row >= 0 && static_cast<size_t>(row) < num_embeddings,
            "Embedding: index out of range");

        const T *src = weight + static_cast<size_t>(row) * embedding_dim;

        T *dst = out + i * embedding_dim;

        std::copy_n(src, embedding_dim, dst);
    }
}

} // namespace

namespace llaisys::ops::cpu {
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
        return embedding_(
            reinterpret_cast<float *>(out),
            index,
            reinterpret_cast<const float *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            index,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            index,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
