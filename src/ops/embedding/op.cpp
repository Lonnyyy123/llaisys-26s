#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding: index must be a 1D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "Embedding: weight must be a 2D tensor");
    CHECK_ARGUMENT(out->ndim() == 2, "Embedding: out must be a 2D tensor");
    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64,
        "Embedding: index must have int64 data type");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(
        out->shape()[0] == index->shape()[0]
            && out->shape()[1] == weight->shape()[1],
        "Embedding: out shape must be [index length, embedding dimension]");
    CHECK_ARGUMENT(
        out->isContiguous() && index->isContiguous() && weight->isContiguous(),
        "Embedding: all tensors must be contiguous");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            out->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            out->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            out->dtype(),
            index->numel(),
            weight->shape()[0],
            weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
