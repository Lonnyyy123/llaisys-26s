#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->ndim() == 3, "RoPE: out must be a 3D tensor");
    CHECK_ARGUMENT(in->ndim() == 3, "RoPE: in must be a 3D tensor");
    CHECK_ARGUMENT(pos_ids->ndim() == 1, "RoPE: pos_ids must be a 1D tensor");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(
        pos_ids->dtype() == LLAISYS_DTYPE_I64,
        "RoPE: pos_ids must have int64 data type");
    CHECK_ARGUMENT(
        pos_ids->shape()[0] == in->shape()[0],
        "RoPE: pos_ids length must equal sequence length");
    CHECK_ARGUMENT(
        in->shape()[2] % 2 == 0,
        "RoPE: head dimension must be even");
    CHECK_ARGUMENT(theta > 0.0F, "RoPE: theta must be positive");
    CHECK_ARGUMENT(
        out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
        "RoPE: all tensors must be contiguous");

    const size_t seq_len = in->shape()[0];
    const size_t num_heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            seq_len,
            num_heads,
            head_dim,
            theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            seq_len,
            num_heads,
            head_dim,
            theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            out->dtype(),
            seq_len,
            num_heads,
            head_dim,
            theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
