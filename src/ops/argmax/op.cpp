#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax: vals must not be empty");
    CHECK_ARGUMENT(
        max_idx->ndim() == 1 && max_idx->numel() == 1,
        "Argmax: max_idx must be a 1D tensor with one element");
    CHECK_ARGUMENT(
        max_val->ndim() == 1 && max_val->numel() == 1,
        "Argmax: max_val must be a 1D tensor with one element");
    CHECK_ARGUMENT(
        max_idx->dtype() == LLAISYS_DTYPE_I64,
        "Argmax: max_idx must have int64 data type");
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(
        max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
        "Argmax: all tensors must be contiguous");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            reinterpret_cast<int64_t *>(max_idx->data()),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            reinterpret_cast<int64_t *>(max_idx->data()),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(
            reinterpret_cast<int64_t *>(max_idx->data()),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
