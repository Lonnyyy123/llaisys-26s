#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->ndim() == 2, "RMS norm: out must be a 2D tensor");
    CHECK_ARGUMENT(in->ndim() == 2, "RMS norm: in must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 1, "RMS norm: weight must be a 1D tensor");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(
        weight->shape()[0] == in->shape()[1],
        "RMS norm: weight length must equal the row length");
    CHECK_ARGUMENT(eps >= 0.0F, "RMS norm: eps must be non-negative");
    CHECK_ARGUMENT(
        out->isContiguous() && in->isContiguous() && weight->isContiguous(),
        "RMS norm: all tensors must be contiguous");

    const size_t rows = in->shape()[0];
    const size_t cols = in->shape()[1];
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            rows,
            cols,
            eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            rows,
            cols,
            eps);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
