#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_ARGUMENT(out->ndim() == 2, "Linear: out must be a 2D tensor");
    CHECK_ARGUMENT(in->ndim() == 2, "Linear: in must be a 2D tensor");
    CHECK_ARGUMENT(weight->ndim() == 2, "Linear: weight must be a 2D tensor");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1, "Linear: bias must be a 1D tensor");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    const size_t rows = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];
    CHECK_ARGUMENT(
        weight->shape()[1] == in_features,
        "Linear: in and weight feature dimensions must match");
    CHECK_ARGUMENT(
        out->shape()[0] == rows && out->shape()[1] == out_features,
        "Linear: out shape must be [rows, out features]");
    CHECK_ARGUMENT(
        bias == nullptr || bias->shape()[0] == out_features,
        "Linear: bias length must equal out features");
    CHECK_ARGUMENT(
        out->isContiguous() && in->isContiguous() && weight->isContiguous()
            && (bias == nullptr || bias->isContiguous()),
        "Linear: all tensors must be contiguous");

    const std::byte *bias_data = bias == nullptr ? nullptr : bias->data();
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            rows,
            in_features,
            out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            rows,
            in_features,
            out_features);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            rows,
            in_features,
            out_features);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
