#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace {

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < out_features; j++) {
            float sum = 0.0;
            for (size_t k = 0; k < in_features; k++) {
                const float in_v = llaisys::utils::cast<float>(in[i * in_features + k]);
                const float weight_v = llaisys::utils::cast<float>(weight[j * in_features + k]);
                sum += in_v * weight_v;
            }
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * out_features + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            rows,
            in_features,
            out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            rows,
            in_features,
            out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            rows,
            in_features,
            out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
