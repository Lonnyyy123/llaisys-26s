#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
namespace {

template <typename T>
void rms_norm_(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t cols,
    float eps) {
    for (size_t i = 0; i < rows; ++i) {
        float square_sum = 0.0F;
        for (size_t j = 0; j < cols; ++j) {
            const float in_v = llaisys::utils::cast<float>(in[i * cols + j]);
            square_sum += in_v * in_v;
        }
        const float inverse_rms = 1.0F / std::sqrt(square_sum / static_cast<float>(cols) + eps);
        for (size_t j = 0; j < cols; ++j) {
            const float in_v = llaisys::utils::cast<float>(in[i * cols + j]);
            const float weight_v = llaisys::utils::cast<float>(weight[j]);
            const float out_v = in_v * inverse_rms * weight_v;
            out[i * cols + j] = llaisys::utils::cast<T>(out_v);
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
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
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            rows,
            cols,
            eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            rows,
            cols,
            eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            rows,
            cols,
            eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
