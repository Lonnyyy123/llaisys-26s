#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t best_idx = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        const float val = llaisys::utils::cast<float>(vals[i]);
        if (std::isnan(val) || val > best_val) {
            best_idx = i;
            best_val = val;
            if (std::isnan(val)) {
                break;
            }
        }
    }

    *max_idx = static_cast<int64_t>(best_idx);
    *max_val = vals[best_idx];
}

} // namespace

namespace llaisys::ops::cpu {
void argmax(
    int64_t *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(
            max_idx,
            reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals),
            numel);
    case LLAISYS_DTYPE_F16:
        return argmax_(
            max_idx,
            reinterpret_cast<llaisys::fp16_t *>(max_val),
            reinterpret_cast<const llaisys::fp16_t *>(vals),
            numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_(
            max_idx,
            reinterpret_cast<llaisys::bf16_t *>(max_val),
            reinterpret_cast<const llaisys::bf16_t *>(vals),
            numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
