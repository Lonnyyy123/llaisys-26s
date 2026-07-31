#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
namespace {

template <typename T>
void rope_(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    size_t base = head_dim / 2;
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_heads; j++) {
            for (size_t k = 0; k < head_dim / 2; k++) {
                float phi = static_cast<float>(pos_ids[i]) / std::pow(theta, 2.0F * k / head_dim);
                float cos = std::cos(phi),sin = std::sin(phi);
                float a_out = cos *llaisys::utils::cast<float>(in[i * num_heads * head_dim + j * head_dim + k])
                            - sin *llaisys::utils::cast<float>(in[i * num_heads * head_dim + j * head_dim + k+base]);
                float b_out = cos *llaisys::utils::cast<float>(in[i * num_heads * head_dim + j * head_dim + k+base])
                            + sin *llaisys::utils::cast<float>(in[i * num_heads * head_dim + j * head_dim + k]);
                out[i * num_heads * head_dim + j * head_dim + k] = llaisys::utils::cast<T>(a_out);
                out[i * num_heads * head_dim + j * head_dim + k+base] = llaisys::utils::cast<T>(b_out);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void rope(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ids,
            seq_len,
            num_heads,
            head_dim,
            theta);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ids,
            seq_len,
            num_heads,
            head_dim,
            theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ids,
            seq_len,
            num_heads,
            head_dim,
            theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
