#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>
#include <vector>
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
    const size_t half_dim = head_dim / 2;
    std::vector<float> cos_table(seq_len * half_dim);
    std::vector<float> sin_table(seq_len * half_dim);

    // The rotation angles depend only on position and dimension, not on head.
    // Compute the trigonometric values once and reuse them for every head.
    for (size_t i = 0; i < seq_len; i++) {
        const float position = static_cast<float>(pos_ids[i]);
        for (size_t k = 0; k < half_dim; k++) {
            const float phi = position / std::pow(
                theta,
                2.0F * static_cast<float>(k) / static_cast<float>(head_dim));
            cos_table[i * half_dim + k] = std::cos(phi);
            sin_table[i * half_dim + k] = std::sin(phi);
        }
    }

    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < num_heads; j++) {
            const size_t input_base = (i * num_heads + j) * head_dim;
            for (size_t k = 0; k < half_dim; k++) {
                const float cos_angle = cos_table[i * half_dim + k];
                const float sin_angle = sin_table[i * half_dim + k];
                const float first = llaisys::utils::cast<float>(in[input_base + k]);
                const float second = llaisys::utils::cast<float>(in[input_base + k + half_dim]);
                const float a_out = cos_angle * first - sin_angle * second;
                const float b_out = cos_angle * second + sin_angle * first;
                out[input_base + k] = llaisys::utils::cast<T>(a_out);
                out[input_base + k + half_dim] = llaisys::utils::cast<T>(b_out);
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
