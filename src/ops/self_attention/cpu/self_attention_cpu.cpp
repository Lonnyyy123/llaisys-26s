#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

template <typename T>
void self_attention_(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    const size_t heads_per_kv = num_heads / num_kv_heads;
    const size_t past_len = kv_len - q_len;
    std::vector<float> scores(kv_len);
    std::vector<float> probs(kv_len);
    std::vector<float> output(value_dim);
    for (size_t qi = 0; qi < q_len; qi++) {
        const size_t visible_kv_len = past_len + qi + 1;
        for (size_t qh = 0; qh < num_heads; qh++) {
            const size_t kv_head = qh / heads_per_kv;
            const T *q_row = q + (qi * num_heads + qh) * qk_dim;
            for (size_t ki = 0; ki < visible_kv_len; ki++) {
                const T *k_row = k + (ki * num_kv_heads + kv_head) * qk_dim;
                float dot = 0.0F;
                for (size_t d = 0; d < qk_dim; d++) {
                    dot += llaisys::utils::cast<float>(q_row[d]) *
                           llaisys::utils::cast<float>(k_row[d]);
                }
                scores[ki] = dot * scale;
            }

            const float max_score = *std::max_element(
                scores.begin(), scores.begin() + visible_kv_len);
            float expsum = 0.0F;
            for (size_t vi = 0; vi < visible_kv_len; vi++) {
                probs[vi] = std::exp(scores[vi] - max_score);
                expsum += probs[vi];
            }
            const float inv_expsum = 1.0F / expsum;
            for (size_t vi = 0; vi < visible_kv_len; vi++) {
                probs[vi] *= inv_expsum;
            }

            std::fill(output.begin(), output.end(), 0.0F);
            for (size_t vi = 0; vi < visible_kv_len; vi++) {
                const float probability = probs[vi];
                const T *v_row = v + (vi * num_kv_heads + kv_head) * value_dim;
                for (size_t vd = 0; vd < value_dim; vd++) {
                    output[vd] += probability *
                                  llaisys::utils::cast<float>(v_row[vd]);
                }
            }
            T *output_row = attn_val + (qi * num_heads + qh) * value_dim;
            for (size_t vd = 0; vd < value_dim; vd++) {
                output_row[vd] = llaisys::utils::cast<T>(output[vd]);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {
void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
