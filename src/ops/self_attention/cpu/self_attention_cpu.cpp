#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
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
    for (size_t qi = 0; qi < q_len; qi++) {
        const size_t visible_kv_len = past_len + qi + 1;
        for (size_t qh = 0; qh < num_heads; qh++) {
            size_t kv_head = qh / heads_per_kv;
            for (size_t ki = 0; ki < kv_len; ki++) {
                if (ki >= visible_kv_len) {
                    scores[ki] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0F;
                for (size_t d = 0; d < qk_dim; d++) {
                    const size_t q_offset = (qi * num_heads + qh) * qk_dim + d;
                    const size_t k_offset = (ki * num_kv_heads + kv_head) * qk_dim + d;
                    dot += llaisys::utils::cast<float>(q[q_offset]) * llaisys::utils::cast<float>(k[k_offset]);
                }
                scores[ki] = dot * scale;
            }
            float max_score = *std::max_element(scores.begin(), scores.end());
            std::transform(scores.begin(), scores.end(), scores.begin(), [=](float x) { return std::exp(x - max_score); });
            float expsum = std::accumulate(scores.begin(), scores.end(), 0.0F);
            std::transform(scores.begin(), scores.end(), probs.begin(), [=](float x) { return x / expsum; });
            for (size_t vd = 0; vd < value_dim; vd++) {
                float out = 0.0F;
                for (size_t vi = 0; vi < kv_len; vi++) {
                    const size_t v_offset = (vi * num_kv_heads + kv_head) * value_dim+vd;
                    out += probs[vi] * llaisys::utils::cast<float>(v[v_offset]);
                }
                const size_t attn_offset = (qi * num_heads + qh) * value_dim+vd;
                attn_val[attn_offset] = llaisys::utils::cast<T>(out);
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
