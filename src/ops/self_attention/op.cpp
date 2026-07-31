#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3, "Self attention: attn_val must be a 3D tensor");
    CHECK_ARGUMENT(q->ndim() == 3, "Self attention: q must be a 3D tensor");
    CHECK_ARGUMENT(k->ndim() == 3, "Self attention: k must be a 3D tensor");
    CHECK_ARGUMENT(v->ndim() == 3, "Self attention: v must be a 3D tensor");

    const size_t q_len = q->shape()[0];
    const size_t kv_len = k->shape()[0];
    const size_t num_heads = q->shape()[1];
    const size_t num_kv_heads = k->shape()[1];
    const size_t qk_dim = q->shape()[2];
    const size_t value_dim = v->shape()[2];

    CHECK_ARGUMENT(kv_len >= q_len, "Self attention: key length must be at least query length");
    CHECK_ARGUMENT(k->shape()[2] == qk_dim, "Self attention: q and k dimensions must match");
    CHECK_ARGUMENT(
        v->shape()[0] == kv_len && v->shape()[1] == num_kv_heads,
        "Self attention: k and v sequence/head dimensions must match");
    CHECK_ARGUMENT(
        num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "Self attention: query heads must be divisible by key/value heads");
    CHECK_ARGUMENT(
        attn_val->shape()[0] == q_len
            && attn_val->shape()[1] == num_heads
            && attn_val->shape()[2] == value_dim,
        "Self attention: attn_val shape must be [query length, query heads, value dimension]");
    CHECK_ARGUMENT(
        attn_val->isContiguous() && q->isContiguous()
            && k->isContiguous() && v->isContiguous(),
        "Self attention: all tensors must be contiguous");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
    }

    llaisys::core::context().setDevice(
        attn_val->deviceType(),
        attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
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
