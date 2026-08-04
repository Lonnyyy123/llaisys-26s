#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {

constexpr size_t KV_TILE_SIZE = 128;

__device__ float warpReduceMax(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ float decodeBlockReduceMax(float value, float *warp_values) {
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;
    value = warpReduceMax(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_values[lane] : -FLT_MAX;
    if (warp == 0) {
        value = warpReduceMax(value);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

__device__ float decodeBlockReduceSum(float value, float *warp_values) {
    const size_t lane = threadIdx.x & 31;
    const size_t warp = threadIdx.x >> 5;
    value = warpReduceSum(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < (blockDim.x + 31) / 32 ? warp_values[lane] : 0.0F;
    if (warp == 0) {
        value = warpReduceSum(value);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        warp_values[0] = value;
    }
    __syncthreads();
    return warp_values[0];
}

template <typename T>
__global__ void decodeSelfAttentionKernel(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *acc = scores + KV_TILE_SIZE;
    float *warp_values = acc + value_dim;

    const size_t qh = static_cast<size_t>(blockIdx.x);
    const size_t heads_per_kv = num_heads / num_kv_heads;
    const size_t kv_head = qh / heads_per_kv;

    for (size_t vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
        acc[vd] = 0.0F;
    }
    __syncthreads();

    float max_score = -FLT_MAX;
    float exp_sum = 0.0F;
    const size_t q_offset_base = qh * qk_dim;

    for (size_t tile_start = 0; tile_start < kv_len; tile_start += KV_TILE_SIZE) {
        const size_t remaining = kv_len - tile_start;
        const size_t tile_count = remaining < KV_TILE_SIZE ? remaining : KV_TILE_SIZE;
        float score = -FLT_MAX;
        if (threadIdx.x < tile_count) {
            const size_t ki = tile_start + threadIdx.x;
            const size_t k_offset_base = (ki * num_kv_heads + kv_head) * qk_dim;
            float dot = 0.0F;
            for (size_t d = 0; d < qk_dim; ++d) {
                dot += llaisys::ops::nvidia::loadAsFloat(q[q_offset_base + d])
                    * llaisys::ops::nvidia::loadAsFloat(k[k_offset_base + d]);
            }
            score = dot * scale;
        }

        const float tile_max = decodeBlockReduceMax(score, warp_values);
        const float new_max = fmaxf(max_score, tile_max);
        const float old_scale = max_score == -FLT_MAX ? 0.0F : expf(max_score - new_max);

        float score_exp = 0.0F;
        if (threadIdx.x < tile_count) {
            score_exp = expf(score - new_max);
            scores[threadIdx.x] = score_exp;
        }
        const float tile_exp_sum = decodeBlockReduceSum(score_exp, warp_values);

        for (size_t vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
            float tile_acc = 0.0F;
            for (size_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                const size_t vi = tile_start + tile_idx;
                const size_t v_offset = (vi * num_kv_heads + kv_head) * value_dim + vd;
                tile_acc += scores[tile_idx]
                    * llaisys::ops::nvidia::loadAsFloat(v[v_offset]);
            }
            acc[vd] = acc[vd] * old_scale + tile_acc;
        }
        exp_sum = exp_sum * old_scale + tile_exp_sum;
        max_score = new_max;
        __syncthreads();
    }

    for (size_t vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
        const size_t out_offset = qh * value_dim + vd;
        attn_val[out_offset] = llaisys::ops::nvidia::storeFromFloat<T>(acc[vd] / exp_sum);
    }
}

__device__ float blockReduceMax(float value, float *shared) {
    const size_t tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] = fmaxf(shared[tid], shared[tid + offset]);
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ float blockReduceSum(float value, float *shared) {
    const size_t tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    for (size_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared[tid] += shared[tid + offset];
        }
        __syncthreads();
    }
    return shared[0];
}

template <typename T>
__global__ void selfAttentionKernel(
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
    extern __shared__ float shared[];
    float *scores = shared;
    float *acc = scores + KV_TILE_SIZE;
    float *reduction = acc + value_dim;

    const size_t query_head = static_cast<size_t>(blockIdx.x);
    const size_t qi = query_head / num_heads;
    const size_t qh = query_head % num_heads;
    const size_t heads_per_kv = num_heads / num_kv_heads;
    const size_t past_len = kv_len - q_len;
    const size_t kv_head = qh / heads_per_kv;
    const size_t visible_kv_len = past_len + qi + 1;

    for (size_t vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
        acc[vd] = 0.0F;
    }
    __syncthreads();

    float max_score = -FLT_MAX;
    float exp_sum = 0.0F;

    for (size_t tile_start = 0; tile_start < visible_kv_len; tile_start += KV_TILE_SIZE) {
        const size_t remaining = visible_kv_len - tile_start;
        const size_t tile_count = remaining < KV_TILE_SIZE ? remaining : KV_TILE_SIZE;
        const size_t tid = threadIdx.x;

        float score = -FLT_MAX;
        if (tid < tile_count) {
            const size_t ki = tile_start + tid;
            float dot = 0.0F;
            for (size_t d = 0; d < qk_dim; ++d) {
                const size_t q_offset = (qi * num_heads + qh) * qk_dim + d;
                const size_t k_offset = (ki * num_kv_heads + kv_head) * qk_dim + d;
                dot += llaisys::ops::nvidia::loadAsFloat(q[q_offset])
                    * llaisys::ops::nvidia::loadAsFloat(k[k_offset]);
            }
            score = dot * scale;
        }

        const float tile_max = blockReduceMax(score, reduction);
        const float new_max = fmaxf(max_score, tile_max);
        const float old_scale =
            max_score == -FLT_MAX ? 0.0F : expf(max_score - new_max);

        float score_exp = 0.0F;
        if (tid < tile_count) {
            score_exp = expf(score - new_max);
            scores[tid] = score_exp;
        }
        const float tile_exp_sum = blockReduceSum(score_exp, reduction);

        for (size_t vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
            float tile_acc = 0.0F;
            for (size_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
                const size_t ki = tile_start + tile_idx;
                const size_t v_offset =
                    (ki * num_kv_heads + kv_head) * value_dim + vd;
                tile_acc += scores[tile_idx]
                    * llaisys::ops::nvidia::loadAsFloat(v[v_offset]);
            }
            acc[vd] = acc[vd] * old_scale + tile_acc;
        }

        exp_sum = exp_sum * old_scale + tile_exp_sum;
        max_score = new_max;
        __syncthreads();
    }

    for (size_t vd = threadIdx.x; vd < value_dim; vd += blockDim.x) {
        const float out = acc[vd] / exp_sum;
        const size_t out_offset = (qi * num_heads + qh) * value_dim + vd;
        attn_val[out_offset] = llaisys::ops::nvidia::storeFromFloat<T>(out);
    }
}

template <typename T>
void launchSelfAttentionKernel(
     std::byte *attn_val,
     const std::byte *q,
     const std::byte *k,
     const std::byte *v,
     size_t q_len,
     size_t kv_len,
     size_t num_heads,
     size_t num_kv_heads,
     size_t qk_dim,
     size_t value_dim,
     float scale) {
    if (q_len == 0 || num_heads == 0 || value_dim == 0) {
        return;
    }

    constexpr int block_size = 256;
    const int grid_size = static_cast<int>(q_len * num_heads);
    const size_t shared_bytes =
        (KV_TILE_SIZE + value_dim + static_cast<size_t>(block_size)) * sizeof(float);
    selfAttentionKernel<<<grid_size, block_size, shared_bytes>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        q_len,
        kv_len,
        num_heads,
        num_kv_heads,
        qk_dim,
        value_dim,
        scale);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch selfAttentionKernel");
}

template <typename T>
void launchDecodeSelfAttentionKernel(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    constexpr int block_size = 128;
    const size_t shared_bytes =
        (KV_TILE_SIZE + value_dim + static_cast<size_t>((block_size + 31) / 32))
        * sizeof(float);
    decodeSelfAttentionKernel<<<static_cast<int>(num_heads), block_size, shared_bytes>>>(
        reinterpret_cast<T *>(attn_val),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        kv_len,
        num_heads,
        num_kv_heads,
        qk_dim,
        value_dim,
        scale);
    llaisys::ops::nvidia::checkCuda(
        cudaGetLastError(), "launch decodeSelfAttentionKernel");
}

} // namespace

namespace llaisys::ops::nvidia {
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
    if (q_len == 1) {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            return launchDecodeSelfAttentionKernel<float>(
                attn_val, q, k, v, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale);
        case LLAISYS_DTYPE_F16:
            return launchDecodeSelfAttentionKernel<__half>(
                attn_val, q, k, v, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale);
        case LLAISYS_DTYPE_BF16:
            return launchDecodeSelfAttentionKernel<uint16_t>(
                attn_val, q, k, v, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttentionKernel<float>(
            attn_val, q, k, v, q_len, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttentionKernel<__half>(
            attn_val, q, k, v, q_len, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttentionKernel<uint16_t>(
            attn_val, q, k, v, q_len, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
