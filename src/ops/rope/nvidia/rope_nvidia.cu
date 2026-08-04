#include "rope_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

#include <cstdint>
#include <mutex>
#include <vector>

namespace {

struct RopeFrequencyCacheEntry {
    int device;
    size_t half_dim;
    float theta;
    float *inv_freq;
};

__global__ void initializeInvFrequency(float *inv_freq, size_t half_dim, float theta) {
    for (size_t i = threadIdx.x; i < half_dim; i += blockDim.x) {
        const float exponent =
            2.0F * static_cast<float>(i) / static_cast<float>(half_dim * 2);
        inv_freq[i] = 1.0F / powf(theta, exponent);
    }
}

// RoPE is called once for Q and once for K in every layer. Keep the frequency
// table on the device so powf is paid only once per (device, dimension, theta).
const float *getInvFrequency(size_t half_dim, float theta) {
    static std::mutex mutex;
    static std::vector<RopeFrequencyCacheEntry> entries;

    int device = 0;
    llaisys::ops::nvidia::checkCuda(cudaGetDevice(&device), "cudaGetDevice for RoPE");

    std::lock_guard<std::mutex> lock(mutex);
    for (const auto &entry : entries) {
        if (entry.device == device && entry.half_dim == half_dim && entry.theta == theta) {
            return entry.inv_freq;
        }
    }

    float *device_freq = nullptr;
    llaisys::ops::nvidia::checkCuda(
        cudaMalloc(&device_freq, half_dim * sizeof(float)),
        "cudaMalloc RoPE inverse frequency");
    initializeInvFrequency<<<1, 256>>>(device_freq, half_dim, theta);
    llaisys::ops::nvidia::checkCuda(
        cudaGetLastError(),
        "initialize RoPE inverse frequency");
    entries.push_back({device, half_dim, theta, device_freq});
    return device_freq;
}

template <typename T>
__global__ void ropeKernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    const float *inv_freq) {
    const size_t block = static_cast<size_t>(blockIdx.x);
    const size_t seq_idx = block / num_heads;
    const size_t head_idx = block % num_heads;
    const size_t half_dim = head_dim / 2;

    if (seq_idx >= seq_len) {
        return;
    }

    for (size_t k = threadIdx.x; k < half_dim; k += blockDim.x) {
        const size_t base = (seq_idx * num_heads + head_idx) * head_dim;

        const size_t idx0 = base + k;
        const size_t idx1 = base + k + half_dim;

        const float a = llaisys::ops::nvidia::loadAsFloat(in[idx0]);
        const float b = llaisys::ops::nvidia::loadAsFloat(in[idx1]);

        const float angle = static_cast<float>(pos_ids[seq_idx]) * inv_freq[k];

        const float cos_angle = cosf(angle);
        const float sin_angle = sinf(angle);

        out[idx0] = llaisys::ops::nvidia::storeFromFloat<T>(a * cos_angle - b * sin_angle);

        out[idx1] = llaisys::ops::nvidia::storeFromFloat<T>(b * cos_angle + a * sin_angle);
    }
}

template <typename T>
void launchRopeKernel(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    if (seq_len == 0 || num_heads == 0 || head_dim == 0) {
        return;
    }
    const float *inv_freq = getInvFrequency(head_dim / 2, theta);
    const int block_size = 256;
    const int grid_size = static_cast<int>(seq_len * num_heads);
    ropeKernel<<<grid_size, block_size>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        pos_ids,
        seq_len,
        num_heads,
        head_dim,
        inv_freq);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch ropeKernel");
}

} // namespace

namespace llaisys::ops::nvidia {
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
        return launchRopeKernel<float>(out, in, pos_ids, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return launchRopeKernel<__half>(out, in, pos_ids, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return launchRopeKernel<uint16_t>(out, in, pos_ids, seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
