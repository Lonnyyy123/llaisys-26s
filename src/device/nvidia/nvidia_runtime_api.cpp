#include "../runtime_api.hpp"

#define LLAISYS_MUSA_RUNTIME_ONLY
#include "nvidia_compat.hpp"

namespace llaisys::device::nvidia {

namespace runtime_api {
void checkCuda(cudaError_t err, const char *message) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] CUDA runtime error: " << message << ": "
                  << cudaGetErrorString(err) << EXCEPTION_LOCATION_MSG
                  << std::endl;
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void checkCudaCleanup(cudaError_t err, const char *message) {
    if (err == cudaErrorCudartUnloading) {
        return;
    }
    checkCuda(err, message);
}

cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        CHECK_ARGUMENT(false, "unsupported CUDA memcpy kind");
        return cudaMemcpyDefault;
    }
}

int getDeviceCount() {
    int count = 0;
    const auto err = cudaGetDeviceCount(&count);
    if (err == cudaErrorNoDevice) {
        return 0;
    }
    checkCuda(err, "cudaGetDeviceCount");
    return count;
}

void setDevice(int device_id) {
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    checkCudaCleanup(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    checkCudaCleanup(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    checkCudaCleanup(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    CHECK_ARGUMENT(dst != nullptr, "CUDA memcpy destination cannot be null");
    CHECK_ARGUMENT(src != nullptr, "CUDA memcpy source cannot be null");
    checkCuda(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)), "cudaMemcpy");
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    CHECK_ARGUMENT(dst != nullptr, "CUDA memcpy destination cannot be null");
    CHECK_ARGUMENT(src != nullptr, "CUDA memcpy source cannot be null");
    checkCuda(
        cudaMemcpyAsync(
            dst,
            src,
            size,
            toCudaMemcpyKind(kind),
            reinterpret_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
