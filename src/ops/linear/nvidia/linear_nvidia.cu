#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../nvidia/nvidia_common.cuh"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdint>

namespace {

using CublasHandle = void *;
using CublasStatus = int;
using CublasOperation = int;
using CublasComputeType = int;
using CublasGemmAlgo = int;

constexpr CublasStatus CUBLAS_STATUS_SUCCESS_VALUE = 0;
#if defined(__MUSACC__)
constexpr CublasOperation CUBLAS_OP_N_VALUE = 111;
constexpr CublasOperation CUBLAS_OP_T_VALUE = 112;
constexpr CublasComputeType CUBLAS_COMPUTE_32F_VALUE = 68;
constexpr CublasGemmAlgo CUBLAS_GEMM_DEFAULT_TENSOR_OP_VALUE = 1;
#else
constexpr CublasOperation CUBLAS_OP_N_VALUE = 0;
constexpr CublasOperation CUBLAS_OP_T_VALUE = 1;
constexpr CublasComputeType CUBLAS_COMPUTE_32F_VALUE = 68;
constexpr CublasGemmAlgo CUBLAS_GEMM_DEFAULT_TENSOR_OP_VALUE = 99;
#endif

using CublasCreateFn = CublasStatus (*)(CublasHandle *);
using CublasDestroyFn = CublasStatus (*)(CublasHandle);
using CublasSgemmFn = CublasStatus (*)(
    CublasHandle,
    CublasOperation,
    CublasOperation,
    int,
    int,
    int,
    const float *,
    const float *,
    int,
    const float *,
    int,
    const float *,
    float *,
    int);
using CublasGemmExFn = CublasStatus (*)(
    CublasHandle,
    CublasOperation,
    CublasOperation,
    int,
    int,
    int,
    const void *,
    const void *,
    cudaDataType_t,
    int,
    const void *,
    cudaDataType_t,
    int,
    const void *,
    void *,
    cudaDataType_t,
    int,
    CublasComputeType,
    CublasGemmAlgo);

struct CublasApi {
    CublasCreateFn create = nullptr;
    CublasDestroyFn destroy = nullptr;
    CublasSgemmFn sgemm = nullptr;
    CublasGemmExFn gemm_ex = nullptr;
};

void checkCublas(CublasStatus status, const char *message) {
    if (status != CUBLAS_STATUS_SUCCESS_VALUE) {
        std::cerr << "[ERROR] cuBLAS linear error: " << message << ": "
                  << status << EXCEPTION_LOCATION_MSG << std::endl;
        throw std::runtime_error("cuBLAS error");
    }
}

[[noreturn]] void throwMissingCublasSymbol(const char *name) {
    std::cerr << "[ERROR] Failed to load cuBLAS symbol: " << name
              << EXCEPTION_LOCATION_MSG << std::endl;
    throw std::runtime_error("missing cuBLAS symbol");
}

#ifdef _WIN32
HMODULE loadCublasLibrary() {
#if defined(__MUSACC__)
    constexpr const char *library_name = "libmublas.so";
#else
    constexpr const char *library_name = "cublas64_12.dll";
#endif
    auto *handle = LoadLibraryExA(
        library_name,
        nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
    if (handle == nullptr) {
        handle = LoadLibraryA(library_name);
    }
    if (handle == nullptr) {
        std::cerr << "[ERROR] Failed to load BLAS runtime library"
                  << EXCEPTION_LOCATION_MSG << std::endl;
        throw std::runtime_error("failed to load BLAS runtime");
    }
    return handle;
}

template <typename Fn>
Fn loadCublasSymbol(HMODULE handle, const char *name) {
    auto *symbol = GetProcAddress(handle, name);
    if (symbol == nullptr) {
        throwMissingCublasSymbol(name);
    }
    return reinterpret_cast<Fn>(symbol);
}
#else
void *loadCublasLibrary() {
#if defined(__MUSACC__)
    constexpr const char *library_name = "libmublas.so";
#else
    constexpr const char *library_name = "libcublas.so.12";
#endif
#if defined(__MUSACC__)
    void *handle = dlopen(library_name, RTLD_LAZY | RTLD_LOCAL);
#else
    void *handle = dlopen(library_name, RTLD_LAZY | RTLD_LOCAL);
#endif
#if !defined(__MUSACC__)
    if (handle == nullptr) {
        handle = dlopen("libcublas.so", RTLD_LAZY | RTLD_LOCAL);
    }
#endif
    if (handle == nullptr) {
        std::cerr << "[ERROR] Failed to load BLAS runtime library"
                  << EXCEPTION_LOCATION_MSG << std::endl;
        throw std::runtime_error("failed to load BLAS runtime");
    }
    return handle;
}

template <typename Fn>
Fn loadCublasSymbol(void *handle, const char *name) {
    void *symbol = dlsym(handle, name);
    if (symbol == nullptr) {
        throwMissingCublasSymbol(name);
    }
    return reinterpret_cast<Fn>(symbol);
}
#endif

const CublasApi &cublasApi() {
    static const CublasApi api = [] {
        auto handle = loadCublasLibrary();
        CublasApi loaded;
#if defined(__MUSACC__)
        loaded.create = loadCublasSymbol<CublasCreateFn>(handle, "mublasCreate");
        loaded.destroy = loadCublasSymbol<CublasDestroyFn>(handle, "mublasDestroy");
        loaded.sgemm = loadCublasSymbol<CublasSgemmFn>(handle, "mublasSgemm");
        loaded.gemm_ex = loadCublasSymbol<CublasGemmExFn>(
            handle,
            "_Z12mublasGemmExP15_mublasHandle_t17mublasOperation_tS1_iiiPKvS3_14musaDataType_tiS3_S4_iS3_PvS4_i19mublasComputeType_t16mublasGemmAlgo_t");
#else
        loaded.create = loadCublasSymbol<CublasCreateFn>(handle, "cublasCreate_v2");
        loaded.destroy = loadCublasSymbol<CublasDestroyFn>(handle, "cublasDestroy_v2");
        loaded.sgemm = loadCublasSymbol<CublasSgemmFn>(handle, "cublasSgemm_v2");
        loaded.gemm_ex = loadCublasSymbol<CublasGemmExFn>(handle, "cublasGemmEx");
#endif
        return loaded;
    }();
    return api;
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t rows, size_t out_features) {
    const size_t total = rows * out_features;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total;
         i += stride) {
        const size_t col = i % out_features;
        out[i] = llaisys::ops::nvidia::storeFromFloat<T>(
            llaisys::ops::nvidia::loadAsFloat(out[i])
            + llaisys::ops::nvidia::loadAsFloat(bias[col]));
    }
}

template <typename T>
void launchAddBiasKernel(T *out, const T *bias, size_t rows, size_t out_features) {
    if (bias == nullptr || rows == 0 || out_features == 0) {
        return;
    }
    constexpr int block_size = 256;
    const int grid_size = llaisys::ops::nvidia::gridSizeFor(rows * out_features, block_size);
    addBiasKernel<<<grid_size, block_size>>>(out, bias, rows, out_features);
    llaisys::ops::nvidia::checkCuda(cudaGetLastError(), "launch addBiasKernel");
}

void gemmF32(
    CublasHandle handle,
    float *out,
    const float *in,
    const float *weight,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    const auto &api = cublasApi();
    const float alpha = 1.0F;
    const float beta = 0.0F;
    checkCublas(
        api.sgemm(
            handle,
            CUBLAS_OP_T_VALUE,
            CUBLAS_OP_N_VALUE,
            static_cast<int>(out_features),
            static_cast<int>(rows),
            static_cast<int>(in_features),
            &alpha,
            weight,
            static_cast<int>(in_features),
            in,
            static_cast<int>(in_features),
            &beta,
            out,
            static_cast<int>(out_features)),
        "cublasSgemm linear");
}

template <typename T>
void gemmEx(
    CublasHandle handle,
    T *out,
    const T *in,
    const T *weight,
    cudaDataType_t data_type,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    const auto &api = cublasApi();
    const float alpha = 1.0F;
    const float beta = 0.0F;
    checkCublas(
        api.gemm_ex(
            handle,
            CUBLAS_OP_T_VALUE,
            CUBLAS_OP_N_VALUE,
            static_cast<int>(out_features),
            static_cast<int>(rows),
            static_cast<int>(in_features),
            &alpha,
            weight,
            data_type,
            static_cast<int>(in_features),
            in,
            data_type,
            static_cast<int>(in_features),
            &beta,
            out,
            data_type,
            static_cast<int>(out_features),
            CUBLAS_COMPUTE_32F_VALUE,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP_VALUE),
        "cublasGemmEx linear");
}

class ScopedCublasHandleCache {
public:
    ~ScopedCublasHandleCache() {
        if (_handle != nullptr) {
            cublasApi().destroy(_handle);
        }
    }

    CublasHandle get() {
        int current_device = 0;
        llaisys::ops::nvidia::checkCuda(
            cudaGetDevice(&current_device),
            "cudaGetDevice for cuBLAS handle");

        if (_handle != nullptr && _device != current_device) {
            checkCublas(cublasApi().destroy(_handle), "cublasDestroy");
            _handle = nullptr;
            _device = -1;
        }

        if (_handle == nullptr) {
            checkCublas(cublasApi().create(&_handle), "cublasCreate");
            _device = current_device;
        }

        return _handle;
    }

private:
    int _device = -1;
    CublasHandle _handle = nullptr;
};

CublasHandle cachedCublasHandle() {
    thread_local ScopedCublasHandleCache cache;
    return cache.get();
}

template <typename T>
void launchLinearKernel(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    size_t rows,
    size_t in_features,
    size_t out_features);

template <>
void launchLinearKernel<float>(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    CublasHandle handle = cachedCublasHandle();

    auto *typed_out = reinterpret_cast<float *>(out);
    const auto *typed_in = reinterpret_cast<const float *>(in);
    const auto *typed_weight = reinterpret_cast<const float *>(weight);
    const auto *typed_bias = reinterpret_cast<const float *>(bias);
    gemmF32(handle, typed_out, typed_in, typed_weight, rows, in_features, out_features);
    launchAddBiasKernel(typed_out, typed_bias, rows, out_features);

}

template <>
void launchLinearKernel<__half>(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    CublasHandle handle = cachedCublasHandle();

    auto *typed_out = reinterpret_cast<__half *>(out);
    const auto *typed_in = reinterpret_cast<const __half *>(in);
    const auto *typed_weight = reinterpret_cast<const __half *>(weight);
    const auto *typed_bias = reinterpret_cast<const __half *>(bias);
    gemmEx(handle, typed_out, typed_in, typed_weight, CUDA_R_16F, rows, in_features, out_features);
    launchAddBiasKernel(typed_out, typed_bias, rows, out_features);

}

template <>
void launchLinearKernel<uint16_t>(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    CublasHandle handle = cachedCublasHandle();

    auto *typed_out = reinterpret_cast<uint16_t *>(out);
    const auto *typed_in = reinterpret_cast<const uint16_t *>(in);
    const auto *typed_weight = reinterpret_cast<const uint16_t *>(weight);
    const auto *typed_bias = reinterpret_cast<const uint16_t *>(bias);
    gemmEx(handle, typed_out, typed_in, typed_weight, CUDA_R_16BF, rows, in_features, out_features);
    launchAddBiasKernel(typed_out, typed_bias, rows, out_features);

}

} // namespace

namespace llaisys::ops::nvidia {
void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t rows,
    size_t in_features,
    size_t out_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchLinearKernel<float>(out, in, weight, bias, rows, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return launchLinearKernel<__half>(out, in, weight, bias, rows, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return launchLinearKernel<uint16_t>(out, in, weight, bias, rows, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
