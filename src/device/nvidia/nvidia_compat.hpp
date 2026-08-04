#pragma once

#if defined(__MUSACC__)

#if defined(LLAISYS_MUSA_RUNTIME_ONLY)
#include <musa_runtime_api.h>
#else
#include <musa_runtime.h>
#endif

#ifndef LLAISYS_MUSA_RUNTIME_ONLY
#include <library_types.h>
#include <musa_bf16.h>
#include <musa_fp16.h>
#endif

#define cudaError_t musaError_t
#define cudaStream_t musaStream_t
#define cudaMemcpyKind musaMemcpyKind
#define cudaDeviceProp musaDeviceProp
#define cudaDataType_t musaDataType_t

#define cudaSuccess musaSuccess
#define cudaErrorNoDevice musaErrorNoDevice
#define cudaErrorCudartUnloading musaErrorMusartUnloading

#define cudaMemcpyHostToHost musaMemcpyHostToHost
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaMemcpyDefault musaMemcpyDefault

#define cudaGetDeviceCount musaGetDeviceCount
#define cudaGetDevice musaGetDevice
#define cudaSetDevice musaSetDevice
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaGetLastError musaGetLastError
#define cudaGetErrorString musaGetErrorString

#define cudaStreamCreate musaStreamCreate
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamSynchronize musaStreamSynchronize

#define cudaMalloc musaMalloc
#define cudaFree musaFree
#define cudaMallocHost musaMallocHost
#define cudaFreeHost musaFreeHost
#define cudaMemcpy musaMemcpy
#define cudaMemcpyAsync musaMemcpyAsync

#define CUDA_R_16F MUSA_R_16F
#define CUDA_R_16BF MUSA_R_16BF

#endif
