import os
import sys
import ctypes
from pathlib import Path

from .runtime import load_runtime
from .runtime import LlaisysRuntimeAPI
from .llaisys_types import llaisysDeviceType_t, DeviceType
from .llaisys_types import llaisysDataType_t, DataType
from .llaisys_types import llaisysMemcpyKind_t, MemcpyKind
from .llaisys_types import llaisysStream_t
from .tensor import llaisysTensor_t
from .tensor import load_tensor
from .ops import load_ops
from .qwen2 import LlaisysQwen2Meta, LlaisysQwen2Weights
from .qwen2 import llaisysQwen2Model_t, llaisysQwen2Weights_t
from .qwen2 import load_qwen2


_DLL_DIRECTORY_HANDLES = []


def add_windows_dll_directories(lib_dir):
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    candidates = [Path(lib_dir)]
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        cuda_path = os.environ.get(env_name)
        if cuda_path:
            candidates.append(Path(cuda_path) / "bin")

    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if path_entry:
            candidates.append(Path(path_entry))

    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(resolved)))


def load_shared_library():
    lib_dir = Path(__file__).parent

    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    lib_path = os.path.join(lib_dir, libname)

    if not os.path.isfile(lib_path):
        raise FileNotFoundError(f"Shared library not found: {lib_path}")

    add_windows_dll_directories(lib_dir)
    return ctypes.CDLL(str(lib_path))


LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)


__all__ = [
    "LIB_LLAISYS",
    "LlaisysRuntimeAPI",
    "llaisysStream_t",
    "llaisysTensor_t",
    "llaisysDataType_t",
    "DataType",
    "llaisysDeviceType_t",
    "DeviceType",
    "llaisysMemcpyKind_t",
    "MemcpyKind",
    "llaisysStream_t",
    "llaisysQwen2Model_t",
    "LlaisysQwen2Meta",
    "LlaisysQwen2Weights",
    "llaisysQwen2Weights_t",
    "load_qwen2",
]
