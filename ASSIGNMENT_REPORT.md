# LLAISYS 作业 1-4 实验报告

## 1. 实验概述

本文记录 Tensor、CPU 算子、Qwen2 推理、NVIDIA CUDA 后端以及摩尔线程 MUSA 后端的实现、复现流程和实验结果。

测试模型为 `DeepSeek-R1-Distill-Qwen-1.5B`。模型权重通过 Hugging Face 镜像下载，推理结果使用贪心解码，并与 PyTorch Transformers 实现进行对比。

## 2. 复现环境

### Windows 本地环境

- 系统：Windows x64
- Python：`.venv\Scripts\python.exe`
- 构建工具：xmake
- 主要验证内容：Tensor、CPU 算子、模型推理接口和 CUDA 代码编译

### Linux NVIDIA 服务器

- CPU：Intel Xeon Platinum 8358P
- 内存：48 GB
- GPU：NVIDIA GeForce RTX 5090，32 GB 显存
- CUDA：13.3
- Python：3.12
- 模型目录：`/data/hf/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`

### Linux 摩尔线程服务器

- 系统：Linux x86_64
- CPU：Intel Xeon Platinum 8358P
- 内存：48 GB
- GPU：摩尔线程 MTT S5000，80 GB 显存
- 驱动：3.3.5-server
- MUSA：4.3.5
- MUSA 架构参数：`mp_31`
- 模型目录：同上

模型下载使用国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf
```

## 3. 复现流程

### CPU 构建和测试

```powershell
xmake build llaisys
xmake install llaisys
$env:PYTHONPATH = (Resolve-Path 'python').Path
.\.venv\Scripts\python.exe test\test_tensor.py
.\.venv\Scripts\python.exe test\ops\argmax.py --device cpu
.\.venv\Scripts\python.exe test\ops\embedding.py --device cpu
.\.venv\Scripts\python.exe test\ops\linear.py --device cpu
.\.venv\Scripts\python.exe test\ops\rms_norm.py --device cpu
.\.venv\Scripts\python.exe test\ops\rope.py --device cpu
.\.venv\Scripts\python.exe test\ops\self_attention.py --device cpu
.\.venv\Scripts\python.exe test\ops\swiglu.py --device cpu
```

### NVIDIA 构建和测试

```bash
export PATH=/root/.local/bin:/usr/local/cuda-13.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LD_LIBRARY_PATH
export CPATH=/usr/local/cuda-13.3/include:$CPATH
export LIBRARY_PATH=/usr/local/cuda-13.3/lib64:$LIBRARY_PATH
cd /data/llaisys
xmake f --root --nv-gpu=true --cuda=/usr/local/cuda-13.3 --cuda-arch=sm_120 --ccache=false -y
xmake build --root -r llaisys
xmake install --root llaisys
export PYTHONPATH=/data/llaisys/python
export TRITON_PTXAS_PATH=/usr/local/cuda-13.3/bin/ptxas
python3 test/ops/rope.py --device nvidia
python3 test/ops/rms_norm.py --device nvidia
```

### 摩尔线程 MUSA 构建和测试

MUSA SDK 的 Runtime API 与 CUDA Runtime API 结构相近，因此后端复用了原有的 NVIDIA C API 和设备枚举，在编译期通过 `nvidia_compat.hpp` 将 Runtime、数据类型和内存拷贝 API 映射到 MUSA。MUSA kernel 使用 `mcc` 编译，目标架构为 `mp_31`。

服务器初始没有预装 xmake，后续已安装 xmake，并使用项目中的 MUSA 配置完成正式构建和安装。由于 xmake 默认会把 `.cu` 识别为 CUDA 源文件，项目增加了 `musa-g++` wrapper，让 xmake 使用 `mcc` 编译 MUSA kernel，同时绕过自动添加的 `-x c++` 参数。

```bash
export MUSA_HOME=/usr/local/musa
export PATH=$MUSA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH
export XMAKE_ROOT=y
cd /data/llaisys
xmake f -y --nv-gpu=n --musa-gpu=y \
    --musa-path=$MUSA_HOME --musa-arch=mp_31 --cc=gcc --cxx=g++
xmake build -j2
xmake install llaisys
```

## 4. 作业 1：Tensor

实现内容包括：

- `Tensor::load`
- `Tensor::isContiguous`
- `Tensor::view`
- `Tensor::permute`
- `Tensor::slice`
- `contiguous`
- `reshape`
- CPU、GPU 之间的 Tensor 拷贝

`permute` 和 `slice` 通过修改 shape、stride 和 storage offset 创建视图，不搬运底层数据；`view` 根据 stride 判断是否可以零拷贝重解释；`reshape` 在可行时复用 view，否则先生成 contiguous 副本。

测试结果：

```text
test/test_tensor.py: passed
```

## 5. 作业 2：CPU 算子

实现的 CPU 算子：`argmax`、`embedding`、`linear`、`rms_norm`、`rope`、`self_attention` 和 `swiglu`。

支持的数据类型：`F32`、`F16` 和 `BF16`。半精度算子的中间计算转换为 `float`，最后再转换回目标数据类型。

测试结果：

```text
test/ops/argmax.py: passed
test/ops/embedding.py: passed
test/ops/linear.py: passed
test/ops/rms_norm.py: passed
test/ops/rope.py: passed
test/ops/self_attention.py: passed
test/ops/swiglu.py: passed
```

## 6. 作业 3：Qwen2 推理

Python 前端读取 `config.json`、解析 safetensors 文件，并通过 ctypes 调用 C API；模型 forward 和权重 Tensor 管理均在 C/C++ 后端执行。

推理流程如下：

```text
token ids -> embedding -> attention RMSNorm -> Q/K/V projection -> RoPE
-> 更新 KV Cache -> causal self-attention -> output projection -> residual add
-> MLP RMSNorm -> gate/up projection -> SwiGLU -> down projection
-> residual add -> final RMSNorm -> lm_head -> argmax
```

KV Cache 形状为 `[maxseq, num_kv_heads, head_dim]`。首次推理输入完整 prompt，之后每次只输入一个新 token，并把新生成的 K/V 写入对应位置，避免重复计算历史 token 的 K/V。

```powershell
.\.venv\Scripts\python.exe test\test_infer.py --model "<model_path>" --test --max_steps 8
```

PyTorch 和 LLAISYS 的 token 输出一致。

## 7. 作业 4：NVIDIA CUDA 后端

实现了 NVIDIA Runtime API，包括设备管理、Stream、同步、设备内存、Pinned Host Memory 以及同步/异步内存拷贝。

已实现 NVIDIA 算子：

- `add`
- `argmax`
- `embedding`
- `linear`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`

### 7.1 RoPE 频率预计算

原始实现会在每个线程中实时执行 `powf`、`cosf` 和 `sinf`。现在第一次遇到某个 `(device, head_dim, theta)` 组合时，用 CUDA kernel 计算并缓存 `inv_freq`：

```text
inv_freq[i] = 1 / powf(theta, 2 * i / head_dim)
angle = position * inv_freq[i]
```

后续 Q/K 的 RoPE 调用直接复用 GPU 上的频率表，避免重复执行 `powf`。频率表在 CUDA kernel 中初始化，以保持与 CUDA `powf` 一致的数值精度。

### 7.2 RMSNorm warp-level reduction

平方和归约由共享内存树形归约改为 warp-level reduction：每个线程计算局部平方和，使用 `__shfl_down_sync` 完成 warp 内归约，只把每个 warp 的结果写入共享内存，再由第一个 warp 完成最终归约。这样减少了共享内存访问和同步次数。

### 7.3 Decode 专用 self-attention

当 `q_len == 1` 时使用单独的 decode kernel：每个 query head 对应一个 CUDA block，使用 128 个线程，采用 warp-level max/sum reduction，并按 KV tile 执行 online softmax，避免保存完整 attention 矩阵。`q_len > 1` 时继续使用通用 prefill kernel。

### 7.4 资源复用

模型执行器增加 workspace，对相同 shape 和 dtype 的临时 Tensor 进行复用，减少 decode 阶段重复的 `cudaMalloc/cudaFree`。linear 算子使用 cuBLAS，并缓存每个线程的 cuBLAS handle，避免每次调用都创建和销毁 handle。

## 8. NVIDIA 正确性测试

远程 RTX 5090 上已验证：

```text
rope: passed, F32/F16/BF16
rms_norm: passed, F32/F16/BF16
self_attention: custom decode/prefill test passed
add: passed
linear: passed
```

官方 `test/ops/self_attention.py` 的 PyTorch 参考代码存在 attention mask 在 CPU、attention bias 在 CUDA 的设备不一致问题。因此未修改测试文件，使用等价手动测试覆盖了 `q_len=1` decode 和 `q_len>1` prefill 两条路径。

## 9. NVIDIA 端到端性能

测试条件：RTX 5090 32 GB，batch size 为 1，prompt 为 `Who are you?`（8 个 token），关闭 EOS 提前终止，使用贪心解码；每次测试前进行短 warmup，并使用 CUDA synchronize 计时。PyTorch 对比对象为 Hugging Face `Transformers.generate()`。

### 中等长度

| 新生成 token | PyTorch 时间 | LLAISYS 时间 | PyTorch 吞吐 | LLAISYS 吞吐 | 加速比 | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 128 | 1.660 s | 0.431 s | 77.10 tok/s | 297.09 tok/s | 3.85x | 是 |
| 192 | 2.485 s | 0.676 s | 77.27 tok/s | 284.19 tok/s | 3.68x | 是 |
| 240 | 3.107 s | 0.868 s | 77.25 tok/s | 276.34 tok/s | 3.58x | 是 |

### 2000 多 token 长序列

将 `LLAISYS_MAX_SEQ_LEN` 设置为 2304，生成 2048 个新 token：

| 新生成 token | PyTorch 时间 | LLAISYS 时间 | PyTorch 吞吐 | LLAISYS 吞吐 | 加速比 | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2048 | 26.957 s | 15.637 s | 75.97 tok/s | 130.97 tok/s | 1.72x | 是 |

长序列下加速比从约 3.6-3.9 倍下降到 1.72 倍，主要原因是 KV Cache 变长后，attention 对历史 K/V 的读取和计算量增加，GPU 计算占比变高，Python 调度开销占比下降。即使如此，LLAISYS 仍保持更高吞吐，并且生成 token 完全一致。

此前约 240 token 的 LLAISYS 吞吐约为 245 tok/s；加入 cuBLAS handle 缓存、workspace 复用以及 decode attention 优化后，在相近测试条件下提升到约 276 tok/s，约提升 12%。

## 10. 平台支持状态

| 平台 | 后端 | 状态 | 说明 |
| --- | --- | --- | --- |
| Windows x64 | CPU | 已验证 | 作业 1-3 的 CPU 构建和测试通过。 |
| Linux x64 | NVIDIA CUDA 13.3 | 已验证 | 在 RTX 5090 上完成构建、算子测试和端到端推理。 |
| Windows x64 | NVIDIA CUDA | 已编译 | CUDA 源码和 NVIDIA 目标通过 NVCC 编译，但本机当前没有可用 NVIDIA device。 |
| Linux x64 | CPU | 已构建 | 服务器上 CPU 源码参与完整 Linux 构建，未单独记录 CPU 性能。 |
| Linux x64 | 摩尔线程 MUSA 4.3.5 | 已验证 | MTT S5000 上完成 Runtime、8 个 MUSA kernel、共享库、端到端推理和 token 一致性测试。 |
| 沐曦等其他平台 | 未实现 | 未验证 | 当前没有对应 Runtime API 和算子后端。 |

## 11. 摩尔线程 MUSA 正确性与性能

在 MTT S5000 上，使用与 NVIDIA 测试相同的 `DeepSeek-R1-Distill-Qwen-1.5B`、`Who are you?` 提示词和贪心解码。PyTorch 使用服务器预装的 `torch_musa`，通过 `torch.device("cuda")` 映射到 `musa:0`；LLAISYS 使用 MUSA 共享库和原有 Python ctypes 前端。

首次短测同时包含 PyTorch/MUSA runtime 的初始化开销，因此短长度的加速比主要用于验证趋势，不作为稳定吞吐结论。

| 最大新生成 token | PyTorch 时间 | LLAISYS 时间 | PyTorch 吞吐 | LLAISYS 吞吐 | 加速比 | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 8 | 10.46 s | 0.11 s | 0.76 tok/s | 72.73 tok/s | 95.09x | 是 |
| 32 | 10.96 s | 0.44 s | 2.92 tok/s | 72.73 tok/s | 24.91x | 是 |
| 128 | 12.09 s | 1.16 s | 10.59 tok/s | 110.34 tok/s | 10.42x | 是 |

为测试长上下文，另写了临时固定长度 benchmark，关闭 EOS 提前终止，强制生成 2048 个新 token，并使用相同的贪心 argmax 解码。结果如下：

| 新生成 token | PyTorch 时间 | LLAISYS 时间 | PyTorch 吞吐 | LLAISYS 吞吐 | LLAISYS/PyTorch 速度 | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2048 | 54.925 s | 74.211 s | 37.29 tok/s | 27.60 tok/s | 0.74x（慢 1.35 倍） | 是 |

这说明 MUSA 后端在短序列上受益于较低的 Python/框架调度开销，但长到 2048 后，当前 self-attention 对 KV Cache 的逐步读取成为瓶颈，LLAISYS 尚未超过 `torch_musa`。此前 `test_infer.py` 的 2048 参数会因模型提前生成 EOS 而实际只生成约 128 个 token，因此不能作为 2048 长度测试。

另外，Runtime 冒烟测试在 MTT S5000 上识别到 1 个设备，完成设备内存分配、Host-to-Device 拷贝、GPU `add` kernel 和 Device-to-Host 拷贝，输出为 `[11, 22, 33, 44, 55, 66]`。

MUSA 的线性层使用 muBLAS。MUSA 4.3.5 的 `mublasGemmEx` 导出为 C++ name-mangled 符号，因此加载器按该 SDK 的实际导出符号加载；`mublasCreate`、`mublasSgemm` 和 GEMM Ex 均已通过模型推理验证。xmake 构建和安装后的共享库又完成了 32 token 端到端一致性测试。

## 12. 结论

作业 1-3 已完成 Tensor、CPU 算子和 Qwen2 C/C++ 推理链路；作业 4 已完成 NVIDIA Runtime API、CUDA 算子、KV Cache decode 路径和针对生成阶段的性能优化，并在摩尔线程 MUSA 上完成第二个平台的适配与验证。

在 RTX 5090 上，LLAISYS 可以复现 PyTorch 的 token 输出。在 2048 个新 token 的长序列测试中，LLAISYS 达到 130.97 tok/s，PyTorch Transformers 达到 75.97 tok/s，LLAISYS 加速约 1.72 倍。在 MTT S5000 上，8、32、128 和固定 2048 token 测试均复现了 PyTorch 的 token 输出；其中 2048 token 时 LLAISYS 为 27.60 tok/s，PyTorch 为 37.29 tok/s，仍有约 1.35 倍减速。

CI 的最终状态需要在 Pull Request 页面查看；本报告记录的是本地 Windows、远程 Linux NVIDIA 以及远程 Linux 摩尔线程环境中的实际复现结果。
