# LLAISYS 作业 1-4 实验报告

## 1. 实验概述

本文记录 Tensor、CPU 算子、Qwen2 推理、NVIDIA CUDA 后端以及摩尔线程 MUSA 后端的实现、复现流程和实验结果。

测试模型为 `DeepSeek-R1-Distill-Qwen-1.5B`。模型权重通过 Hugging Face 镜像下载，推理结果使用贪心解码，并与 PyTorch Transformers 实现进行对比。

## 2. 环境

### Windows 本地环境

- 系统：Windows x64
- 主要验证内容：Tensor、CPU 算子、模型推理接口

### Linux NVIDIA 服务器

- CPU：Intel Xeon Platinum 8358P
- 内存：48 GB
- GPU：NVIDIA GeForce RTX 5090，32 GB 显存
- CUDA：13.3

### Linux 摩尔线程服务器

- 系统：Linux x86_64
- CPU：Intel Xeon Platinum 8358P
- 内存：48 GB
- GPU：摩尔线程 MTT S5000，80 GB 显存
- 驱动：3.3.5-server
- MUSA：4.3.5
- MUSA 架构参数：`mp_31`

## 3. 作业 1：Tensor

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

## 4. 作业 2：CPU 算子

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

## 5. 作业 3：Qwen2 推理

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

优化后的 CPU 端到端推理结果如下：

| 实现 | 推理耗时 |
| --- | ---: | ---: |
| LLAISYS CPU | 39.79 s |
| PyTorch CPU | 53.26 s |

LLAISYS CPU 平均耗时为 39.79 s，相较 PyTorch CPU 快约 **1.35 倍**。生成 token 与 PyTorch 完全一致。

## 6. 作业 4：NVIDIA CUDA 后端

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

### 6.1 RoPE 频率预计算

为每个 `(device, head_dim, theta)` 组合预计算并缓存 GPU 端的 `inv_freq`。首次使用该组合时，由 CUDA kernel 计算频率表：

```text
inv_freq[i] = 1 / powf(theta, 2 * i / head_dim)
angle = position * inv_freq[i]
```

后续 Q/K 的 RoPE 调用直接复用该频率表，只计算 `position * inv_freq[i]` 对应的旋转角度。

### 6.2 RMSNorm warp-level reduction

平方和归约采用 warp-level reduction：每个线程计算局部平方和，使用 `__shfl_down_sync` 完成 warp 内归约；每个 warp 仅写入一个部分和，再由第一个 warp 完成最终归约，减少共享内存访问和同步次数。

### 6.3 Decode 专用 self-attention

当 `q_len == 1` 时使用单独的 decode kernel：每个 query head 对应一个 CUDA block，使用 128 个线程，采用 warp-level max/sum reduction，并按 KV tile 执行 online softmax，避免保存完整 attention 矩阵。`q_len > 1` 时继续使用通用 prefill kernel。

### 6.4 资源复用

模型执行器增加 workspace，对相同 shape 和 dtype 的临时 Tensor 进行复用，减少 decode 阶段重复的 `cudaMalloc/cudaFree`。linear 算子使用 cuBLAS，并缓存每个线程的 cuBLAS handle，避免每次调用都创建和销毁 handle。

## 7. NVIDIA 正确性测试

远程 RTX 5090 上已验证：

```text
rope: passed, F32/F16/BF16
rms_norm: passed, F32/F16/BF16
self_attention: custom decode/prefill test passed
add: passed
linear: passed
```

## 8. NVIDIA 端到端性能

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

## 9. 摩尔线程 MUSA 正确性与性能

在 MTT S5000 上，使用与 NVIDIA 测试相同的 `DeepSeek-R1-Distill-Qwen-1.5B`、`Who are you?` 提示词和贪心解码。PyTorch 使用服务器预装的 `torch_musa`，通过 `torch.device("cuda")` 映射到 `musa:0`；LLAISYS 使用 MUSA 共享库和原有 Python ctypes 前端。

为与 NVIDIA 端到端测试保持一致，固定生成 `128`、`192`、`240` 和 `2048` 个新 token，关闭 EOS 提前终止并使用贪心 argmax 解码。模型加载不计入计时，计时覆盖 prefill 和后续 decode；每组均逐 token 对比两边输出。

| 新生成 token | PyTorch 时间 | LLAISYS 时间 | PyTorch 吞吐 | LLAISYS 吞吐 | 加速比 | 输出一致 |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 128 | 11.205 s | 1.895 s | 11.42 tok/s | 67.55 tok/s | 5.91x | 是 |
| 192 | 12.703 s | 2.974 s | 15.11 tok/s | 64.56 tok/s | 4.27x | 是 |
| 240 | 13.487 s | 3.857 s | 17.79 tok/s | 62.23 tok/s | 3.50x | 是 |
| 2048 | 52.881 s | 74.170 s | 38.73 tok/s | 27.61 tok/s | 0.71x（慢 1.40 倍） | 是 |

在 128 到 240 token 范围内，LLAISYS 的端到端吞吐为 62.23 到 67.55 tok/s，相比 `torch_musa` 快 3.50 到 5.91 倍。序列增长到 2048 token 后，当前 self-attention 对 KV Cache 的逐步读取成为瓶颈，LLAISYS 为 27.61 tok/s，尚未超过 `torch_musa` 的 38.73 tok/s。

## 10. 平台支持状态

| 平台 | 后端 | 状态 | 说明 |
| --- | --- | --- | --- |
| Windows x64 | CPU | 已验证 | 作业 1-3 的 CPU 构建和测试通过。 |
| Linux x64 | NVIDIA CUDA 13.3 | 已验证 | 在 RTX 5090 上完成构建、算子测试和端到端推理。 |
| Linux x64 | 摩尔线程 MUSA 4.3.5 | 已验证 | MTT S5000 上完成 Runtime、8 个 MUSA kernel、共享库、端到端推理和 token 一致性测试。 |

