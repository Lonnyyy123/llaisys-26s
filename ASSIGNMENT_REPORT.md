# LLAISYS 作业 1-3 实验报告

## 概述

本文记录作业 1、作业 2、作业 3 的实现内容、复现流程、复现结果以及平台支持状态。

- 作业 1 实现 Tensor 的数据加载、连续性判断和元数据变换。
- 作业 2 实现推理所需的 CPU 算子。
- 作业 3 实现 `DeepSeek-R1-Distill-Qwen-1.5B` 的 Qwen2 推理流程，并支持 KV Cache 增量解码。

以下本地验证均在 Windows CPU 后端完成。

## 实验环境

- 操作系统：Windows
- Python：`.venv\Scripts\python.exe`
- 构建系统：`xmake`
- 测试模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 本地模型路径：

```powershell
C:\Users\22583\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
```

## 复现流程

构建并安装动态库：

```powershell
xmake build llaisys
xmake install llaisys
```

开发阶段使用本仓库中的 Python 源码：

```powershell
$env:PYTHONPATH = (Resolve-Path 'python').Path
```

运行作业 1 Tensor 测试：

```powershell
.\.venv\Scripts\python.exe test\test_tensor.py
```

运行作业 2 算子测试：

```powershell
.\.venv\Scripts\python.exe test\ops\argmax.py --device cpu
.\.venv\Scripts\python.exe test\ops\embedding.py --device cpu
.\.venv\Scripts\python.exe test\ops\linear.py --device cpu
.\.venv\Scripts\python.exe test\ops\rms_norm.py --device cpu
.\.venv\Scripts\python.exe test\ops\rope.py --device cpu
.\.venv\Scripts\python.exe test\ops\self_attention.py --device cpu
.\.venv\Scripts\python.exe test\ops\swiglu.py --device cpu
```

运行作业 3 Qwen2 推理测试：

```powershell
.\.venv\Scripts\python.exe test\test_infer.py --model "C:\Users\22583\.cache\huggingface\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B\snapshots\ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" --test --max_steps 8
```

## 作业 1：Tensor

实现内容：

- `Tensor::load`
- `Tensor::isContiguous`
- `Tensor::view`
- `Tensor::permute`
- `Tensor::slice`
- challenging 部分的 `contiguous`、`reshape` 和设备间拷贝支持

其中，多个 Tensor 元数据操作支持非连续输入：

- `permute` 不移动数据，只重排 shape 和 strides，可产生非连续 tensor。
- `slice` 不移动数据，只调整 shape 和 storage offset，可产生非连续 tensor。
- `view` 会根据原 tensor 的 shape 和 strides 判断是否能在不搬移数据的情况下重新解释形状；如果 stride 不兼容，会抛出错误。
- `reshape` 会优先尝试走零拷贝 view；如果原 tensor 非连续且不能直接 view，则先通过 `contiguous` 生成连续副本，再 reshape。
- `contiguous` 可以把非连续 tensor 拷贝成连续 tensor。

当前限制：

- `Tensor::load` 只支持加载到 contiguous tensor。
- 作业 2 中实现的算子目前也要求输入输出 tensor contiguous。

本地结果：

```text
test/test_tensor.py: passed
```

## 作业 2：算子

实现了以下 CPU 算子：

- `argmax`
- `embedding`
- `linear`
- `rms_norm`
- `rope`
- `self_attention`
- `swiglu`

浮点算子支持以下数据类型：

- `F32`
- `F16`
- `BF16`

本地结果：

```text
test/ops/argmax.py: passed
test/ops/embedding.py: passed
test/ops/linear.py: passed
test/ops/rms_norm.py: passed
test/ops/rope.py: passed
test/ops/self_attention.py: passed
test/ops/swiglu.py: passed
```

## 作业 3：大语言模型推理

实现了 Qwen2 模型推理，模型逻辑在 C/C++ 后端执行。

Python 前端负责：

- 读取 `config.json`
- 读取 safetensors 元数据和原始权重字节
- 通过 ctypes 调用 C API
- 将权重 buffer 和 token ids 传入后端
- 在测试脚本中使用 Hugging Face tokenizer 完成 decode

C/C++ 后端负责：

- 创建和销毁 Qwen2 模型对象
- 分配模型权重 tensor
- 通过 `tensorLoad` 加载权重
- 执行 Transformer forward
- 维护和更新 KV Cache
- 计算最终 logits 并通过 argmax 选择下一个 token

涉及的主要文件：

- `python/llaisys/libllaisys/qwen2.py`
- `python/llaisys/models/qwen2.py`
- `src/llaisys/qwen2.cc`

推理流程：

```text
token ids
  -> embedding
  -> 每层 attention RMSNorm
  -> q/k/v projection
  -> 对 q/k 执行 RoPE
  -> 更新 KV Cache
  -> 基于缓存 K/V 执行 causal self-attention
  -> attention output projection
  -> residual add
  -> MLP RMSNorm
  -> gate/up projection
  -> SwiGLU
  -> down projection
  -> residual add
  -> final RMSNorm
  -> lm_head
  -> argmax
```

KV Cache 行为：

- 第一次 `infer` 接收完整 prompt，并填充每一层的 K/V cache。
- 后续 decode 阶段每次只接收最新生成的一个 token。
- 每层维护 `k_cache` 和 `v_cache`，形状为 `[maxseq, num_kv_heads, head_dim]`。
- RoPE position id 会根据当前 cache 长度进行偏移。

本地 `--test --max_steps 8` 结果：

```text
PyTorch tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16]

LLAISYS tokens:
[151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16]

结果：passed
```

本地 `--test --max_steps 8` 计时：

```text
PyTorch: 56.49s
LLAISYS: 128.27s
```

当前实现可以在测试的 argmax 解码路径上复现 PyTorch 的 token 输出。CPU 性能仍慢于 PyTorch，后续可以继续优化内存复用、算子实现和 GPU 后端。

## 平台支持状态

| 平台 | 后端 | 状态 | 说明 |
| --- | --- | --- | --- |
| Windows x64 | CPU | 已支持并本地测试 | 作业 1-3 的 CPU 路径本地测试通过。 |
| Windows x64 | NVIDIA CUDA | 未完成 | 当前环境未检测到可用 CUDA，且多个 NVIDIA 算子分支仍未实现。 |
| Linux | CPU | 预期可构建 | 本报告未在 Linux 本地验证。 |
| Linux | NVIDIA CUDA | 未验证 | 需要 CUDA 工具链和完整 GPU 算子实现。 |

## CI 状态

CI 状态需要在 Pull Request 推送后查看。当前本地 CPU 构建和测试结果如上所述。
