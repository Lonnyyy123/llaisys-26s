from typing import Sequence
from ..libllaisys import LIB_LLAISYS, DataType, LlaisysQwen2Meta, load_qwen2
from ..libllaisys import DeviceType

import json
import math
import os
from pathlib import Path
import safetensors
from ctypes import byref, c_char, c_int, c_int64, c_size_t, c_void_p, cast


_QWEN2_API_LOADED = False


def _load_qwen2_api():
    global _QWEN2_API_LOADED
    if _QWEN2_API_LOADED:
        return
    load_qwen2(LIB_LLAISYS)
    _QWEN2_API_LOADED = True


def _dtype_from_config(config):
    dtype = str(config.get("torch_dtype", "float32")).lower()
    if dtype in ("bfloat16", "bf16"):
        return DataType.BF16
    if dtype in ("float16", "fp16", "half"):
        return DataType.F16
    if dtype in ("float32", "fp32", "float"):
        return DataType.F32
    raise ValueError(f"Unsupported Qwen2 dtype: {dtype}")


def _single_token_id(value, name):
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} cannot be empty")
        value = value[0]
    return int(value)


def _meta_from_config(config):
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", hidden_size // num_heads))
    config_maxseq = int(config["max_position_embeddings"])
    cache_maxseq = int(os.environ.get("LLAISYS_MAX_SEQ_LEN", min(config_maxseq, 2048)))
    if cache_maxseq <= 0:
        raise ValueError("LLAISYS_MAX_SEQ_LEN must be positive")

    return LlaisysQwen2Meta(
        _dtype_from_config(config),
        int(config["num_hidden_layers"]),
        hidden_size,
        num_heads,
        int(config.get("num_key_value_heads", num_heads)),
        head_dim,
        int(config["intermediate_size"]),
        min(cache_maxseq, config_maxseq),
        int(config["vocab_size"]),
        float(config.get("rms_norm_eps", 1e-6)),
        float(config.get("rope_theta", 10000.0)),
        _single_token_id(config.get("eos_token_id", -1), "eos_token_id"),
    )


_SAFETENSORS_DTYPE_SIZE = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _read_safetensors_header(path):
    with path.open("rb") as f:
        header_size = int.from_bytes(f.read(8), byteorder="little")
        header = json.loads(f.read(header_size).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + header_size


def _tensor_shape(tensor):
    ndim = int(LIB_LLAISYS.tensorGetNdim(tensor))
    shape = (c_size_t * ndim)()
    LIB_LLAISYS.tensorGetShape(tensor, shape)
    return tuple(int(shape[i]) for i in range(ndim))


def _load_safetensors_entry(weight_file, data_start, entry, dst_tensor, name):
    expected_shape = _tensor_shape(dst_tensor)
    actual_shape = tuple(int(dim) for dim in entry["shape"])
    if actual_shape != expected_shape:
        raise ValueError(
            f"Shape mismatch for {name}: safetensors has {actual_shape}, "
            f"backend tensor expects {expected_shape}"
        )

    dtype = entry["dtype"]
    if dtype not in _SAFETENSORS_DTYPE_SIZE:
        raise ValueError(f"Unsupported safetensors dtype for {name}: {dtype}")

    start, end = (int(x) for x in entry["data_offsets"])
    nbytes = end - start
    expected_bytes = math.prod(actual_shape) * _SAFETENSORS_DTYPE_SIZE[dtype]
    if nbytes != expected_bytes:
        raise ValueError(
            f"Byte size mismatch for {name}: safetensors has {nbytes}, "
            f"shape/dtype imply {expected_bytes}"
        )

    raw = bytearray(nbytes)
    weight_file.seek(data_start + start)
    read_bytes = weight_file.readinto(raw)
    if read_bytes != nbytes:
        raise IOError(f"Failed to read all bytes for {name}: {read_bytes}/{nbytes}")

    src = (c_char * nbytes).from_buffer(raw)
    LIB_LLAISYS.tensorLoad(dst_tensor, cast(src, c_void_p))


def _layer_index(name):
    parts = name.split(".")
    if len(parts) < 4 or parts[:2] != ["model", "layers"]:
        return None
    return int(parts[2]), ".".join(parts[3:])


def _weight_target(weights, name):
    if name == "model.embed_tokens.weight":
        return weights.in_embed
    if name == "lm_head.weight":
        return weights.out_embed
    if name == "model.norm.weight":
        return weights.out_norm_w

    layer = _layer_index(name)
    if layer is None:
        raise KeyError(f"Unknown Qwen2 weight: {name}")

    layer_id, suffix = layer
    layer_weights = {
        "input_layernorm.weight": weights.attn_norm_w,
        "self_attn.q_proj.weight": weights.attn_q_w,
        "self_attn.q_proj.bias": weights.attn_q_b,
        "self_attn.k_proj.weight": weights.attn_k_w,
        "self_attn.k_proj.bias": weights.attn_k_b,
        "self_attn.v_proj.weight": weights.attn_v_w,
        "self_attn.v_proj.bias": weights.attn_v_b,
        "self_attn.o_proj.weight": weights.attn_o_w,
        "post_attention_layernorm.weight": weights.mlp_norm_w,
        "mlp.gate_proj.weight": weights.mlp_gate_w,
        "mlp.up_proj.weight": weights.mlp_up_w,
        "mlp.down_proj.weight": weights.mlp_down_w,
    }
    if suffix not in layer_weights:
        raise KeyError(f"Unknown Qwen2 layer weight: {name}")
    return layer_weights[suffix][layer_id]


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        _load_qwen2_api()
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen2 config not found: {config_path}")

        with config_path.open("r", encoding="utf-8-sig") as f:
            config = json.load(f)

        self.model_path = model_path
        self.device = DeviceType(device)
        self.device_id = 0
        self.meta = _meta_from_config(config)
        self.maxseq = int(self.meta.maxseq)
        self.end_token = int(self.meta.end_token)

        device_ids = (c_int * 1)(self.device_id)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta),
            self.device,
            device_ids,
            c_int(1),
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")
        self._weights = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        for file in sorted(model_path.glob("*.safetensors")):
            entries, data_start = _read_safetensors_header(file)
            with file.open("rb") as weight_file:
                with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                    for name_ in data_.keys():
                        dst_tensor = _weight_target(self._weights, name_)
                        _load_safetensors_entry(
                            weight_file,
                            data_start,
                            entries[name_],
                            dst_tensor,
                            name_,
                        )

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):

        if max_new_tokens is None:
            max_new_tokens = max(0, self.maxseq - len(inputs))
        else:
            max_new_tokens = int(max_new_tokens)

        outputs = list(inputs)
        if max_new_tokens <= 0:
            return outputs

        model = self._model
        if model is None:
            raise RuntimeError("Qwen2 model has not been initialized")

        _load_qwen2_api()
        infer = LIB_LLAISYS.llaisysQwen2ModelInfer

        ntoken = len(outputs)
        prompt_buf = (c_int64 * ntoken)(*outputs)
        next_token = int(infer(model, prompt_buf, c_size_t(ntoken)))
        outputs.append(next_token)
        if next_token == self.end_token:
            return outputs

        for _ in range(max_new_tokens - 1):
            token_buf = (c_int64 * 1)(next_token)
            next_token = int(infer(model, token_buf, c_size_t(1)))
            outputs.append(next_token)
            if next_token == self.end_token:
                break

        return outputs
