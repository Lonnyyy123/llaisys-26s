#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../core/llaisys_core.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../utils/check.hpp"

#include <cmath>
#include <memory>
#include <vector>

namespace {

llaisysTensor_t createTensor(
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id) {
    return new LlaisysTensor{
        llaisys::Tensor::create(shape, dtype, device, device_id)};
}

void destroyTensor(llaisysTensor_t tensor) {
    delete tensor;
}

void destroyTensorVector(std::vector<llaisysTensor_t> &tensors) {
    for (auto *tensor : tensors) {
        destroyTensor(tensor);
    }
    tensors.clear();
}

} // namespace

struct LlaisysQwen2Model {
    struct WorkspaceEntry {
        std::vector<size_t> shape;
        llaisysDataType_t dtype;
        std::vector<llaisys::tensor_t> tensors;
        size_t used = 0;
    };

    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    std::vector<int> device_ids;
    LlaisysQwen2Weights weights{};

    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;

    size_t past_len = 0;
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;

    llaisys::tensor_t input_ids;
    llaisys::tensor_t position_ids;
    llaisys::tensor_t hidden_states;
    llaisys::tensor_t normed_hidden_states;
    std::vector<WorkspaceEntry> workspace;

    ~LlaisysQwen2Model() {
        destroyTensor(weights.in_embed);
        destroyTensor(weights.out_embed);
        destroyTensor(weights.out_norm_w);
        destroyTensorVector(attn_norm_w);
        destroyTensorVector(attn_q_w);
        destroyTensorVector(attn_q_b);
        destroyTensorVector(attn_k_w);
        destroyTensorVector(attn_k_b);
        destroyTensorVector(attn_v_w);
        destroyTensorVector(attn_v_b);
        destroyTensorVector(attn_o_w);
        destroyTensorVector(mlp_norm_w);
        destroyTensorVector(mlp_gate_w);
        destroyTensorVector(mlp_up_w);
        destroyTensorVector(mlp_down_w);
    }
};

namespace {

void createLayerWeights(
    std::vector<llaisysTensor_t> &tensors,
    size_t nlayer,
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id) {
    tensors.resize(nlayer);
    for (auto &tensor : tensors) {
        tensor = createTensor(shape, dtype, device, device_id);
    }
}

void exposeLayerWeights(LlaisysQwen2Model *model) {
    model->weights.attn_norm_w = model->attn_norm_w.data();
    model->weights.attn_q_w = model->attn_q_w.data();
    model->weights.attn_q_b = model->attn_q_b.data();
    model->weights.attn_k_w = model->attn_k_w.data();
    model->weights.attn_k_b = model->attn_k_b.data();
    model->weights.attn_v_w = model->attn_v_w.data();
    model->weights.attn_v_b = model->attn_v_b.data();
    model->weights.attn_o_w = model->attn_o_w.data();
    model->weights.mlp_norm_w = model->mlp_norm_w.data();
    model->weights.mlp_gate_w = model->mlp_gate_w.data();
    model->weights.mlp_up_w = model->mlp_up_w.data();
    model->weights.mlp_down_w = model->mlp_down_w.data();
}

void validateMeta(const LlaisysQwen2Meta *meta, int ndevice) {
    CHECK_ARGUMENT(meta != nullptr, "Qwen2: meta cannot be null");
    CHECK_ARGUMENT(ndevice == 1, "Qwen2: only one device is supported for now");
    CHECK_ARGUMENT(meta->nlayer > 0, "Qwen2: nlayer must be positive");
    CHECK_ARGUMENT(meta->hs > 0, "Qwen2: hidden size must be positive");
    CHECK_ARGUMENT(meta->nh > 0, "Qwen2: number of attention heads must be positive");
    CHECK_ARGUMENT(meta->nkvh > 0, "Qwen2: number of KV heads must be positive");
    CHECK_ARGUMENT(meta->dh > 0, "Qwen2: head dimension must be positive");
    CHECK_ARGUMENT(meta->di > 0, "Qwen2: intermediate size must be positive");
    CHECK_ARGUMENT(meta->voc > 0, "Qwen2: vocab size must be positive");
    CHECK_ARGUMENT(meta->nh % meta->nkvh == 0, "Qwen2: attention heads must be divisible by KV heads");
    CHECK_ARGUMENT(meta->nh * meta->dh == meta->hs, "Qwen2: nh * dh must equal hidden size");
}

LlaisysQwen2Model *checkedModel(LlaisysQwen2Model *model) {
    CHECK_ARGUMENT(model != nullptr, "Qwen2: model cannot be null");
    return model;
}

void validateInferArgs(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    checkedModel(model);
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2: token ids cannot be null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2: ntoken must be positive");
    CHECK_ARGUMENT(ntoken <= model->meta.maxseq, "Qwen2: ntoken exceeds max sequence length");
}

int modelDeviceId(const LlaisysQwen2Model *model) {
    return model->device_ids.empty() ? 0 : model->device_ids[0];
}

llaisys::tensor_t allocateModelTensor(
    const LlaisysQwen2Model *model,
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype) {
    return llaisys::Tensor::create(
        shape,
        dtype,
        model->device,
        modelDeviceId(model));
}

void resetWorkspace(LlaisysQwen2Model *model) {
    for (auto &entry : model->workspace) {
        entry.used = 0;
    }
}

llaisys::tensor_t createModelTensor(
    LlaisysQwen2Model *model,
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype) {
    for (auto &entry : model->workspace) {
        if (entry.dtype == dtype && entry.shape == shape) {
            if (entry.used == entry.tensors.size()) {
                entry.tensors.push_back(allocateModelTensor(model, shape, dtype));
            }
            return entry.tensors[entry.used++];
        }
    }

    LlaisysQwen2Model::WorkspaceEntry entry;
    entry.shape = shape;
    entry.dtype = dtype;
    entry.tensors.push_back(allocateModelTensor(model, shape, dtype));
    entry.used = 1;
    model->workspace.push_back(std::move(entry));
    return model->workspace.back().tensors.front();
}

void runEmbedding(LlaisysQwen2Model *model, const int64_t *token_ids, size_t ntoken) {
    model->input_ids = createModelTensor(model, {ntoken}, LLAISYS_DTYPE_I64);
    model->input_ids->load(token_ids);

    model->hidden_states = createModelTensor(
        model,
        {ntoken, model->meta.hs},
        model->meta.dtype);

    llaisys::ops::embedding(
        model->hidden_states,
        model->input_ids,
        model->weights.in_embed->tensor);
}

void buildPositionIds(LlaisysQwen2Model *model, size_t past_len, size_t ntoken) {
    std::vector<int64_t> pos_ids(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        pos_ids[i] = static_cast<int64_t>(past_len + i);
    }
    model->position_ids = createModelTensor(model, {ntoken}, LLAISYS_DTYPE_I64);
    model->position_ids->load(pos_ids.data());
}

void copyTensor(llaisys::tensor_t dst, llaisys::tensor_t src) {
    CHECK_SAME_DEVICE(dst, src);
    CHECK_SAME_DTYPE(dst->dtype(), src->dtype());
    CHECK_SAME_SHAPE(dst->shape(), src->shape());
    CHECK_ARGUMENT(
        dst->isContiguous() && src->isContiguous(),
        "Qwen2: copy tensors must be contiguous");

    const size_t bytes = src->numel() * src->elementSize();
    if (bytes == 0) {
        return;
    }

    llaisys::core::context().setDevice(dst->deviceType(), dst->deviceId());
    const auto kind = dst->deviceType() == LLAISYS_DEVICE_CPU
                        ? LLAISYS_MEMCPY_H2H
                        : LLAISYS_MEMCPY_D2D;
    llaisys::core::context().runtime().api()->memcpy_sync(
        dst->data(),
        src->data(),
        bytes,
        kind);
}

llaisys::tensor_t runRMSNorm(
    LlaisysQwen2Model *model,
    llaisys::tensor_t input,
    llaisys::tensor_t weight) {
    auto output = createModelTensor(
        model,
        input->shape(),
        model->meta.dtype);

    llaisys::ops::rms_norm(
        output,
        input,
        weight,
        model->meta.epsilon);
    return output;
}

llaisys::tensor_t runLinear(
    LlaisysQwen2Model *model,
    llaisys::tensor_t input,
    llaisys::tensor_t weight,
    llaisys::tensor_t bias = nullptr) {
    auto output = createModelTensor(
        model,
        {input->shape()[0], weight->shape()[0]},
        model->meta.dtype);

    llaisys::ops::linear(output, input, weight, bias);
    return output;
}

llaisys::tensor_t runAdd(
    LlaisysQwen2Model *model,
    llaisys::tensor_t a,
    llaisys::tensor_t b) {
    auto output = createModelTensor(model, a->shape(), model->meta.dtype);
    llaisys::ops::add(output, a, b);
    return output;
}

void runSelfAttentionBlock(LlaisysQwen2Model *model, size_t layer, size_t past_len) {
    const size_t seq_len = model->hidden_states->shape()[0];
    const size_t q_dim = model->meta.nh * model->meta.dh;
    const size_t kv_len = past_len + seq_len;

    model->normed_hidden_states = runRMSNorm(
        model,
        model->hidden_states,
        model->weights.attn_norm_w[layer]->tensor);

    auto q_proj = runLinear(
        model,
        model->normed_hidden_states,
        model->weights.attn_q_w[layer]->tensor,
        model->weights.attn_q_b[layer]->tensor);
    auto k_proj = runLinear(
        model,
        model->normed_hidden_states,
        model->weights.attn_k_w[layer]->tensor,
        model->weights.attn_k_b[layer]->tensor);
    auto v_proj = runLinear(
        model,
        model->normed_hidden_states,
        model->weights.attn_v_w[layer]->tensor,
        model->weights.attn_v_b[layer]->tensor);

    auto q = q_proj->view({seq_len, model->meta.nh, model->meta.dh});
    auto k = k_proj->view({seq_len, model->meta.nkvh, model->meta.dh});
    auto v = v_proj->view({seq_len, model->meta.nkvh, model->meta.dh});

    auto q_rope = createModelTensor(
        model,
        {seq_len, model->meta.nh, model->meta.dh},
        model->meta.dtype);
    auto k_rope = createModelTensor(
        model,
        {seq_len, model->meta.nkvh, model->meta.dh},
        model->meta.dtype);

    llaisys::ops::rope(q_rope, q, model->position_ids, model->meta.theta);
    llaisys::ops::rope(k_rope, k, model->position_ids, model->meta.theta);

    auto k_cache_update = model->k_cache[layer]->slice(0, past_len, kv_len);
    auto v_cache_update = model->v_cache[layer]->slice(0, past_len, kv_len);
    copyTensor(k_cache_update, k_rope);
    copyTensor(v_cache_update, v);

    auto attn_values = createModelTensor(
        model,
        {seq_len, model->meta.nh, model->meta.dh},
        model->meta.dtype);
    auto k_context = model->k_cache[layer]->slice(0, 0, kv_len);
    auto v_context = model->v_cache[layer]->slice(0, 0, kv_len);
    const float scale = 1.0F / std::sqrt(static_cast<float>(model->meta.dh));
    llaisys::ops::self_attention(attn_values, q_rope, k_context, v_context, scale);

    auto attn_output = runLinear(
        model,
        attn_values->view({seq_len, q_dim}),
        model->weights.attn_o_w[layer]->tensor);
    model->hidden_states = runAdd(model, model->hidden_states, attn_output);
}

void runMLPBlock(LlaisysQwen2Model *model, size_t layer) {
    const size_t seq_len = model->hidden_states->shape()[0];

    model->normed_hidden_states = runRMSNorm(
        model,
        model->hidden_states,
        model->weights.mlp_norm_w[layer]->tensor);

    auto gate = runLinear(
        model,
        model->normed_hidden_states,
        model->weights.mlp_gate_w[layer]->tensor);
    auto up = runLinear(
        model,
        model->normed_hidden_states,
        model->weights.mlp_up_w[layer]->tensor);
    auto activated = createModelTensor(
        model,
        {seq_len, model->meta.di},
        model->meta.dtype);
    llaisys::ops::swiglu(activated, gate, up);

    auto mlp_output = runLinear(
        model,
        activated,
        model->weights.mlp_down_w[layer]->tensor);
    model->hidden_states = runAdd(model, model->hidden_states, mlp_output);
}

int64_t readScalarI64(llaisys::tensor_t tensor) {
    if (tensor->deviceType() == LLAISYS_DEVICE_CPU) {
        return *reinterpret_cast<const int64_t *>(tensor->data());
    }

    auto host_tensor = tensor->to(LLAISYS_DEVICE_CPU, 0);
    return *reinterpret_cast<const int64_t *>(host_tensor->data());
}

int64_t runFinalHead(LlaisysQwen2Model *model) {
    const size_t seq_len = model->hidden_states->shape()[0];
    auto final_hidden = runRMSNorm(
        model,
        model->hidden_states,
        model->weights.out_norm_w->tensor);
    auto last_hidden = final_hidden->slice(0, seq_len - 1, seq_len);
    auto logits = runLinear(
        model,
        last_hidden,
        model->weights.out_embed->tensor);

    auto max_idx = createModelTensor(model, {1}, LLAISYS_DTYPE_I64);
    auto max_val = createModelTensor(model, {1}, model->meta.dtype);
    llaisys::ops::argmax(max_idx, max_val, logits->view({model->meta.voc}));
    return readScalarI64(max_idx);
}

int64_t runInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
    validateInferArgs(model, token_ids, ntoken);
    if (ntoken > 1) {
        model->past_len = 0;
    }
    CHECK_ARGUMENT(
        model->past_len + ntoken <= model->meta.maxseq,
        "Qwen2: KV cache capacity exceeded");

    const size_t past_len = model->past_len;
    resetWorkspace(model);
    runEmbedding(model, token_ids, ntoken);
    buildPositionIds(model, past_len, ntoken);
    for (size_t layer = 0; layer < model->meta.nlayer; ++layer) {
        runSelfAttentionBlock(model, layer, past_len);
        runMLPBlock(model, layer);
    }
    model->past_len += ntoken;
    return runFinalHead(model);
}

} // namespace

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        validateMeta(meta, ndevice);

        const int device_id = device_ids == nullptr ? 0 : device_ids[0];
        auto model = std::make_unique<LlaisysQwen2Model>();
        model->meta = *meta;
        model->device = device;
        model->device_ids.push_back(device_id);

        const auto dtype = meta->dtype;
        const auto nlayer = meta->nlayer;
        const auto hs = meta->hs;
        const auto q_dim = meta->nh * meta->dh;
        const auto kv_dim = meta->nkvh * meta->dh;

        model->weights.in_embed = createTensor({meta->voc, hs}, dtype, device, device_id);
        model->weights.out_embed = createTensor({meta->voc, hs}, dtype, device, device_id);
        model->weights.out_norm_w = createTensor({hs}, dtype, device, device_id);

        createLayerWeights(model->attn_norm_w, nlayer, {hs}, dtype, device, device_id);
        createLayerWeights(model->attn_q_w, nlayer, {q_dim, hs}, dtype, device, device_id);
        createLayerWeights(model->attn_q_b, nlayer, {q_dim}, dtype, device, device_id);
        createLayerWeights(model->attn_k_w, nlayer, {kv_dim, hs}, dtype, device, device_id);
        createLayerWeights(model->attn_k_b, nlayer, {kv_dim}, dtype, device, device_id);
        createLayerWeights(model->attn_v_w, nlayer, {kv_dim, hs}, dtype, device, device_id);
        createLayerWeights(model->attn_v_b, nlayer, {kv_dim}, dtype, device, device_id);
        createLayerWeights(model->attn_o_w, nlayer, {hs, q_dim}, dtype, device, device_id);
        createLayerWeights(model->mlp_norm_w, nlayer, {hs}, dtype, device, device_id);
        createLayerWeights(model->mlp_gate_w, nlayer, {meta->di, hs}, dtype, device, device_id);
        createLayerWeights(model->mlp_up_w, nlayer, {meta->di, hs}, dtype, device, device_id);
        createLayerWeights(model->mlp_down_w, nlayer, {hs, meta->di}, dtype, device, device_id);
        exposeLayerWeights(model.get());

        model->k_cache.reserve(nlayer);
        model->v_cache.reserve(nlayer);
        for (size_t layer = 0; layer < nlayer; ++layer) {
            model->k_cache.push_back(llaisys::Tensor::create(
                {meta->maxseq, meta->nkvh, meta->dh},
                dtype,
                device,
                device_id));
            model->v_cache.push_back(llaisys::Tensor::create(
                {meta->maxseq, meta->nkvh, meta->dh},
                dtype,
                device,
                device_id));
        }

        return model.release();
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return &checkedModel(model)->weights;
    }

    int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model *model,
        int64_t *token_ids,
        size_t ntoken) {
        return runInfer(model, token_ids, ntoken);
    }
}
