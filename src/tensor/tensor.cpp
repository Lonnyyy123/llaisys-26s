#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <optional>
#include <sstream>

namespace llaisys {
namespace {

std::vector<ptrdiff_t> contiguousStrides(const std::vector<size_t> &shape) {
    std::vector<ptrdiff_t> strides(shape.size());
    ptrdiff_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= static_cast<ptrdiff_t>(shape[i]);
    }
    return strides;
}

std::optional<std::vector<ptrdiff_t>> computeViewStrides(
    const std::vector<size_t> &old_shape,
    const std::vector<ptrdiff_t> &old_strides,
    const std::vector<size_t> &new_shape,
    size_t numel) {
    if (old_shape.empty()) {
        return contiguousStrides(new_shape);
    }

    if (numel == 0) {
        if (old_shape == new_shape) {
            return old_strides;
        }
        return contiguousStrides(new_shape);
    }

    for (const ptrdiff_t stride : old_strides) {
        if (stride < 0) {
            return std::nullopt;
        }
    }

    std::vector<ptrdiff_t> new_strides(new_shape.size());
    ptrdiff_t view_dim = static_cast<ptrdiff_t>(new_shape.size()) - 1;
    ptrdiff_t chunk_base_stride = old_strides.back();
    size_t tensor_numel = 1;
    size_t view_numel = 1;

    for (ptrdiff_t tensor_dim = static_cast<ptrdiff_t>(old_shape.size()) - 1;
         tensor_dim >= 0;
         --tensor_dim) {
        tensor_numel *= old_shape[tensor_dim];

        const bool chunk_boundary =
            tensor_dim == 0 ||
            (old_shape[tensor_dim - 1] != 1 &&
             old_strides[tensor_dim - 1] !=
                 static_cast<ptrdiff_t>(tensor_numel) * chunk_base_stride);

        if (!chunk_boundary) {
            continue;
        }

        while (view_dim >= 0 &&
               (view_numel < tensor_numel || new_shape[view_dim] == 1)) {
            new_strides[view_dim] =
                static_cast<ptrdiff_t>(view_numel) * chunk_base_stride;
            view_numel *= new_shape[view_dim];
            --view_dim;
        }

        if (view_numel != tensor_numel) {
            return std::nullopt;
        }

        if (tensor_dim > 0) {
            chunk_base_stride = old_strides[tensor_dim - 1];
            tensor_numel = 1;
            view_numel = 1;
        }
    }

    if (view_dim != -1) {
        return std::nullopt;
    }

    return new_strides;
}

void copyTensorToHost(const Tensor &source, std::byte *host) {
    const size_t numel = source.numel();
    const size_t element_size = source.elementSize();
    const size_t bytes = numel * element_size;
    if (bytes == 0) {
        return;
    }

    ASSERT(host != nullptr, "host destination pointer cannot be null");

    const LlaisysRuntimeAPI *api = nullptr;
    if (source.deviceType() != LLAISYS_DEVICE_CPU) {
        core::context().setDevice(source.deviceType(), source.deviceId());
        api = core::context().runtime().api();
    }

    if (source.isContiguous()) {
        if (source.deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(host, source.data(), bytes);
        } else {
            api->memcpy_sync(
                host,
                source.data(),
                bytes,
                LLAISYS_MEMCPY_D2H);
        }
        return;
    }

    for (const ptrdiff_t stride : source.strides()) {
        CHECK_ARGUMENT(
            stride >= 0,
            "negative strides are not supported by contiguous");
    }

    std::vector<size_t> indices(source.ndim(), 0);
    for (size_t linear = 0; linear < numel; ++linear) {
        ptrdiff_t source_element_offset = 0;
        for (size_t dim = 0; dim < source.ndim(); ++dim) {
            source_element_offset +=
                static_cast<ptrdiff_t>(indices[dim]) * source.strides()[dim];
        }

        const std::byte *source_ptr =
            source.data() +
            source_element_offset * static_cast<ptrdiff_t>(element_size);
        std::byte *host_ptr = host + linear * element_size;

        if (source.deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(host_ptr, source_ptr, element_size);
        } else {
            api->memcpy_sync(
                host_ptr,
                source_ptr,
                element_size,
                LLAISYS_MEMCPY_D2H);
        }

        for (size_t dim = source.ndim(); dim-- > 0;) {
            ++indices[dim];
            if (indices[dim] < source.shape()[dim]) {
                break;
            }
            indices[dim] = 0;
        }
    }
}

void copyTensorToContiguous(const Tensor &source, Tensor &destination) {
    ASSERT(destination.isContiguous(), "destination tensor must be contiguous");
    ASSERT(
        source.numel() == destination.numel(),
        "source and destination element counts must match");
    ASSERT(
        source.dtype() == destination.dtype(),
        "source and destination data types must match");

    const size_t bytes = source.numel() * source.elementSize();
    if (bytes == 0) {
        return;
    }

    if (source.isContiguous()) {
        if (source.deviceType() == LLAISYS_DEVICE_CPU) {
            destination.load(source.data());
            return;
        }

        if (destination.deviceType() == LLAISYS_DEVICE_CPU) {
            core::context().setDevice(source.deviceType(), source.deviceId());
            core::context().runtime().api()->memcpy_sync(
                destination.data(),
                source.data(),
                bytes,
                LLAISYS_MEMCPY_D2H);
            return;
        }

        if (source.deviceType() == destination.deviceType() &&
            source.deviceId() == destination.deviceId()) {
            core::context().setDevice(
                destination.deviceType(),
                destination.deviceId());
            core::context().runtime().api()->memcpy_sync(
                destination.data(),
                source.data(),
                bytes,
                LLAISYS_MEMCPY_D2D);
            return;
        }
    }

    std::vector<std::byte> host(bytes);
    copyTensorToHost(source, host.data());
    destination.load(host.data());
}

} // namespace

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (this->numel() == 0) {
        return true;
    }

    ptrdiff_t expected = 1;
    for (size_t i = this->ndim(); i-- > 0;) {
        if (this->shape()[i] == 1) {
            continue;
        }
        if (this->strides()[i] != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(this->shape()[i]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    const auto &old_shape = this->shape();
    const auto &old_strides = this->strides();
    const size_t ndim = this->ndim();

    CHECK_ARGUMENT(
        order.size() == ndim,
        "permutation size must match tensor dimensions");

    std::vector<bool> visited(ndim, 0);

    std::vector<size_t> new_shape;
    std::vector<ptrdiff_t> new_strides;

    new_shape.reserve(ndim);
    new_strides.reserve(ndim);

    for (const size_t dim : order) {
        CHECK_ARGUMENT(
            dim < ndim,
            "permutation dimension is out of range");

        CHECK_ARGUMENT(
            !visited[dim],
            "permutation contains duplicate dimensions");

        visited[dim] = 1;

        new_shape.push_back(old_shape[dim]);
        new_strides.push_back(old_strides[dim]);
    }

    TensorMeta new_meta{
        this->dtype(),
        std::move(new_shape),
        std::move(new_strides),
    };

    return std::shared_ptr<Tensor>(new Tensor(std::move(new_meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    const size_t new_numel = std::accumulate(
        shape.begin(),
        shape.end(),
        size_t{1},
        std::multiplies<size_t>{});

    CHECK_ARGUMENT(
        new_numel == this->numel(),
        "unmatched tensor shape");

    auto new_strides =
        computeViewStrides(this->shape(), this->strides(), shape, this->numel());
    CHECK_ARGUMENT(
        new_strides.has_value(),
        "view shape is incompatible with tensor strides");

    TensorMeta new_meta{
        this->dtype(),
        shape,
        std::move(*new_strides),
    };

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(new_meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(
        dim < this->ndim(),
        "slice dimension is out of range");

    CHECK_ARGUMENT(
        start <= end,
        "slice start must not be greater than end");

    CHECK_ARGUMENT(
        end <= this->shape()[dim],
        "slice end is out of range");

    TensorMeta new_meta = _meta;

    new_meta.shape[dim] = end - start;

    const ptrdiff_t element_offset =
        static_cast<ptrdiff_t>(start) * _meta.strides[dim];

    CHECK_ARGUMENT(
        element_offset >= 0,
        "negative strides are not supported by slice");

    const size_t new_offset =
        _offset +
        static_cast<size_t>(element_offset) * this->elementSize();

    return std::shared_ptr<Tensor>(
        new Tensor(
            std::move(new_meta),
            _storage,
            new_offset));
}

void Tensor::load(const void *src_) {
    const size_t numel = this->numel();
    const size_t element_size = this->elementSize();
    const size_t bytes = numel * element_size;
    if (bytes == 0) {
        return;
    }

    CHECK_ARGUMENT(src_ != nullptr, "source pointer cannot be null");

    core::context().setDevice(this->deviceType(), this->deviceId());

    const auto kind = this->deviceType() == LLAISYS_DEVICE_CPU
                        ? LLAISYS_MEMCPY_H2H
                        : LLAISYS_MEMCPY_H2D;

    CHECK_ARGUMENT(
        this->isContiguous(),
        "load only supports contiguous tensors");

    core::context().runtime().api()->memcpy_sync(
        this->data(),
        src_,
        bytes,
        kind);
}

tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(
            new Tensor(_meta, _storage, _offset));
    }

    auto result = Tensor::create(
        this->shape(),
        this->dtype(),
        this->deviceType(),
        this->deviceId());
    copyTensorToContiguous(*this, *result);
    return result;
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    const size_t new_numel = std::accumulate(
        shape.begin(),
        shape.end(),
        size_t{1},
        std::multiplies<size_t>{});
    CHECK_ARGUMENT(
        new_numel == this->numel(),
        "unmatched tensor shape");

    auto new_strides =
        computeViewStrides(this->shape(), this->strides(), shape, this->numel());
    if (new_strides.has_value()) {
        TensorMeta new_meta{
            this->dtype(),
            shape,
            std::move(*new_strides),
        };
        return std::shared_ptr<Tensor>(
            new Tensor(std::move(new_meta), _storage, _offset));
    }

    return this->contiguous()->view(shape);
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    CHECK_ARGUMENT(
        device_type >= LLAISYS_DEVICE_CPU &&
            device_type < LLAISYS_DEVICE_TYPE_COUNT,
        "invalid device type");

    const int target_device =
        device < 0
        ? (device_type == this->deviceType() ? this->deviceId() : 0)
        : device;

    if (device_type == this->deviceType() &&
        target_device == this->deviceId()) {
        return std::shared_ptr<Tensor>(
            new Tensor(_meta, _storage, _offset));
    }

    auto result = Tensor::create(
        this->shape(),
        this->dtype(),
        device_type,
        target_device);
    copyTensorToContiguous(*this, *result);
    return result;
}

} // namespace llaisys
