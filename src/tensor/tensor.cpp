#include "tensor.hpp"

#include "../utils.hpp"
#include "llaisys.h"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

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

tensor_t Tensor::cat_two(const tensor_t a, const tensor_t b, int dim) {
    // Only support concatenation along dimension 0 for now
    if (dim != 0) {
        throw std::runtime_error("Only concatenation along dimension 0 is supported.");
    }

    // Check dimensions
    if (a->ndim() != b->ndim()) {
        throw std::runtime_error("Tensors must have the same number of dimensions to concatenate.");
    }

    // Check shapes for all dims except 0
    for (size_t i = 1; i < a->ndim(); ++i) {
        if (a->shape()[i] != b->shape()[i]) {
            throw std::runtime_error("Tensors must have the same shape in all dimensions except dimension 0.");
        }
    }

    // New shape
    std::vector<size_t> new_shape = a->shape();
    new_shape[0] += b->shape()[0];

    // Create new tensor
    auto result = Tensor::create(new_shape, a->dtype(), a->deviceType(), a->deviceId());

    // Calculate size to copy for a and b (in bytes)
    size_t a_bytes = a->numel() * a->elementSize();
    size_t b_bytes = b->numel() * b->elementSize();

    // Copy data from a
    if (a->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(result->data(), a->data(), a_bytes);
        std::memcpy(result->data() + a_bytes, b->data(), b_bytes);
    } else {
        core::context().setDevice(a->deviceType(), a->deviceId());
        core::context().runtime().api()->memcpy_sync(
            result->data(), a->data(), a_bytes, LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->memcpy_sync(
            result->data() + a_bytes, b->data(), b_bytes, LLAISYS_MEMCPY_D2D);
        core::context().runtime().api()->device_synchronize();
    }

    return result;
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
    const auto nd = this->ndim();
    if (nd == 0) {
        return true; // 标量
    }
    const auto &shape = this->shape();
    const auto &strides = this->strides();

    // 空张量
    for (auto s : shape) {
        if (s == 0) {
            return true;
        }
    }

    ptrdiff_t expect = shape[nd - 1];
    for (size_t i = nd - 1; i-- > 0;) {
        if (std::abs(strides[i]) != expect) {
            return false;
        }
        expect *= shape[i];
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    if (order.size() != this->ndim()) {
        throw std::runtime_error("Permute order size does not match tensor ndim.");
    }

    std::vector<size_t> new_shape(order.size());
    std::vector<ptrdiff_t> new_strides(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        new_shape[i] = this->shape()[order[i]];
    }
    for (size_t i = 0; i < order.size(); ++i) {
        new_strides[i] = this->strides()[order[i]];
    }

    TensorMeta meta{
        this->dtype(),
        new_shape,
        new_strides};

    return std::make_shared<Tensor>(meta, _storage);
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    auto tmp = std::accumulate(shape.begin(), shape.end(), 1, [](const size_t &a, const size_t &b) {
        return a * b;
    });
    auto new_size = static_cast<size_t>(tmp);
    if (new_size != this->numel()) {
        throw std::runtime_error("New shape numel does not match.");
    }

    std::vector<ptrdiff_t> new_strides(shape.size(), 1);
    for (int i = shape.size() - 2; i >= 0; --i) {
        new_strides[i] = new_strides[i + 1] * shape[i + 1];
    }
    auto new_tensor = Tensor::create(shape, this->dtype(), this->deviceType(), this->deviceId());
    new_tensor->load(this->data());
    return new_tensor;
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "Slice dimension out of bounds");
    CHECK_ARGUMENT(start <= end, "Slice start must be <= end");
    CHECK_ARGUMENT(end <= this->shape()[dim], "Slice end out of bounds");

    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;
    std::vector<ptrdiff_t> new_strides = this->strides();

    size_t new_offset = _offset + start * this->strides()[dim] * this->elementSize();

    TensorMeta meta{
        this->dtype(),
        new_shape,
        new_strides};

    return std::make_shared<Tensor>(meta, _storage, new_offset);
}

void Tensor::load(const void *src_) {
    size_t size = this->numel();
    llaisysDataType_t dtype = this->dtype(); // enum
    size_t element_size = utils::dsize(dtype);
    if (size * element_size > _storage->size()) {
        throw std::runtime_error("Tensor load size exceeds storage size.");
    }
    const std::byte *src = reinterpret_cast<const std::byte *>(src_);
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(this->data(), src, size * element_size);
    } else {
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(),
            src,
            size * element_size,
            LLAISYS_MEMCPY_H2D);
        core::context().runtime().api()->device_synchronize();
    }
    // trigger the github workflow
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
