#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstdint>
#include <cstring>

template <typename T, typename I>
void embedding_(T *out, I *index, T *weight, size_t index_len, std::vector<size_t> shape) {
    for (size_t i = 0; i < index_len; ++i) {
        size_t row_number = static_cast<size_t>(index[i]);
        size_t out_offset = i * shape[1];
        size_t weight_offset = row_number * shape[1];
        std::memcpy(out + out_offset, weight + weight_offset, shape[1] * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, size_t index_len, std::vector<size_t> shape, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BF16: {
        embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                   reinterpret_cast<int64_t *>(index),
                   reinterpret_cast<llaisys::bf16_t *>(weight),
                   index_len, shape);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                   reinterpret_cast<int64_t *>(index),
                   reinterpret_cast<llaisys::fp16_t *>(weight),
                   index_len, shape);
        break;
    }
    case LLAISYS_DTYPE_F32: {
        embedding_(reinterpret_cast<float *>(out),
                   reinterpret_cast<int64_t *>(index),
                   reinterpret_cast<float *>(weight),
                   index_len, shape);
        break;
    }
    default:
        throw std::runtime_error("Unsupported data type for embedding operation.");
    }
}
} // namespace llaisys::ops::cpu