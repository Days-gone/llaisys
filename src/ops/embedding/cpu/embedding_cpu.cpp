#include "embedding_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

/**
 * @brief Embedding lookup operation.
 *
 * @tparam T Data type of the output and weight tensors.
 * @tparam I Data type of the index tensor.
 * @param out Output tensor.
 * @param index Index tensor.
 * @param weight Weight tensor.
 * @param index_len Length of the index tensor.
 * @param d_vocab Vocabulary size (number of rows in the weight matrix).
 * @param d_model Model dimension (number of columns in the weight matrix).
 */
template <typename T, typename I>
void embedding_(T *out, I *index, T *weight, size_t index_len, size_t d_vocab, size_t d_model) {
    // in(index_len)
    // out(index_len, d_model)
    // weight(d_vocab, d_model)

    for (size_t i = 0; i < index_len; ++i) {
        size_t row_number = static_cast<size_t>(index[i]);

        // 添加边界检查
        if (row_number >= d_vocab) {
            // 处理越界情况，可以设置为零向量或抛出异常
            std::memset(out + i * d_model, 0, d_model * sizeof(T));
            continue;
        }

        size_t out_offset = i * d_model;
        size_t weight_offset = row_number * d_model;
        std::memcpy(out + out_offset, weight + weight_offset, d_model * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, size_t d_idx, size_t d_vocab, size_t d_model, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BF16: {
        embedding_(reinterpret_cast<llaisys::bf16_t *>(out),
                   reinterpret_cast<int64_t *>(index),
                   reinterpret_cast<llaisys::bf16_t *>(weight),
                   d_idx, d_vocab, d_model);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        embedding_(reinterpret_cast<llaisys::fp16_t *>(out),
                   reinterpret_cast<int64_t *>(index),
                   reinterpret_cast<llaisys::fp16_t *>(weight),
                   d_idx, d_vocab, d_model);
        break;
    }
    case LLAISYS_DTYPE_F32: {
        embedding_(reinterpret_cast<float *>(out),
                   reinterpret_cast<int64_t *>(index),
                   reinterpret_cast<float *>(weight),
                   d_idx, d_vocab, d_model);
        break;
    }
    default:
        throw std::runtime_error("Unsupported data type for embedding operation.");
    }
}
} // namespace llaisys::ops::cpu