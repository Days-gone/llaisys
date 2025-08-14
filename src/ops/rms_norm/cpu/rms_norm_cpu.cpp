#include "rms_norm_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstddef>
#include <type_traits>

/**
 * @brief Performs RMS normalization on the input tensor.
 *
 * @param out Pointer to the output tensor.
 * @param in Pointer to the input tensor.
 * @param weight Pointer to the weight tensor.
 * @param eps Small value to avoid division by zero.
 * @param d_b Batch size.
 * @param d_1 Dimension 1 of the input tensor.
 * @param d_2 Dimension 2 of the input tensor.
 * @param dtype Data type of the tensors.
 */
template <typename T>
void rms_norm_(T *out, T *in, T *weight, float eps, size_t d_b, size_t d_1) {
    for (size_t row = 0; row < d_b; row++) {
        float sum = 0.0f;
        // 计算该行的平方和
        for (size_t col = 0; col < d_1; col++) {
            if constexpr (std::is_same_v<T, llaisys::fp16_t> || std::is_same_v<T, llaisys::bf16_t>) {
                // 对于 fp16 和 bf16，转换为 float 进行计算
                float val = llaisys::utils::cast<float>(in[row * d_1 + col]);
                sum += val * val;
            } else {
                float val = static_cast<float>(in[row * d_1 + col]);
                sum += val * val;
            }
        }
        float norm = std::sqrt(sum / d_1 + eps);
        // 点乘weight向量
        for (size_t col = 0; col < d_1; col++) {
            out[row * d_1 + col] = llaisys::utils::cast<T>((llaisys::utils::cast<float>(in[row * d_1 + col]) * llaisys::utils::cast<float>(weight[col])) / norm);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, size_t d_b, size_t d_1, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16: {
        rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<llaisys::fp16_t *>(weight), eps, d_b, d_1);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<llaisys::bf16_t *>(weight), eps, d_b, d_1);
        break;
    }
    default: {
        rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<float *>(weight), eps, d_b, d_1);
    }
    }
}
} // namespace llaisys::ops::cpu