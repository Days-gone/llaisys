#include "linear_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cstddef>
#include <cstring>
#include <type_traits>

/**
 * @brief Performs a linear transformation on the input tensor.
 *
 * @param out Pointer to the output tensor.
 * @param in Pointer to the input tensor.
 * @param weight Pointer to the weight tensor.
 * @param bias Pointer to the bias tensor.
 * @param d_b
 * @param d_1 dimension 1 of weight
 * @param d_2 dimension 2 of weight
 */
template <typename T>
void linear_(T *out, T *in, T *weight, T *bias, size_t d_b, size_t d_1, size_t d_2) {
    for (size_t b = 0; b < d_b; ++b) {
        for (size_t o = 0; o < d_2; ++o) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                // fp16 case
                float sum = bias ? llaisys::utils::_bf16_to_f32(bias[o]) : 0.0f;
                for (size_t i = 0; i < d_1; ++i) {
                    float tmp = llaisys::utils::_bf16_to_f32(in[b * d_1 + i]) * llaisys::utils::_bf16_to_f32(weight[o * d_1 + i]);
                    sum += tmp;
                }
                out[b * d_2 + o] = llaisys::utils::_f32_to_bf16(sum);
            } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                // fp16 case
                float sum = bias ? llaisys::utils::_f16_to_f32(bias[o]) : 0.0f;
                for (size_t i = 0; i < d_1; ++i) {
                    float tmp = llaisys::utils::_f16_to_f32(in[b * d_1 + i]) * llaisys::utils::_f16_to_f32(weight[o * d_1 + i]);
                    sum += tmp;
                }
                out[b * d_2 + o] = llaisys::utils::_f32_to_f16(sum);
            } else {
                float sum = bias ? bias[o] : 0.0f;
                for (size_t i = 0; i < d_1; ++i) {
                    sum += in[b * d_1 + i] * weight[o * d_1 + i];
                }
                out[b * d_2 + o] = sum;
            }
        }
    }
}

template <typename T>
void linear_(T *out, T *in, T *weight, size_t d_b, size_t d_1, size_t d_2) {
    linear_(out, in, weight, nullptr, d_b, d_1, d_2);
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, size_t d_b, size_t d_1, size_t d_2, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BF16:
        linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<llaisys::bf16_t *>(in), reinterpret_cast<llaisys::bf16_t *>(weight), reinterpret_cast<llaisys::bf16_t *>(bias), d_b, d_1, d_2);
        break;
    case LLAISYS_DTYPE_F16:
        linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<llaisys::fp16_t *>(in), reinterpret_cast<llaisys::fp16_t *>(weight), reinterpret_cast<llaisys::fp16_t *>(bias), d_b, d_1, d_2);
        break;
    default:
        linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<float *>(in),
            reinterpret_cast<float *>(weight),
            reinterpret_cast<float *>(bias),
            d_b, d_1, d_2);
    }
}
} // namespace llaisys::ops::cpu