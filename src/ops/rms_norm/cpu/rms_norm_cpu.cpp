#include "rms_norm_cpu.hpp"
#include "llaisys.h"

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
void rms_norm_(T *out, T *in, T *weight, float eps, size_t d_b, size_t d_1, size_t d_2) {
    for (size_t b = 0; b < d_b; ++b) {
        for (size_t o = 0; o < d_2; ++o) {
            float sum = 0.0f;
            for (size_t i = 0; i < d_1; ++i) {
                float val = in[b * d_1 + i] * weight[o * d_1 + i];
                sum += val * val;
            }
            sum = std::sqrt(sum / d_1 + eps);
            out[b * d_2 + o] = sum;
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, size_t d_b, size_t d_1, size_t d_2, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F16:
        break;
    case LLAISYS_DTYPE_BF16:
        break;
    default: {
        rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<float *>(in), reinterpret_cast<float *>(weight), eps, d_b, d_1, d_2);
    }
    }
}
} // namespace llaisys::ops::cpu