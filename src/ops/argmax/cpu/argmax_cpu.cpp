#include "argmax_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>
#include <cstdint>
#include <type_traits>

template <typename T, typename Idx>
void argmax_(T *vals, T *max_val, Idx *max_idx, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
            if (llaisys::utils::_bf16_to_f32(vals[i]) > llaisys::utils::_bf16_to_f32(*max_val)) {
                *max_val = vals[i];
                std::cout << "New max found: " << llaisys::utils::_bf16_to_f32(*max_val) << " at index " << i << "\n";
                *max_idx = i;
            }
        } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::_f16_to_f32(vals[i]) > llaisys::utils::_f16_to_f32(*max_val)) {
                *max_val = vals[i];
                std::cout << "New max found: " << llaisys::utils::_f16_to_f32(*max_val) << " at index " << i << "\n";
                *max_idx = i;
            }
        } else {
            if (vals[i] > *max_val) {
                *max_val = vals[i];
                std::cout << "New max found: " << *max_val << " at index " << i << "\n";
                *max_idx = i;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *vals, std::byte *max_val, std::byte *max_idx, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F16: {
        argmax_(reinterpret_cast<llaisys::fp16_t *>(vals), reinterpret_cast<llaisys::fp16_t *>(max_val), reinterpret_cast<int64_t *>(max_idx), size);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        // TODO
        argmax_(reinterpret_cast<llaisys::bf16_t *>(vals), reinterpret_cast<llaisys::bf16_t *>(max_val), reinterpret_cast<int64_t *>(max_idx), size);
        break;
    }
    default: {
        // TODO
        argmax_(reinterpret_cast<float *>(vals), reinterpret_cast<float *>(max_val), reinterpret_cast<int64_t *>(max_idx), size);
        break;
    }
    }
}

} // namespace llaisys::ops::cpu