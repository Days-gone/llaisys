#include "swiglu_cpu.hpp"
#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void swiglu_(T* output, T* gate, T* up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
            float gate_val = llaisys::utils::cast<float>(gate[i]);
            float up_val = llaisys::utils::cast<float>(up[i]);
            
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            float result = up_val * silu;
            
            output[i] = llaisys::utils::cast<T>(result);
        } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
            float gate_val = llaisys::utils::cast<float>(gate[i]);
            float up_val = llaisys::utils::cast<float>(up[i]);
            
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            float result = up_val * silu;
            
            output[i] = llaisys::utils::cast<T>(result);
        } else {
            float gate_val = static_cast<float>(gate[i]);
            float up_val = static_cast<float>(up[i]);
            
            float silu = gate_val / (1.0f + std::exp(-gate_val));
            
            float result = up_val * silu;
            
            output[i] = static_cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte* output, std::byte* gate, std::byte* up, size_t d_seq, size_t d_inter, llaisysDataType_t dtype) {
    // output(d_seq, d_inter)
    // gate(d_seq, d_inter)
    // up(d_seq, d_inter)
    
    size_t numel = d_seq * d_inter;
    
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        swiglu_(reinterpret_cast<float*>(output),
                reinterpret_cast<float*>(gate),
                reinterpret_cast<float*>(up),
                numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_(reinterpret_cast<llaisys::bf16_t*>(output),
                reinterpret_cast<llaisys::bf16_t*>(gate),
                reinterpret_cast<llaisys::bf16_t*>(up),
                numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_(reinterpret_cast<llaisys::fp16_t*>(output),
                reinterpret_cast<llaisys::fp16_t*>(gate),
                reinterpret_cast<llaisys::fp16_t*>(up),
                numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
}