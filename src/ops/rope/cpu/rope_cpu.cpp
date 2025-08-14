#include "rope_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <cmath>


template<typename T>
void rope_(T* out, T* in, int64_t* pos_ids, float theta, size_t d_seq, size_t d_heads, size_t d_dim) {
    for (size_t i = 0; i < d_seq; ++i) {
        int64_t pos = pos_ids[i];
        
        for (size_t j = 0; j < d_heads; ++j) {
            for (size_t k = 0; k < d_dim / 2; ++k) {
                // 计算角度: φ_{i,j} = p_i / θ^{2j/d}
                float freq = static_cast<float>(pos) / std::pow(theta, 2.0f * k / d_dim);
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);
                
                // 获取输入向量的索引
                size_t base_idx = i * d_heads * d_dim + j * d_dim;
                size_t a_idx = base_idx + k;
                size_t b_idx = base_idx + k + d_dim / 2;
                
                if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                    float a = llaisys::utils::cast<float>(in[a_idx]);
                    float b = llaisys::utils::cast<float>(in[b_idx]);
                    
                    // RoPE 变换
                    out[a_idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    out[b_idx] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                    float a = llaisys::utils::cast<float>(in[a_idx]);
                    float b = llaisys::utils::cast<float>(in[b_idx]);
                    
                    // RoPE 变换
                    out[a_idx] = llaisys::utils::cast<T>(a * cos_val - b * sin_val);
                    out[b_idx] = llaisys::utils::cast<T>(b * cos_val + a * sin_val);
                } else {
                    float a = static_cast<float>(in[a_idx]);
                    float b = static_cast<float>(in[b_idx]);
                    
                    // RoPE 变换
                    out[a_idx] = static_cast<T>(a * cos_val - b * sin_val);
                    out[b_idx] = static_cast<T>(b * cos_val + a * sin_val);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte* out, std::byte* in, std::byte* pos_ids, float theta, 
          size_t d_seq, size_t d_heads, size_t d_dim, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        rope_(reinterpret_cast<float*>(out), 
              reinterpret_cast<float*>(in), 
              reinterpret_cast<int64_t*>(pos_ids), 
              theta, d_seq, d_heads, d_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_(reinterpret_cast<llaisys::bf16_t*>(out), 
              reinterpret_cast<llaisys::bf16_t*>(in), 
              reinterpret_cast<int64_t*>(pos_ids), 
              theta, d_seq, d_heads, d_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_(reinterpret_cast<llaisys::fp16_t*>(out), 
              reinterpret_cast<llaisys::fp16_t*>(in), 
              reinterpret_cast<int64_t*>(pos_ids), 
              theta, d_seq, d_heads, d_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
