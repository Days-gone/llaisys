#include "self_atten_cpu.hpp"
#include "../../../utils.hpp"
#include "llaisys.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

template <typename T>
void self_atten_(T *atten_val, T *q, T *k, T *v, float scale,
                 size_t d_seq, size_t d_head, size_t d_dim, size_t d_totallen, size_t d_kvhead, size_t d_v) {
    // q: [d_seq, d_head, d_dim]  -> L = d_seq
    // k: [d_totallen, d_kvhead, d_dim]  -> S = d_totallen
    // v: [d_totallen, d_kvhead, d_v]
    // atten_val: [d_seq, d_head, d_v]

    for (size_t seq_i = 0; seq_i < d_seq; ++seq_i) {
        for (size_t head = 0; head < d_head; ++head) {
            size_t heads_per_kv = d_head / d_kvhead;
            size_t kv_head = head / heads_per_kv;
            
            // 计算注意力分数 A = Q * K^T * scale
            std::vector<float> attention_scores(d_totallen);

            for (size_t total_j = 0; total_j < d_totallen; ++total_j) {
                float score = 0.0f;

                // Q[seq_i, head, :] * K[total_j, kv_head, :]
                for (size_t dim = 0; dim < d_dim; ++dim) {
                    size_t q_idx = seq_i * d_head * d_dim + head * d_dim + dim;
                    size_t k_idx = total_j * d_kvhead * d_dim + kv_head * d_dim + dim;

                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = static_cast<float>(q[q_idx]);
                        k_val = static_cast<float>(k[k_idx]);
                    }

                    score += q_val * k_val;
                }

                attention_scores[total_j] = score * scale;
            }

            // L = d_seq, S = d_totallen
            // diagonal = S - L = d_totallen - d_seq
            size_t L = d_seq;
            size_t S = d_totallen;
            int diagonal = static_cast<int>(S) - static_cast<int>(L);
            
            // 对于位置 (i, j)，如果 j <= i + diagonal，则保留；否则 mask 掉
            float max_score = -INFINITY;
            bool has_valid_position = false;
            
            for (size_t total_j = 0; total_j < d_totallen; ++total_j) {
                if (static_cast<int>(total_j) <= static_cast<int>(seq_i) + diagonal) {
                    max_score = std::max(max_score, attention_scores[total_j]);
                    has_valid_position = true;
                }
            }

            // 如果没有有效位置，使用第一个位置避免 -inf
            if (!has_valid_position) {
                max_score = attention_scores[0];
            }

            float sum_exp = 0.0f;
            for (size_t total_j = 0; total_j < d_totallen; ++total_j) {
                if (static_cast<int>(total_j) <= static_cast<int>(seq_i) + diagonal) {
                    attention_scores[total_j] = std::exp(attention_scores[total_j] - max_score);
                    sum_exp += attention_scores[total_j];
                } else {
                    attention_scores[total_j] = 0.0f;  // mask 掉未来位置
                }
            }

            // 归一化
            if (sum_exp > 0.0f) {
                for (size_t total_j = 0; total_j < d_totallen; ++total_j) {
                    if (static_cast<int>(total_j) <= static_cast<int>(seq_i) + diagonal) {
                        attention_scores[total_j] /= sum_exp;
                    }
                }
            }

            // 计算输出 Y = softmax(A) * V
            for (size_t v_dim = 0; v_dim < d_v; ++v_dim) {
                float output = 0.0f;

                for (size_t total_j = 0; total_j < d_totallen; ++total_j) {
                    size_t v_idx = total_j * d_kvhead * d_v + kv_head * d_v + v_dim;

                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = static_cast<float>(v[v_idx]);
                    }

                    output += attention_scores[total_j] * v_val;
                }

                size_t out_idx = seq_i * d_head * d_v + head * d_v + v_dim;
                if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
                    atten_val[out_idx] = llaisys::utils::cast<T>(output);
                } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
                    atten_val[out_idx] = llaisys::utils::cast<T>(output);
                } else {
                    atten_val[out_idx] = static_cast<T>(output);
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *atten_val, std::byte *q, std::byte *k, std::byte *v, float scale, 
    size_t d_seq, size_t d_head, size_t d_dim, size_t d_totallen, size_t d_kvhead, size_t d_v,
                llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        self_atten_(reinterpret_cast<float *>(atten_val),
                    reinterpret_cast<float *>(q),
                    reinterpret_cast<float *>(k),
                    reinterpret_cast<float *>(v),
                    scale, d_seq, d_head, d_dim, d_totallen, d_kvhead, d_v);
        break;
    case LLAISYS_DTYPE_BF16:
        self_atten_(reinterpret_cast<llaisys::bf16_t *>(atten_val),
                    reinterpret_cast<llaisys::bf16_t *>(q),
                    reinterpret_cast<llaisys::bf16_t *>(k),
                    reinterpret_cast<llaisys::bf16_t *>(v),
                    scale, d_seq, d_head, d_dim, d_totallen, d_kvhead, d_v);
        break;
    case LLAISYS_DTYPE_F16:
        self_atten_(reinterpret_cast<llaisys::fp16_t *>(atten_val),
                    reinterpret_cast<llaisys::fp16_t *>(q),
                    reinterpret_cast<llaisys::fp16_t *>(k),
                    reinterpret_cast<llaisys::fp16_t *>(v),
                    scale, d_seq, d_head, d_dim, d_totallen, d_kvhead, d_v);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu