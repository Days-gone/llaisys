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
    // q: [d_seq, d_head, d_dim]
    // k: [d_totallen, d_kvhead, d_dim]
    // v: [d_totallen, d_kvhead, d_v]
    // atten_val: [d_seq, d_head, d_v]

    for (size_t seq_i = 0; seq_i < d_seq; ++seq_i) {
        for (size_t head = 0; head < d_head; ++head) {
            // 修正头复制逻辑：计算每个KV头对应多少个Q头
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

            // Causal softmax: 只对 total_j <= seq_i 的位置计算softmax
            float max_score = -INFINITY;
            for (size_t total_j = 0; total_j <= seq_i && total_j < d_totallen; ++total_j) {
                max_score = std::max(max_score, attention_scores[total_j]);
            }

            float sum_exp = 0.0f;
            for (size_t total_j = 0; total_j <= seq_i && total_j < d_totallen; ++total_j) {
                attention_scores[total_j] = std::exp(attention_scores[total_j] - max_score);
                sum_exp += attention_scores[total_j];
            }

            // 归一化
            for (size_t total_j = 0; total_j <= seq_i && total_j < d_totallen; ++total_j) {
                attention_scores[total_j] /= sum_exp;
            }

            // 将未来位置的注意力权重设为0
            for (size_t total_j = seq_i + 1; total_j < d_totallen; ++total_j) {
                attention_scores[total_j] = 0.0f;
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