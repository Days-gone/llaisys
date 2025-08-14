#pragma once
#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *atten_val, std::byte *q, std::byte *k, std::byte *v, float scale,
                    size_t d_seq, size_t d_head, size_t d_dim, size_t d_totallen, size_t d_kvhead, size_t d_v, llaisysDataType_t dtype);

}