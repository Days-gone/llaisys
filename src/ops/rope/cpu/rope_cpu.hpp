#pragma once

#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::cpu {
    void rope(std::byte* out, std::byte* in, std::byte* pos_ids, float theta, size_t d_seq, size_t d_heads, size_t d_dim, llaisysDataType_t dtype);

}
