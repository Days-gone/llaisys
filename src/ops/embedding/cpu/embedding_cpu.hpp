#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, size_t d_idx, size_t d_vocab, size_t d_model, llaisysDataType_t dtype);
}