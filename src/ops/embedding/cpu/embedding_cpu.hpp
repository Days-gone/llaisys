#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, std::byte *weight, size_t index_len,std::vector<size_t> shape, llaisysDataType_t dtype);
}