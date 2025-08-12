#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::byte *vals, std::byte *max_val, std::byte *max_idx, llaisysDataType_t type, size_t size);
}