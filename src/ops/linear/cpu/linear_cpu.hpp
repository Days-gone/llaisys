#pragma once
#include "llaisys.h"

namespace llaisys::ops::cpu {
void linear(std::byte *out, std::byte *in, std::byte *weight, std::byte *bias, size_t d_b, size_t d_1, size_t d_2, llaisysDataType_t dtype);

}