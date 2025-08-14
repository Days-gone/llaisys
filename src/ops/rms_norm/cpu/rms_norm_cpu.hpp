#pragma once
#include "llaisys.h"
#include <cmath>


namespace llaisys::ops::cpu {
    void rms_norm(std::byte *out, std::byte *in, std::byte *weight, float eps, size_t d_b, size_t d_1, llaisysDataType_t dtype);
}