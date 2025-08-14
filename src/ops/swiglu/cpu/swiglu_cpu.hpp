#pragma once

#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::cpu {
    void swiglu(std::byte* output, std::byte* gate, std::byte* up, size_t d_seq, size_t d_inter, llaisysDataType_t dtype);
}