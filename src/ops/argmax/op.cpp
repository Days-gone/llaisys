#include "op.hpp"
#include "./cpu/argmax_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::argmax(vals->data(), max_val->data(), max_idx->data(), vals->dtype(), vals->numel());
    } else {
        // Call GPU implementation
        TO_BE_IMPLEMENTED();
    }
}
} // namespace llaisys::ops
