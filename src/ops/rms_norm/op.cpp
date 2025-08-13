#include "op.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "llaisys.h"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::rms_norm(reinterpret_cast<std::byte *>(out->data()), reinterpret_cast<std::byte *>(in->data()), reinterpret_cast<std::byte *>(weight->data()), eps, in->shape()[0], weight->shape()[1], weight->shape()[0], in->dtype());
    }
}
} // namespace llaisys::ops
