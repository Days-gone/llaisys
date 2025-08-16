#include "op.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    if (bias) {
        llaisys::ops::cpu::linear(reinterpret_cast<std::byte *>(out->data()), reinterpret_cast<std::byte *>(in->data()), reinterpret_cast<std::byte *>(weight->data()), reinterpret_cast<std::byte *>(bias->data()), in->shape()[0], weight->shape()[1], weight->shape()[0], in->dtype());
    } else {
        llaisys::ops::cpu::linear(reinterpret_cast<std::byte *>(out->data()), reinterpret_cast<std::byte *>(in->data()), reinterpret_cast<std::byte *>(weight->data()), nullptr, in->shape()[0], weight->shape()[1], weight->shape()[1], in->dtype());
    }
}
} // namespace llaisys::ops
