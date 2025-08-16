#include "op.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        cpu::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[0], weight->shape()[1], weight->dtype());
    }
}
} // namespace llaisys::ops
