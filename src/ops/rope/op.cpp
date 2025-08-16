#include "op.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    llaisys::ops::cpu::rope(reinterpret_cast<std::byte*>(out->data()), reinterpret_cast<std::byte*>(in->data()), reinterpret_cast<std::byte*>(pos_ids->data()), theta, in->shape()[0], in->shape()[1], in->shape()[2], in->dtype());
}
} // namespace llaisys::ops
