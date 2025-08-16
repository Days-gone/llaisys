#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    llaisys::ops::cpu::swiglu(out->data(), gate->data(), up->data(),
                              out->shape()[0], out->shape()[1], out->dtype());
}
} // namespace llaisys::ops
