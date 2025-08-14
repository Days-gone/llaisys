#include "op.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    llaisys::ops::cpu::swiglu(out->data(), gate->data(), up->data(),
                              out->shape()[0], out->shape()[1], out->dtype());
}
} // namespace llaisys::ops
