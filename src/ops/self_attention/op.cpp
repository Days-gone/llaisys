#include "op.hpp"
#include "cpu/self_atten_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    cpu::self_attention(reinterpret_cast<std::byte *>(attn_val->data()),
                        reinterpret_cast<std::byte *>(q->data()),
                        reinterpret_cast<std::byte *>(k->data()),
                        reinterpret_cast<std::byte *>(v->data()),
                        scale, q->shape()[0], q->shape()[1], q->shape()[2], 
                        k->shape()[0], k->shape()[1], v->shape()[2], q->dtype());
}
} // namespace llaisys::ops
