from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..tensor import Tensor

from pathlib import Path
import safetensors
import torch


class Qwen2Layer:
    def __init__(self, layer_idx: int, device: DeviceType = DeviceType.CPU):
        self.layer_idx = layer_idx
        self.device = device

        # 注意力层权重
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None
        self.o_proj_weight = None

        # MLP层权重
        self.gate_proj_weight = None
        self.up_proj_weight = None
        self.down_proj_weight = None

        # 归一化层权重
        self.input_layernorm_weight = None
        self.post_attention_layernorm_weight = None

        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0

    def load_weight(self, name: str, data: "torch.Tensor"):
        """加载单个权重"""
        tensor = Tensor.from_torch(data, device=self.device)

        if f"model.layers.{self.layer_idx}.self_attn.q_proj.weight" in name:
            self.q_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.self_attn.q_proj.bias" in name:
            self.q_proj_bias = tensor
        elif f"model.layers.{self.layer_idx}.self_attn.k_proj.weight" in name:
            self.k_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.self_attn.k_proj.bias" in name:
            self.k_proj_bias = tensor
        elif f"model.layers.{self.layer_idx}.self_attn.v_proj.weight" in name:
            self.v_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.self_attn.v_proj.bias" in name:
            self.v_proj_bias = tensor
        elif f"model.layers.{self.layer_idx}.self_attn.o_proj.weight" in name:
            self.o_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.mlp.gate_proj.weight" in name:
            self.gate_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.mlp.up_proj.weight" in name:
            self.up_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.mlp.down_proj.weight" in name:
            self.down_proj_weight = tensor
        elif f"model.layers.{self.layer_idx}.input_layernorm.weight" in name:
            self.input_layernorm_weight = tensor
        elif f"model.layers.{self.layer_idx}.post_attention_layernorm.weight" in name:
            self.post_attention_layernorm_weight = tensor

    def clean_cache(self) -> None:
        """清理缓存"""
        self.k_cache = None
        self.v_cache = None
        self.cache_len = 0


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        from ..ops import Ops

        self.ops = Ops
        self.device = device

        # 缓存设置
        self.use_kv_cache = False

        # 模型权重
        self.embed_tokens_weight = None
        self.lm_head_weight = None
        self.norm_weight = None
        self.layers = []

        # 从权重文件名推断层数 (0-27 = 28层)
        self.num_layers = 28
        for i in range(self.num_layers):
            self.layers.append(Qwen2Layer(i, device))

        # 加载模型权重
        model_path = Path(model_path)
        for file in sorted(model_path.glob("*.safetensors")):
            # 使用PyTorch框架加载
            data_ = safetensors.safe_open(file, framework="pt", device="cpu")
            for name_ in data_.keys():
                weight_tensor = data_.get_tensor(name_)
                self._load_weight(name_, weight_tensor)

    def _load_weight(self, name: str, data: "torch.Tensor"):
        """加载权重到对应的组件"""
        if name == "model.embed_tokens.weight":
            self.embed_tokens_weight = Tensor.from_torch(data, device=self.device)
        elif name == "lm_head.weight":
            self.lm_head_weight = Tensor.from_torch(data, device=self.device)
        elif name == "model.norm.weight":
            self.norm_weight = Tensor.from_torch(data, device=self.device)
        elif name.startswith("model.layers."):
            # 提取层索引
            layer_idx = int(name.split(".")[2])
            if 0 <= layer_idx < self.num_layers:
                self.layers[layer_idx].load_weight(name, data)
        else:
            raise RuntimeError(f"Unknown weight name: {name}")

    def enable_cache(self) -> None:
        self.use_kv_cache = True
        if not self.use_kv_cache:
            for layer in self.layers:
                layer.clean_cache()

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clean_cache()

    def forward(
        self, input_ids: Tensor, pos_ids: Tensor, use_kv_cache: bool = False
    ) -> Tensor:
        """前向传播"""
        seq_len = input_ids.shape()[0]
        hidden_dim = self.embed_tokens_weight.shape()[1]

        # 词嵌入
        hidden_states = Tensor.zeros(
            [seq_len, hidden_dim],
            device=self.device,
            dtype=DataType.BF16,
        )
        self.ops.embedding(hidden_states, input_ids, self.embed_tokens_weight)

        # 逐层处理
        for i, layer in enumerate(self.layers):
            hidden_states = self._forward_layer(
                hidden_states, layer, pos_ids, use_kv_cache
            )

        # 最终归一化
        normalized = Tensor.zeros_like(hidden_states)
        self.ops.rms_norm(normalized, hidden_states, self.norm_weight, 1e-6)

        # 语言模型头
        vocab_size = self.lm_head_weight.shape()[0]
        logits = Tensor.zeros(
            [seq_len, vocab_size], device=self.device, dtype=normalized.dtype()
        )
        zero_bias = Tensor.zeros(
            [vocab_size], device=self.device, dtype=normalized.dtype()
        )
        self.ops.linear(logits, normalized, self.lm_head_weight, zero_bias)

        return logits

    def _forward_layer(
        self,
        hidden_states: Tensor,
        layer: Qwen2Layer,
        pos_ids: Tensor,
        use_kv_cache: bool,
    ) -> Tensor:
        """单层前向传播"""
        seq_len, hidden_dim = hidden_states.shape()

        # 输入归一化
        normed_input = Tensor.zeros_like(hidden_states)
        self.ops.rms_norm(
            normed_input, hidden_states, layer.input_layernorm_weight, 1e-6
        )

        # 自注意力
        attn_output = self._forward_attention(
            normed_input, layer, pos_ids, use_kv_cache
        )

        # 残差连接
        attn_residual = Tensor.zeros_like(hidden_states)
        self.ops.add(attn_residual, hidden_states, attn_output)

        # 后注意力归一化
        normed_attn = Tensor.zeros_like(attn_residual)
        self.ops.rms_norm(
            normed_attn, attn_residual, layer.post_attention_layernorm_weight, 1e-6
        )

        # MLP
        mlp_output = self._forward_mlp(normed_attn, layer)

        # 最终残差连接
        final_output = Tensor.zeros_like(attn_residual)
        self.ops.add(final_output, attn_residual, mlp_output)

        return final_output

    def _forward_attention(
        self,
        hidden_states: Tensor,
        layer: Qwen2Layer,
        pos_ids: Tensor,
        use_kv_cache: bool,
    ) -> Tensor:
        """自注意力前向传播，支持 KV cache"""
        seq_len, hidden_dim = hidden_states.shape()

        # 根据权重正确定义维度
        q_dim = layer.q_proj_weight.shape()[0]
        kv_dim = layer.k_proj_weight.shape()[0]
        head_dim = 128
        num_q_heads = q_dim // head_dim
        num_kv_heads = kv_dim // head_dim

        # Q 投影（每次都需要计算）
        q = Tensor.zeros(
            [seq_len, q_dim], device=self.device, dtype=hidden_states.dtype()
        )

        # 处理 Q 的 bias
        if layer.q_proj_bias is not None:
            self.ops.linear(q, hidden_states, layer.q_proj_weight, layer.q_proj_bias)
        else:
            q_bias = Tensor.zeros(
                [q_dim], device=self.device, dtype=hidden_states.dtype()
            )
            self.ops.linear(q, hidden_states, layer.q_proj_weight, q_bias)

        # K, V 投影（当前输入的部分）
        k_current = Tensor.zeros(
            [seq_len, kv_dim], device=self.device, dtype=hidden_states.dtype()
        )
        v_current = Tensor.zeros(
            [seq_len, kv_dim], device=self.device, dtype=hidden_states.dtype()
        )

        # 处理 K 的 bias
        if layer.k_proj_bias is not None:
            self.ops.linear(
                k_current, hidden_states, layer.k_proj_weight, layer.k_proj_bias
            )
        else:
            k_bias = Tensor.zeros(
                [kv_dim], device=self.device, dtype=hidden_states.dtype()
            )
            self.ops.linear(k_current, hidden_states, layer.k_proj_weight, k_bias)

        # 处理 V 的 bias
        if layer.v_proj_bias is not None:
            self.ops.linear(
                v_current, hidden_states, layer.v_proj_weight, layer.v_proj_bias
            )
        else:
            v_bias = Tensor.zeros(
                [kv_dim], device=self.device, dtype=hidden_states.dtype()
            )
            self.ops.linear(v_current, hidden_states, layer.v_proj_weight, v_bias)

        # 处理 KV cache - 使用新的 cat_two 方法
        if use_kv_cache and layer.k_cache is not None:
            # 有缓存，使用 cat_two 连接
            k_to_use = Tensor.cat_two(layer.k_cache, k_current, dim=0)
            v_to_use = Tensor.cat_two(layer.v_cache, v_current, dim=0)

            # 更新 cache
            layer.k_cache = k_to_use
            layer.v_cache = v_to_use
            layer.cache_len += seq_len

            kv_seq_len = layer.cache_len
        else:
            # 首次调用或不使用 cache
            k_to_use = k_current
            v_to_use = v_current
            kv_seq_len = seq_len

            if use_kv_cache:
                # 初始化 cache
                layer.k_cache = k_current
                layer.v_cache = v_current
                layer.cache_len = seq_len

        # 重塑为多头格式
        q_reshaped = q.view([seq_len, num_q_heads, head_dim])
        k_reshaped = k_to_use.view([kv_seq_len, num_kv_heads, head_dim])
        v_reshaped = v_to_use.view([kv_seq_len, num_kv_heads, head_dim])

        # RoPE 位置编码
        q_rope = Tensor.zeros_like(q_reshaped)
        k_rope = Tensor.zeros_like(k_reshaped)

        # 为 Q 使用当前的位置编码
        self.ops.rope(q_rope, q_reshaped, pos_ids, 10000.0)

        # 为 K 创建完整的位置编码
        if use_kv_cache and layer.cache_len > seq_len:
            # 创建从 0 到 kv_seq_len-1 的位置编码
            import torch

            full_pos_ids = Tensor.from_torch(
                torch.arange(kv_seq_len, dtype=torch.int64), device=self.device
            )
            self.ops.rope(k_rope, k_reshaped, full_pos_ids, 10000.0)
        else:
            self.ops.rope(k_rope, k_reshaped, pos_ids, 10000.0)

        # 自注意力
        attn_output_reshaped = Tensor.zeros_like(q_rope)
        scale = 1.0 / (head_dim**0.5)
        self.ops.self_attention(attn_output_reshaped, q_rope, k_rope, v_reshaped, scale)

        # 重塑回原始形状
        attn_output = attn_output_reshaped.view([seq_len, q_dim])

        # 输出投影
        output = Tensor.zeros(
            [seq_len, hidden_dim], device=self.device, dtype=hidden_states.dtype()
        )

        # 处理输出投影的 bias
        if hasattr(layer, "o_proj_bias") and layer.o_proj_bias is not None:
            self.ops.linear(output, attn_output, layer.o_proj_weight, layer.o_proj_bias)
        else:
            o_bias = Tensor.zeros(
                [hidden_dim], device=self.device, dtype=hidden_states.dtype()
            )
            self.ops.linear(output, attn_output, layer.o_proj_weight, o_bias)

        return output

    def _forward_mlp(self, hidden_states: Tensor, layer: Qwen2Layer) -> Tensor:
        """MLP前向传播"""
        seq_len, hidden_dim = hidden_states.shape()
        intermediate_size = layer.gate_proj_weight.shape()[0]

        # 门控和上投影
        gate = Tensor.zeros(
            [seq_len, intermediate_size],
            device=self.device,
            dtype=hidden_states.dtype(),
        )
        up = Tensor.zeros(
            [seq_len, intermediate_size],
            device=self.device,
            dtype=hidden_states.dtype(),
        )

        zero_bias = Tensor.zeros(
            [intermediate_size], device=self.device, dtype=hidden_states.dtype()
        )
        self.ops.linear(gate, hidden_states, layer.gate_proj_weight, zero_bias)
        self.ops.linear(up, hidden_states, layer.up_proj_weight, zero_bias)

        # SwiGLU激活
        swiglu_output = Tensor.zeros(
            [seq_len, intermediate_size],
            device=self.device,
            dtype=hidden_states.dtype(),
        )
        self.ops.swiglu(swiglu_output, gate, up)

        # 下投影
        output = Tensor.zeros(
            [seq_len, hidden_dim], device=self.device, dtype=hidden_states.dtype()
        )
        self.ops.linear(output, swiglu_output, layer.down_proj_weight, zero_bias)

        return output

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        use_kv_cache: bool = True,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        """生成文本"""
        if max_new_tokens is None:
            max_new_tokens = 100

        # 设置 cache
        if use_kv_cache:
            self.enable_cache()

        input_ids = list(inputs)

        for step in range(max_new_tokens):
            import torch

            if step == 0 or not use_kv_cache:
                # prefill
                seq_len = len(input_ids)
                input_tensor = Tensor.from_torch(
                    torch.tensor(input_ids, dtype=torch.int64), device=self.device
                )
                pos_tensor = Tensor.from_torch(
                    torch.arange(seq_len, dtype=torch.int64), device=self.device
                )
            else:
                # decode
                seq_len = 1
                input_tensor = Tensor.from_torch(
                    torch.tensor([input_ids[-1]], dtype=torch.int64), device=self.device
                )
                pos_tensor = Tensor.from_torch(
                    torch.tensor([len(input_ids) - 1], dtype=torch.int64),
                    device=self.device,
                )

            # 前向传播
            logits = self.forward(input_tensor, pos_tensor, use_kv_cache=use_kv_cache)

            # 获取最后一个位置的 logits
            last_row = logits.slice(0, logits.shape()[0] - 1, logits.shape()[0])

            # argmax 采样
            max_idx = Tensor.zeros([1], dtype=DataType.I64, device=self.device)
            max_val = Tensor.zeros([1], dtype=DataType.BF16, device=self.device)
            self.ops.argmax(max_idx, max_val, last_row)

            # 获取下一个 token
            next_token = max_idx.item()
            input_ids.append(next_token)

            # 检查结束条件
            if next_token == 151643:
                break

        return input_ids
