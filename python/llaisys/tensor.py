from typing import Sequence, Tuple

from .libllaisys import (
    LIB_LLAISYS,
    llaisysTensor_t,
    llaisysDeviceType_t,
    DeviceType,
    llaisysDataType_t,
    DataType,
)
from ctypes import c_size_t, c_int, c_ssize_t, c_void_p
import numpy as np
import torch


class Tensor:
    def __init__(
        self,
        shape: Sequence[int] = None,
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
        tensor: llaisysTensor_t = None,
    ):
        if tensor:
            self._tensor = tensor
        else:
            _ndim = 0 if shape is None else len(shape)
            _shape = None if shape is None else (c_size_t * len(shape))(*shape)
            self._tensor: llaisysTensor_t = LIB_LLAISYS.tensorCreate(
                _shape,
                c_size_t(_ndim),
                llaisysDataType_t(dtype),
                llaisysDeviceType_t(device),
                c_int(device_id),
            )

            # 如果指定了shape，创建并加载全零数据
            if shape is not None:
                # 映射DataType到torch dtype
                dtype_map = {
                    DataType.F32: torch.float32,
                    DataType.F16: torch.float16,
                    DataType.BF16: torch.bfloat16,
                    DataType.I64: torch.int64,
                    DataType.I32: torch.int32,
                }

                torch_dtype = dtype_map.get(dtype, torch.float32)

                # 创建PyTorch全零tensor
                zero_tensor = torch.zeros(shape, dtype=torch_dtype)

                # 加载全零数据到C++后端
                self.load(zero_tensor.data_ptr())

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
            self._tensor = None

    def shape(self) -> Tuple[int]:
        buf = (c_size_t * self.ndim())()
        LIB_LLAISYS.tensorGetShape(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def strides(self) -> Tuple[int]:
        buf = (c_ssize_t * self.ndim())()
        LIB_LLAISYS.tensorGetStrides(self._tensor, buf)
        return tuple(buf[i] for i in range(self.ndim()))

    def ndim(self) -> int:
        return int(LIB_LLAISYS.tensorGetNdim(self._tensor))

    def dtype(self) -> DataType:
        return DataType(LIB_LLAISYS.tensorGetDataType(self._tensor))

    def device_type(self) -> DeviceType:
        return DeviceType(LIB_LLAISYS.tensorGetDeviceType(self._tensor))

    def device_id(self) -> int:
        return int(LIB_LLAISYS.tensorGetDeviceId(self._tensor))

    def data_ptr(self) -> c_void_p:
        return LIB_LLAISYS.tensorGetData(self._tensor)

    def lib_tensor(self) -> llaisysTensor_t:
        return self._tensor

    def debug(self):
        LIB_LLAISYS.tensorDebug(self._tensor)

    def __repr__(self):
        return f"<Tensor shape={self.shape}, dtype={self.dtype}, device={self.device_type}:{self.device_id}>"

    def load(self, data: c_void_p):
        LIB_LLAISYS.tensorLoad(self._tensor, data)

    def is_contiguous(self) -> bool:
        return bool(LIB_LLAISYS.tensorIsContiguous(self._tensor))

    def view(self, *shape) -> "Tensor":
        # 如果传入的是单个列表/元组，解包它
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(
            tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape)))
        )

    def permute(self, *perm: int) -> llaisysTensor_t:
        assert len(perm) == self.ndim()
        _perm = (c_size_t * len(perm))(*perm)
        return Tensor(tensor=LIB_LLAISYS.tensorPermute(self._tensor, _perm))

    def slice(self, dim: int, start: int, end: int):
        return Tensor(
            tensor=LIB_LLAISYS.tensorSlice(
                self._tensor, c_size_t(dim), c_size_t(start), c_size_t(end)
            )
        )

    @classmethod
    def zeros(
        cls,
        shape: Sequence[int],
        dtype: DataType = DataType.F32,
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
    ):
        """创建全零tensor"""
        # 直接调用构造函数，它会自动初始化为全零
        return cls(shape=shape, dtype=dtype, device=device, device_id=device_id)

    @classmethod
    def zeros_like(cls, other: "Tensor"):
        """创建与other相同形状的全零tensor"""
        return cls.zeros(
            shape=other.shape(),
            dtype=other.dtype(),
            device=other.device_type(),
            device_id=other.device_id(),
        )

    @classmethod
    def from_torch(
        cls,
        tensor: "torch.Tensor",
        device: DeviceType = DeviceType.CPU,
        device_id: int = 0,
    ):
        """从PyTorch tensor创建Tensor"""
        import torch

        # 映射torch dtype到DataType
        dtype_map = {
            torch.float32: DataType.F32,
            torch.float16: DataType.F16,
            torch.bfloat16: DataType.BF16,
            torch.int64: DataType.I64,
            torch.int32: DataType.I32,
        }

        # 获取对应的DataType和处理tensor
        if tensor.dtype in dtype_map:
            tensor_dtype = dtype_map[tensor.dtype]
            torch_tensor = (
                tensor.cpu().contiguous()
                if tensor.device.type != "cpu"
                else tensor.contiguous()
            )
        else:
            # 默认转换为float32
            torch_tensor = tensor.float().cpu().contiguous()
            tensor_dtype = DataType.F32

        # 创建tensor（会自动初始化为全零）
        llaisys_tensor = cls(
            shape=torch_tensor.shape,
            dtype=tensor_dtype,
            device=device,
            device_id=device_id,
        )

        # 直接加载实际数据，覆盖全零初始化
        llaisys_tensor.load(torch_tensor.data_ptr())

        return llaisys_tensor
