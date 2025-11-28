from __future__ import annotations
from typing import Dict, Iterable, Sequence, Optional, Union
import torch
import pyarrow as pa
import numpy as np

TensorDict = Dict[str, torch.Tensor]

class ColumnRefGPU:
    def __init__(self, name: str):
        self.name = name

    def _ensure_tensor(self, other, device: torch.device) -> torch.Tensor:
        if isinstance(other, torch.Tensor):
            return other.to(device)
        return torch.tensor(other, device=device)

    def _binary_op(self, other, op):
        def _fn(data: TensorDict) -> torch.Tensor:
            if self.name not in data:
                raise KeyError(f"Coluna '{self.name}' não existe no DataStarGPUFrame.")
            col = data[self.name]
            other_t = self._ensure_tensor(other, col.device)
            return op(col, other_t)
        return _fn

    def __eq__(self, other): return self._binary_op(other, torch.eq)
    def __ne__(self, other): return self._binary_op(other, torch.ne)
    def __gt__(self, other): return self._binary_op(other, torch.gt)
    def __ge__(self, other): return self._binary_op(other, torch.ge)
    def __lt__(self, other): return self._binary_op(other, torch.lt)
    def __le__(self, other): return self._binary_op(other, torch.le)

    def isin(self, values: Sequence):
        def _fn(data: TensorDict) -> torch.Tensor:
            if self.name not in data:
                raise KeyError(f"Coluna '{self.name}' não existe no DataStarGPUFrame.")
            col = data[self.name]
            vals = torch.tensor(list(values), device=col.device)
            return (col.unsqueeze(-1) == vals).any(dim=-1)
        return _fn


def colg(name: str) -> ColumnRefGPU:
    return ColumnRefGPU(name)


class DataStarGPUFrame:
    """
    DataFrame minimalista em GPU: dict[str, Tensor] + operações filter/select/batch.
    """
    def __init__(self, data: TensorDict, device: Union[str, torch.device] = "cuda"):
        if not isinstance(data, dict):
            raise TypeError("DataStarGPUFrame espera dict[str, torch.Tensor].")

        device = torch.device(device)
        length = None
        norm_data: TensorDict = {}

        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Coluna '{k}' não é tensor.")
            t = v.to(device)
            if length is None:
                length = t.shape[0]
            elif t.shape[0] != length:
                raise ValueError(f"Todas as colunas devem ter o mesmo número de linhas. '{k}' tem {t.shape[0]}, esperado {length}.")
            norm_data[k] = t

        self._data = norm_data
        self.device = device
        self._length = length or 0

    @classmethod
    def from_arrow_table(cls, table: pa.Table, device: Union[str, torch.device] = "cuda", numeric_only: bool = False) -> "DataStarGPUFrame":
        device = torch.device(device)
        data = {}
        for name in table.column_names:
            col = table[name]
            try:
                arr = col.to_numpy()
            except:
                arr = col.combine_chunks().to_numpy()

            t = None
            if pa.types.is_integer(col.type) or pa.types.is_int8(col.type) or pa.types.is_int16(col.type) or pa.types.is_int32(col.type) or pa.types.is_int64(col.type):
                t = torch.from_numpy(arr).long().to(device)
            elif pa.types.is_floating(col.type):
                t = torch.from_numpy(arr).float().to(device)

            if t is None and not numeric_only:
                 # Attempt generic conversion if not strictly numeric, though for GPU frame numeric is best
                 pass
            elif t is not None:
                data[name] = t

        return cls(data, device=device)

    def filter(self, fn) -> "DataStarGPUFrame":
        mask = fn(self._data)
        if not isinstance(mask, torch.Tensor) or mask.dtype is not torch.bool:
            raise TypeError("filter(GPU) deve retornar torch.Tensor bool.")
        if mask.shape[0] != self._length:
            raise ValueError("Máscara deve ter mesmo número de linhas.")
        new_data = {k: v[mask] for k, v in self._data.items()}
        return self.__class__(new_data, device=self.device)

    def select(self, columns: Sequence[str]) -> "DataStarGPUFrame":
        cols = list(columns)
        for c in cols:
            if c not in self._data:
                raise KeyError(f"Coluna '{c}' não existe no DataStarGPUFrame.")
        new_data = {c: self._data[c] for c in cols}
        return self.__class__(new_data, device=self.device)

    def batch(self, batch_size: int, drop_last: bool = False) -> Iterable["DataStarGPUFrame"]:
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0.")
        n = self._length
        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n and drop_last:
                break
            end = min(end, n)
            new_data = {k: v[start:end] for k, v in self._data.items()}
            yield self.__class__(new_data, device=self.device)

    def to_tensor(self, columns: Optional[Sequence[str]] = None) -> TensorDict:
        if columns is None:
            return dict(self._data)
        out: TensorDict = {}
        for c in columns:
            if c not in self._data:
                raise KeyError(f"Coluna '{c}' não existe no DataStarGPUFrame.")
            out[c] = self._data[c]
        return out

    # Compatibility alias
    def to_tensor_dict(self) -> TensorDict:
        return self.to_tensor()

    def __len__(self) -> int:
        return self._length
