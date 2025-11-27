from typing import Dict, Sequence, Iterable, Optional, Union, Callable
import torch
import pyarrow as pa
import numpy as np

class ColumnRefGPU:
    """
    Referência simbólica de coluna para DataStarGPUFrame.

    Exemplo:
        colg("age") > 30
        colg("score") <= 0.8
    """

    def __init__(self, name: str):
        self.name = name

    def _ensure_tensor(self, other, device: torch.device):
        if isinstance(other, torch.Tensor):
            return other.to(device)
        return torch.tensor(other, device=device)

    def _binary_op(self, other, op):
        def _fn(data: Dict[str, torch.Tensor]):
            if self.name not in data:
                raise KeyError(f"Coluna '{self.name}' não existe no DataStarGPUFrame.")
            col = data[self.name]
            other_t = self._ensure_tensor(other, col.device)
            return op(col, other_t)
        return _fn

    def __eq__(self, other):
        return self._binary_op(other, torch.eq)

    def __ne__(self, other):
        return self._binary_op(other, torch.ne)

    def __gt__(self, other):
        return self._binary_op(other, torch.gt)

    def __ge__(self, other):
        return self._binary_op(other, torch.ge)

    def __lt__(self, other):
        return self._binary_op(other, torch.lt)

    def __le__(self, other):
        return self._binary_op(other, torch.le)

    def isin(self, values: Sequence):
        def _fn(data: Dict[str, torch.Tensor]):
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
    DataFrame GPU-first minimalista para pipelines de ML.

    - Estado interno: dict[str, torch.Tensor] no device
    - Fábricas:
        - from_arrow_table(...)
    - Operações:
        - filter(...)
        - select(...)
        - batch(...)
        - to_tensor(...)
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = "cuda",
    ):
        if not isinstance(data, dict):
            raise TypeError("DataStarGPUFrame espera dict[str, torch.Tensor].")

        self.device = torch.device(device)
        length = None
        norm_data: Dict[str, torch.Tensor] = {}
        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Coluna '{k}' não é tensor.")
            # Move to device if not already, non_blocking for speed if possible
            t = v.to(self.device, non_blocking=True)
            if length is None:
                length = t.shape[0]
            else:
                if t.shape[0] != length:
                    raise ValueError(f"Todas as colunas devem ter o mesmo número de linhas. '{k}' tem {t.shape[0]}, esperado {length}.")
            norm_data[k] = t

        self._data = norm_data
        self._length = length or 0

    @classmethod
    def from_arrow_table(
        cls,
        table: pa.Table,
        device: Union[str, torch.device] = "cuda",
        numeric_only: bool = True,
    ) -> "DataStarGPUFrame":
        data: Dict[str, torch.Tensor] = {}

        for name in table.column_names:
            col = table[name]
            typ = col.type

            if numeric_only and not (pa.types.is_integer(typ) or pa.types.is_floating(typ) or pa.types.is_int8(typ)):
                continue

            # Conversão otimizada via Numpy para evitar overhead
            arr = col.to_numpy(zero_copy_only=False)

            if pa.types.is_integer(typ) or pa.types.is_int8(typ):
                # Use long for general integer compatibility in Torch, or specific types if needed
                # DataStarLoader used torch.long
                dtype = torch.long
            elif pa.types.is_floating(typ):
                dtype = torch.float32
            else:
                continue

            t = torch.from_numpy(arr).to(dtype)
            data[name] = t

        return cls(data, device=device)

    def filter(self, fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]):
        mask = fn(self._data)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("filter(GPU) deve retornar torch.Tensor bool.")
        if mask.dtype != torch.bool:
            raise TypeError("Máscara deve ser bool.")
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
        """
        Itera em mini-batches no próprio device.
        """
        if batch_size <= 0:
            raise ValueError("batch_size deve ser > 0.")
        n = self._length

        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n and drop_last:
                break
            end = min(end, n)
            # Slicing na GPU é muito rápido e view-based (geralmente)
            new_data = {k: v[start:end] for k, v in self._data.items()}
            yield self.__class__(new_data, device=self.device)

    def to_tensor(self, columns: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        if columns is None:
            return dict(self._data)
        out: Dict[str, torch.Tensor] = {}
        for c in columns:
            if c not in self._data:
                raise KeyError(f"Coluna '{c}' não existe no DataStarGPUFrame.")
            out[c] = self._data[c]
        return out

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._data[key]
        raise TypeError("Use select() or access columns by name.")
