import torch
import pyarrow as pa
import numpy as np
from typing import Dict, Union, Iterable, Sequence, Optional

class colg:
    """Helper for column filtering logic matching the user's DSL."""
    def __init__(self, col_name):
        self.col_name = col_name

    def __gt__(self, other):
        return lambda data: data[self.col_name] > other

    def __lt__(self, other):
        return lambda data: data[self.col_name] < other

    def __ge__(self, other):
        return lambda data: data[self.col_name] >= other

    def __le__(self, other):
        return lambda data: data[self.col_name] <= other

    def __eq__(self, other):
        return lambda data: data[self.col_name] == other

    def __ne__(self, other):
        return lambda data: data[self.col_name] != other

class DataStarGPUFrame:
    """
    DataFrame otimizado para GPU. Mantém dados já na VRAM.
    Implementation matches DataStarLoader.txt.
    """
    def __init__(self, data: Dict[str, torch.Tensor], device: Union[str, torch.device]):
        self.device = torch.device(device)
        self._data = data
        # Assume coerência no tamanho das colunas
        self._length = next(iter(data.values())).shape[0] if data else 0

    @classmethod
    def from_arrow_table(cls, table: pa.Table, device: Union[str, torch.device]) -> "DataStarGPUFrame":
        data = {}
        # Ensure device is torch.device
        device = torch.device(device)

        for name in table.column_names:
            col = table[name]
            # Conversão otimizada via Numpy para evitar overhead
            arr = col.to_numpy()

            # Mapeamento de tipos Arrow -> Torch
            if pa.types.is_integer(col.type) or pa.types.is_int8(col.type):
                t = torch.from_numpy(arr).long().to(device)
            elif pa.types.is_floating(col.type):
                t = torch.from_numpy(arr).float().to(device)
            else:
                continue # Tipos não suportados são ignorados por enquanto

            data[name] = t
        return cls(data, device)

    def filter(self, fn) -> "DataStarGPUFrame":
        mask = fn(self._data)
        new_data = {k: v[mask] for k, v in self._data.items()}
        return self.__class__(new_data, self.device)

    def select(self, columns: Sequence[str]) -> "DataStarGPUFrame":
        new_data = {c: self._data[c] for c in columns}
        return self.__class__(new_data, self.device)

    def batch(self, batch_size: int, drop_last: bool = False) -> Iterable["DataStarGPUFrame"]:
        """Gera mini-batches (slicing) sem copiar memória."""
        for start in range(0, self._length, batch_size):
            end = start + batch_size
            if drop_last and end > self._length:
                break
            end = min(end, self._length)

            batch_data = {k: v[start:end] for k, v in self._data.items()}
            yield DataStarGPUFrame(batch_data, self.device)

    def to_tensor(self, columns: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """Retorna o dicionário de tensores puros."""
        if columns:
            return {k: self._data[k] for k in columns}
        return self._data

    def __len__(self):
        return self._length
