import torch
import pyarrow as pa
import numpy as np
from typing import Dict, Union, Iterable, Sequence, Optional

class ColumnSelector:
    """Helper class for column filtering in DataStarGPUFrame."""
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

def colg(name):
    """
    Creates a column selector for filtering.
    Usage: ds.filter(colg("age") > 30)
    """
    return ColumnSelector(name)

class DataStarGPUFrame:
    """
    DataStarGPUFrame: A GPU-first DataFrame for ML pipelines.

    - Internal state: dict[str, torch.Tensor] on device
    - Factory: from_arrow_table(...)
    - Operations: filter(...), select(...), batch(...), to_tensor(...)
    """

    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        device: Union[str, torch.device] = "cuda",
    ):
        if not isinstance(data, dict):
            raise TypeError("DataStarGPUFrame expects dict[str, torch.Tensor].")

        self.device = torch.device(device)
        length = None
        norm_data: Dict[str, torch.Tensor] = {}

        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"Column '{k}' is not a tensor.")
            t = v.to(self.device)
            if length is None:
                length = t.shape[0]
            else:
                if t.shape[0] != length:
                    raise ValueError("All columns must have the same number of rows.")
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

            if numeric_only and not (pa.types.is_integer(typ) or pa.types.is_floating(typ)):
                continue

            # Optimized conversion via Numpy to avoid overhead
            # zero_copy_only=False ensures we get a numpy array even if copy is needed
            arr = col.to_numpy(zero_copy_only=False)

            if pa.types.is_integer(typ) or pa.types.is_int8(typ):
                dtype = torch.long
            elif pa.types.is_floating(typ):
                dtype = torch.float32
            else:
                continue

            t = torch.from_numpy(arr).to(dtype)
            data[name] = t

        return cls(data, device=device)

    def filter(self, fn) -> "DataStarGPUFrame":
        """
        Applies a boolean mask returned by fn(self._data).
        """
        mask = fn(self._data)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("filter(GPU) must return a torch.Tensor bool.")
        if mask.dtype != torch.bool:
            raise TypeError("Mask must be bool.")
        if mask.shape[0] != self._length:
            raise ValueError("Mask must have the same number of rows.")
        new_data = {k: v[mask] for k, v in self._data.items()}
        return self.__class__(new_data, device=self.device)

    def select(self, columns: Sequence[str]) -> "DataStarGPUFrame":
        """
        Returns a new DataStarGPUFrame with only the selected columns.
        """
        cols = list(columns)
        for c in cols:
            if c not in self._data:
                raise KeyError(f"Column '{c}' does not exist in DataStarGPUFrame.")
        new_data = {c: self._data[c] for c in cols}
        return self.__class__(new_data, device=self.device)

    def batch(self, batch_size: int, drop_last: bool = False) -> Iterable["DataStarGPUFrame"]:
        """
        Yields mini-batches on the same device.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        n = self._length

        for start in range(0, n, batch_size):
            end = start + batch_size
            if end > n:
                if drop_last:
                    break
                end = n

            # Slicing tensors shares storage, minimal overhead
            new_data = {k: v[start:end] for k, v in self._data.items()}
            yield self.__class__(new_data, device=self.device)

    def to_tensor(self, columns: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
        """
        Returns the dictionary of tensors.
        If columns is specified, returns only those columns.
        """
        if columns is None:
            return dict(self._data)
        out: Dict[str, torch.Tensor] = {}
        for c in columns:
            if c not in self._data:
                raise KeyError(f"Column '{c}' does not exist in DataStarGPUFrame.")
            out[c] = self._data[c]
        return out

    def to_tensor_dict(self) -> Dict[str, torch.Tensor]:
        """Alias for to_tensor() without arguments, for compatibility."""
        return self.to_tensor()

    def __len__(self):
        return self._length
