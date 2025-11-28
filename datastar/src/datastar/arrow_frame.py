from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Union
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import torch
import numpy as np

class ColumnRefArrow:
    def __init__(self, name: str):
        self.name = name

    def _binary_op(self, other, op_name: str):
        def _fn(table: pa.Table):
            col = table[self.name]
            other_scalar = other if isinstance(other, pa.Scalar) else pa.scalar(other)
            return getattr(pc, op_name)(col, other_scalar)
        return _fn

    def __eq__(self, other):
        return self._binary_op(other, "equal")

    def __ne__(self, other):
        return self._binary_op(other, "not_equal")

    def __gt__(self, other):
        return self._binary_op(other, "greater")

    def __ge__(self, other):
        return self._binary_op(other, "greater_equal")

    def __lt__(self, other):
        return self._binary_op(other, "less")

    def __le__(self, other):
        return self._binary_op(other, "less_equal")

    def isin(self, values: Sequence):
        def _fn(table: pa.Table):
            col = table[self.name]
            value_set = pa.array(values)
            return pc.is_in(col, value_set=value_set)
        return _fn


def col_arrow(name: str) -> ColumnRefArrow:
    return ColumnRefArrow(name)


class DataStarArrowFrame:
    def __init__(self, table: pa.Table, device: Union[str, torch.device] = "cpu"):
        if not isinstance(table, pa.Table):
            raise TypeError("DataStarArrowFrame espera um pyarrow.Table interno.")
        self._table = table
        self.device = torch.device(device)

    @classmethod
    def from_csv(
        cls,
        path: str,
        device: Union[str, torch.device] = "cpu",
        **read_csv_kwargs,
    ) -> "DataStarArrowFrame":
        table = pacsv.read_csv(path, **read_csv_kwargs)
        return cls(table, device=device)

    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        device: Union[str, torch.device] = "cpu",
    ) -> "DataStarArrowFrame":
        return cls(table, device=device)

    def filter(self, fn: Callable[[pa.Table], pa.Array]) -> "DataStarArrowFrame":
        mask = fn(self._table)
        if not isinstance(mask, (pa.Array, pa.ChunkedArray)):
            raise TypeError("filter (arrow) deve retornar pa.Array/ChunkedArray.")
        if mask.type != pa.bool_():
            raise TypeError("Máscara deve ser booleana.")
        filtered = self._table.filter(mask)
        return self.__class__(filtered, device=self.device)

    def select(self, columns: Sequence[str]) -> "DataStarArrowFrame":
        proj = self._table.select(list(columns))
        return self.__class__(proj, device=self.device)

    def to_tensor(
        self,
        columns: Optional[Sequence[str]] = None,
        dtypes: Optional[Dict[str, torch.dtype]] = None,
    ) -> Dict[str, torch.Tensor]:
        if columns is None:
            columns = self._table.column_names
        if dtypes is None:
            dtypes = {}
        tensors: Dict[str, torch.Tensor] = {}
        for c in columns:
            if c not in self._table.column_names:
                raise KeyError(f"Coluna '{c}' não existe no DataStarArrowFrame.")
            col = self._table[c]
            try:
                arr = col.to_numpy(zero_copy_only=False)
            except:
                arr = col.combine_chunks().to_numpy()

            dtype = dtypes.get(c)
            if dtype is None:
                if pa.types.is_integer(col.type):
                    dtype = torch.long
                elif pa.types.is_floating(col.type):
                    dtype = torch.float32
                else:
                    dtype = torch.float32

            tensors[c] = torch.from_numpy(arr).to(dtype=dtype, device=self.device)
        return tensors

    def __len__(self):
        return self._table.num_rows
