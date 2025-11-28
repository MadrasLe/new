from __future__ import annotations
from typing import Callable, Dict, Optional, Sequence, Union
import pandas as pd
import torch
import numpy as np

class ColumnRef:
    def __init__(self, name: str):
        self.name = name

    def _binary_op(self, other, op):
        def _fn(df: pd.DataFrame):
            return op(df[self.name], other)
        return _fn

    def __eq__(self, other):
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._binary_op(other, lambda a, b: a != b)

    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b)

    def __lt__(self, other):
        return self._binary_op(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._binary_op(other, lambda a, b: a <= b)

    def isin(self, values: Sequence):
        def _fn(df: pd.DataFrame):
            return df[self.name].isin(values)
        return _fn

def col(name: str) -> ColumnRef:
    return ColumnRef(name)

class DataStarFrame:
    def __init__(self, df: pd.DataFrame, device: Union[str, torch.device] = "cpu"):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("DataStarFrame espera um pandas.DataFrame interno.")
        self._df = df.reset_index(drop=True)
        self.device = torch.device(device)

    @classmethod
    def from_csv(
        cls,
        path: str,
        device: Union[str, torch.device] = "cpu",
        **read_csv_kwargs,
    ) -> "DataStarFrame":
        df = pd.read_csv(path, **read_csv_kwargs)
        return cls(df, device=device)

    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        device: Union[str, torch.device] = "cpu",
    ) -> "DataStarFrame":
        return cls(df, device=device)

    def filter(self, fn: Callable[[pd.DataFrame], pd.Series]) -> "DataStarFrame":
        mask = fn(self._df)
        if not isinstance(mask, pd.Series):
            raise TypeError("filter deve retornar um pd.Series booleano.")
        if mask.dtype != bool:
            raise TypeError("Máscara de filter deve ser bool.")
        new_df = self._df[mask].reset_index(drop=True)
        return self.__class__(new_df, device=self.device)

    def select(self, columns: Sequence[str]) -> "DataStarFrame":
        new_df = self._df[list(columns)].copy()
        return self.__class__(new_df, device=self.device)

    def to_tensor(
        self,
        columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if columns is None:
            columns = list(self._df.columns)

        tensors: Dict[str, torch.Tensor] = {}
        for c in columns:
            if c not in self._df.columns:
                raise KeyError(f"Coluna '{c}' não existe no DataStarFrame.")
            series = self._df[c]
            if series.dtype == object:
                raise TypeError(f"Coluna '{c}' é object; converta antes.")
            arr = series.to_numpy()

            if pd.api.types.is_integer_dtype(series.dtype):
                dtype = torch.long
            elif pd.api.types.is_float_dtype(series.dtype):
                dtype = torch.float32
            else:
                dtype = torch.float32

            tensors[c] = torch.tensor(arr, dtype=dtype, device=self.device)
        return tensors

    def __len__(self):
        return len(self._df)
