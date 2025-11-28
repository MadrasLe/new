from __future__ import annotations
import os
import json
import struct
import mmap
from typing import Dict
import torch
import numpy as np

class DataStarFormat:
    """
    Formato binário .star com header JSON + payload contíguo.
    Pensado para leitura via mmap e conversão rápida para tensores Torch.
    """

    @staticmethod
    def save(filepath: str, data: Dict[str, torch.Tensor]) -> None:
        header = {}
        offset = 0
        cpu_arrays = []

        for name, tensor in data.items():
            if tensor.is_cuda:
                tensor = tensor.cpu()
            npy = tensor.detach().numpy()
            if not npy.flags['C_CONTIGUOUS']:
                npy = np.ascontiguousarray(npy)

            cpu_arrays.append(npy)
            header[name] = {
                "dtype": str(npy.dtype),
                "shape": npy.shape,
                "offset": offset,
                "nbytes": npy.nbytes,
            }
            offset += npy.nbytes

        header_bytes = json.dumps(header).encode("utf-8")

        with open(filepath, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
            for arr in cpu_arrays:
                f.write(arr.tobytes())

    @staticmethod
    def load(filepath: str) -> Dict[str, torch.Tensor]:
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 8:
            raise FileNotFoundError(f"Arquivo inválido ou inexistente: {filepath}")

        with open(filepath, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            header_len = struct.unpack("<Q", mm[:8])[0]
            header = json.loads(mm[8: 8 + header_len].decode("utf-8"))
            data_start = 8 + header_len

            result: Dict[str, torch.Tensor] = {}
            for name, meta in header.items():
                dtype = np.dtype(meta["dtype"])
                start = data_start + meta["offset"]
                count = int(np.prod(meta["shape"]))

                view = np.frombuffer(mm, dtype=dtype, offset=start, count=count)
                view = view.reshape(meta["shape"])
                result[name] = torch.from_numpy(view.copy())
                del view

            mm.close()
        return result
