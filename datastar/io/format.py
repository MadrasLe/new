import json
import struct
import mmap
import os
import torch
import numpy as np
from typing import Dict

class DataStarFormat:
    """
    Formato binário proprietário de alta performance via Memory Mapping.
    Estrutura: [Header Size (u64)] + [JSON Header] + [Raw Bytes...]
    """
    @staticmethod
    def save(filepath: str, data: Dict[str, torch.Tensor]):
        header = {}
        offset = 0
        cpu_arrays = []

        # 1. Prepara metadados
        for name, tensor in data.items():
            if tensor.is_cuda: tensor = tensor.cpu()
            # Garante contiguidade na memória pra escrita ser linear
            npy = tensor.numpy()
            if not npy.flags['C_CONTIGUOUS']:
                npy = np.ascontiguousarray(npy)

            cpu_arrays.append(npy)

            header[name] = {
                'dtype': str(npy.dtype),
                'shape': npy.shape,
                'offset': offset,
                'nbytes': npy.nbytes
            }
            offset += npy.nbytes

        header_bytes = json.dumps(header).encode('utf-8')

        # 2. Grava no disco
        with open(filepath, 'wb') as f:
            f.write(struct.pack('<Q', len(header_bytes)))
            f.write(header_bytes)
            for arr in cpu_arrays:
                f.write(arr.tobytes())

    @staticmethod
    def load(filepath: str) -> Dict[str, torch.Tensor]:
        """Lê arquivo .star instantaneamente via mmap."""
        # Se o arquivo não existir ou estiver vazio, retorna vazio
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 8:
            raise FileNotFoundError(f"Arquivo inválido ou inexistente: {filepath}")

        with open(filepath, 'rb') as f:
            # Mapeia arquivo na memória (Zero-Copy Read)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # Parse Header
            header_len = struct.unpack('<Q', mm[:8])[0]
            header = json.loads(mm[8 : 8 + header_len].decode('utf-8'))
            data_start = 8 + header_len

            result = {}
            for name, meta in header.items():
                dtype = np.dtype(meta['dtype'])
                start = data_start + meta['offset']
                count = int(np.prod(meta['shape']))

                # Cria view numpy apontando pro disco/cache
                view = np.frombuffer(mm, dtype=dtype, offset=start, count=count)
                view = view.reshape(meta['shape'])

                # Conversão para Tensor (com copy para segurança de memória e compatibilidade Torch)
                # O custo desse copy é ínfimo comparado ao parse do Parquet
                result[name] = torch.from_numpy(view.copy())

                # IMPORTANT: Delete view to release buffer reference before closing mmap
                del view

            # Close mmap explicitly
            mm.close()
            return result
