import os
import pytest
import torch
import numpy as np
from datastar.core.frame import DataStarGPUFrame
from datastar.io.format import DataStarFormat
from datastar.io.stream import DataStarStream, TieredCacheManager

# Try to import C++ extension
try:
    from datastar.loader_cpp import StarLoader
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

@pytest.fixture
def sample_data():
    return {
        "age": torch.tensor([25, 35, 45, 55], dtype=torch.int64),
        "score": torch.tensor([0.5, 0.8, 0.2, 0.9], dtype=torch.float32),
    }

def test_cpp_loader(sample_data, tmp_path):
    if not _HAS_CPP:
        pytest.skip("C++ extension not built/available")

    # Create .star files
    files = []
    for i in range(3):
        p = tmp_path / f"data_{i}.star"
        DataStarFormat.save(str(p), sample_data)
        files.append(str(p))

    # Initialize C++ loader directly
    batch_size = 2
    loader = StarLoader(files, batch_size, 10)

    batches = []
    while True:
        try:
            batch = loader.next()
            batches.append(batch)
        except Exception:
            break

    # 3 files * 4 rows = 12 rows. Batch size 2 => 6 batches.
    assert len(batches) == 6

    # Check data integrity
    b0 = batches[0]
    assert "age" in b0
    assert b0["age"].shape[0] == 2
    # First file, first batch: 25, 35
    assert b0["age"][0].item() == 25
    assert b0["age"][1].item() == 35

def test_stream_cpp_integration(sample_data, tmp_path):
    if not _HAS_CPP:
        pytest.skip("C++ extension not built/available")

    files = []
    for i in range(2):
        p = tmp_path / f"data_{i}.star"
        DataStarFormat.save(str(p), sample_data)
        files.append(str(p))

    manager = TieredCacheManager(str(tmp_path / "*.star"), str(tmp_path / "cache"))
    stream = DataStarStream(manager, device="cpu", use_cpp=True)

    batches = list(stream.stream_cpp(files, batch_size=2))
    assert len(batches) == 4
    assert isinstance(batches[0], DataStarGPUFrame)

def test_cpp_loader_uint16(tmp_path):
    if not _HAS_CPP:
        pytest.skip("C++ extension not built/available")

    # Manually create a file simulating uint16 packer output
    # Since DataStarFormat.save uses torch tensors, we need to hack it or use numpy
    # DataStarFormat uses numpy internally.

    import json
    import struct

    # 0, 65535, 12345, 32768
    data_uint16 = np.array([0, 65535, 12345, 32768], dtype=np.uint16)

    header = {
        "tokens": {
            "dtype": "uint16",
            "shape": [4],
            "offset": 0,
            "nbytes": data_uint16.nbytes
        }
    }

    header_bytes = json.dumps(header).encode('utf-8')
    filepath = tmp_path / "uint16.star"

    with open(filepath, 'wb') as f:
        f.write(struct.pack('<Q', len(header_bytes)))
        f.write(header_bytes)
        f.write(data_uint16.tobytes())

    loader = StarLoader([str(filepath)], 4, 10)
    batch = loader.next()

    t = batch["tokens"]
    # Check type promotion
    assert t.dtype == torch.int32

    # Check values
    assert t[0].item() == 0
    assert t[1].item() == 65535 # Should be positive, not -1
    assert t[2].item() == 12345
    assert t[3].item() == 32768 # Should be positive, not -32768

if __name__ == "__main__":
    pytest.main([__file__])
