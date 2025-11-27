import os
import shutil
import pytest
import torch
import pyarrow as pa
import pandas as pd
import numpy as np
import time
from datastar.core.frame import DataStarGPUFrame, colg
from datastar.io.format import DataStarFormat
from datastar.io.stream import TieredCacheManager, DataStarStream

@pytest.fixture
def sample_data():
    return {
        "age": torch.tensor([25, 35, 45, 55], dtype=torch.long),
        "score": torch.tensor([0.5, 0.8, 0.2, 0.9], dtype=torch.float32),
        "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
    }

def test_frame_creation(sample_data):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    df = DataStarGPUFrame(sample_data, device=device)
    assert len(df) == 4
    assert "age" in df.to_tensor()
    assert df["age"].device.type == device

def test_frame_filter(sample_data):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    df = DataStarGPUFrame(sample_data, device=device)
    filtered = df.filter(colg("age") > 30)

    assert len(filtered) == 3
    # Check values
    t = filtered.to_tensor()
    assert (t["age"] > 30).all()

def test_frame_select(sample_data):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    df = DataStarGPUFrame(sample_data, device=device)
    selected = df.select(["age", "label"])

    assert "score" not in selected.to_tensor()
    assert "age" in selected.to_tensor()

def test_frame_batch(sample_data):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    df = DataStarGPUFrame(sample_data, device=device)
    batches = list(df.batch(2))
    assert len(batches) == 2
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2

def test_format_save_load(sample_data, tmp_path):
    device = "cpu"
    filepath = tmp_path / "test.star"

    # Save
    DataStarFormat.save(str(filepath), sample_data)
    assert filepath.exists()

    # Load
    loaded = DataStarFormat.load(str(filepath))
    assert loaded.keys() == sample_data.keys()
    assert torch.all(loaded["age"] == sample_data["age"])

def test_stream_integration(sample_data, tmp_path):
    # Setup directories
    drive_dir = tmp_path / "drive"
    local_dir = tmp_path / "local"
    drive_dir.mkdir()

    # Create dummy files
    for i in range(3):
        DataStarFormat.save(str(drive_dir / f"data_{i}.star"), sample_data)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    manager = TieredCacheManager(
        drive_pattern=str(drive_dir / "*.star"),
        local_cache_dir=str(local_dir),
        max_cache_files=2
    )
    manager.start()

    stream = DataStarStream(manager, device=device)

    batches_seen = 0
    for batch in stream.iter_batches(batch_size=2):
        assert isinstance(batch, DataStarGPUFrame)
        batches_seen += 1

    # 3 files * (4 rows / 2 batch_size) = 6 batches
    assert batches_seen == 6

    # Cleanup
    manager.stop_event.set()

def test_stream_epochs(sample_data, tmp_path):
    # Setup directories
    drive_dir = tmp_path / "drive_epochs"
    local_dir = tmp_path / "local_epochs"
    drive_dir.mkdir()
    local_dir.mkdir()

    # Create dummy files
    for i in range(2):
        DataStarFormat.save(str(drive_dir / f"data_{i}.star"), sample_data)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Enable looping
    manager = TieredCacheManager(
        drive_pattern=str(drive_dir / "*.star"),
        local_cache_dir=str(local_dir),
        max_cache_files=2,
        loop=True
    )
    manager.start()

    stream = DataStarStream(manager, device=device)

    batches_seen = 0
    # Simulate stopping after some batches (e.g. 2 epochs worth)
    # 2 files * 2 batches/file = 4 batches per epoch.
    # We want to see more than 4 batches.

    for batch in stream.iter_batches(batch_size=2):
        batches_seen += 1
        if batches_seen >= 8: # 2 full epochs
            break

    assert batches_seen == 8

    # Cleanup
    manager.stop_event.set()

def test_from_arrow_table():
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [1.1, 2.2, 3.3],
        "c": ["x", "y", "z"]
    })
    table = pa.Table.from_pandas(df)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    ds = DataStarGPUFrame.from_arrow_table(table, device=device)

    # "c" should be ignored as it is string
    assert "a" in ds.to_tensor()
    assert "b" in ds.to_tensor()
    assert "c" not in ds.to_tensor()

    assert ds["a"].dtype == torch.long
    assert ds["b"].dtype == torch.float32

if __name__ == "__main__":
    pytest.main([__file__])
