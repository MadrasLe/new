import os
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import pytest
from datastar import DataStarGPUFrame, colg, DataStarStream, TieredCacheManager

# Define constants
PARQUET_PATH = "test_data.parquet"
BATCH_SIZE = 4096

def generate_parquet_data(path: str, n_rows: int = 10_000, seed: int = 42) -> None:
    if os.path.exists(path):
        os.remove(path)

    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 80, size=n_rows, dtype=np.int32)
    scores = rng.random(size=n_rows, dtype=np.float32)
    labels = rng.integers(0, 2, size=n_rows, dtype=np.int8)

    df = pd.DataFrame({
        "age": ages,
        "score": scores,
        "label": labels,
    })
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)

def test_datastar_pipeline():
    # Setup
    generate_parquet_data(PARQUET_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test DataStarGPUFrame directly (Parquet -> Arrow -> GPUFrame)
    table = pq.read_table(PARQUET_PATH)
    ds = DataStarGPUFrame.from_arrow_table(table, device=device)
    assert len(ds) == 10_000

    # Filter/Select
    ds2 = ds.filter(colg("age") > 30).select(["age", "score", "label"])
    assert len(ds2) < 10_000

    # Cleanup for stream test
    if os.path.exists(PARQUET_PATH):
        os.remove(PARQUET_PATH)

def test_stream_pipeline():
    # Setup mock environment for streaming
    local_dir = "test_cache"
    remote_pattern = "test_remote/*.parquet"

    # Create remote mock data
    os.makedirs("test_remote", exist_ok=True)
    generate_parquet_data("test_remote/part1.parquet")
    generate_parquet_data("test_remote/part2.parquet")

    # Cleanup local cache if exists
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    # Initialize Manager and Stream
    manager = TieredCacheManager(remote_pattern, local_dir)
    manager.start()

    stream = DataStarStream(manager, device="cpu", queue_size=5) # use cpu for CI/test safety

    count_batches = 0
    total_rows = 0

    for batch in stream.iter_batches(4096):
        count_batches += 1
        total_rows += len(batch)

    assert count_batches > 0
    assert total_rows == 20_000

    # Cleanup
    shutil.rmtree("test_remote")
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

import shutil
