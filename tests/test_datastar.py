import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytest
from datastar import DataStarGPUFrame, colg, DataStarFormat

# Define constants
STAR_PATH = "test_data.star"
BATCH_SIZE = 4096

def generate_star_data(path: str, n_rows: int = 10_000, seed: int = 42) -> None:
    if os.path.exists(path):
        os.remove(path)

    rng = np.random.default_rng(seed)
    ages = rng.integers(18, 80, size=n_rows, dtype=np.int32)
    scores = rng.random(size=n_rows, dtype=np.float32)
    labels = rng.integers(0, 2, size=n_rows, dtype=np.int8)

    # Convert to torch for DataStarFormat
    data = {
        "age": torch.from_numpy(ages),
        "score": torch.from_numpy(scores),
        "label": torch.from_numpy(labels),
    }

    DataStarFormat.save(path, data)

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
    generate_star_data(STAR_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Read using DataStarFormat
    # Note: We are testing manual loading here,
    # but DataStarStream also uses DataStarFormat.load internally now.

    raw_data = DataStarFormat.load(STAR_PATH)

    # Initialize DataStarGPUFrame
    ds = DataStarGPUFrame(raw_data, device=device)
    assert len(ds) == 10_000

    # Filter and Select
    ds2 = ds.filter(colg("age") > 30).select(["age", "score", "label"])

    # Check if filtering worked (approx check since we use random data)
    assert len(ds2) < 10_000

    # Model
    model = TinyMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    model.train()
    for batch in ds2.batch(BATCH_SIZE, drop_last=False):
        t = batch.to_tensor(["age", "score", "label"])

        # Verify types
        assert t["age"].device == device

        X = torch.stack([t["age"].float(), t["score"].float()], dim=-1)
        y = t["label"].long()

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    print("Training loop completed successfully.")

    # Cleanup
    if os.path.exists(STAR_PATH):
        os.remove(STAR_PATH)

if __name__ == "__main__":
    test_datastar_pipeline()
