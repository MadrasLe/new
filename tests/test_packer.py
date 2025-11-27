import os
import numpy as np
import pytest
from datastar import DataStarPacker, DataStarFormat
import json
import torch

TXT_PATH = "test_packer.txt"
BIN_PATH = "test_packer.star"

@pytest.fixture
def setup_packer_data():
    with open(TXT_PATH, "w") as f:
        for i in range(100):
            f.write(json.dumps({"text": f"This is sentence number {i}. " * 10}) + "\n")

    yield

    if os.path.exists(TXT_PATH):
        os.remove(TXT_PATH)
    if os.path.exists(BIN_PATH):
        os.remove(BIN_PATH)
    if os.path.exists(BIN_PATH + ".tmp.raw"):
        os.remove(BIN_PATH + ".tmp.raw")

def test_packer_integration(setup_packer_data):
    # Tests the real DataStarPacker with multiprocessing producing .star format
    packer = DataStarPacker(TXT_PATH, BIN_PATH, "gpt2", n_workers=2)
    packer.run()

    assert os.path.exists(BIN_PATH)

    # Load using DataStarFormat to verify structure
    data = DataStarFormat.load(BIN_PATH)

    assert "tokens" in data
    tokens = data["tokens"]

    assert len(tokens) > 0
    # PyTorch 2.4+ supports uint32, but it might come as int32/int64 or proper uint32.
    # We just check it's an integer type.
    assert tokens.dtype in [torch.int32, torch.int64, torch.uint32]

    # Check if content looks reasonable (not empty)
    assert tokens.shape[0] > 0
