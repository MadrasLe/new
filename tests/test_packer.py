import os
import numpy as np
import pytest
from datastar import DataStarPacker
import json
import torch

TXT_PATH = "test_packer.txt"
BIN_PATH = "test_packer.bin"

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

def test_packer_original_behavior(setup_packer_data):
    # Tests the ORIGINAL single-threaded DataStarPacker producing flat uint16 binary
    packer = DataStarPacker(TXT_PATH, BIN_PATH, "gpt2")
    packer.run()

    assert os.path.exists(BIN_PATH)

    # Verify it is a raw flat binary file of uint16
    file_size = os.path.getsize(BIN_PATH)
    assert file_size > 0
    assert file_size % 2 == 0 # uint16 = 2 bytes

    # Verify we can read it back as numpy array
    arr = np.memmap(BIN_PATH, dtype=np.uint16, mode='r')
    assert len(arr) > 0

    # Since we used GPT2 tokenizer, verify values are in range
    assert np.max(arr) < 65535
