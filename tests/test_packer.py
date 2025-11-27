import os
import numpy as np
import pytest
from datastar import DataStarPacker
import json

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

class MockDataStarPacker(DataStarPacker):
    def run(self):
        print(f"📦 DataStar Packer: Starting packing of {self.txt_path}...")

        # Force small allocation to trigger resize
        alloc_size = 100 # Very small
        print(f"💾 Allocating virtual file of ~{alloc_size} tokens (FORCED SMALL)...")

        arr = np.memmap(self.bin_path, dtype=np.uint32, mode='w+', shape=(alloc_size,))

        idx = 0
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    text = json.loads(line)
                    if isinstance(text, dict):
                        if 'text' in text:
                            text = text['text']
                        elif 'content' in text:
                            text = text['content']
                        else:
                            text = str(text)
                except:
                    text = line

                # Handle None EOS token
                eos = self.tokenizer.eos_token if self.tokenizer.eos_token is not None else ""
                text = str(text) + eos

                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                if idx + len(token_ids) > alloc_size:
                    print("⚠️ Expanding memmap (Estimate was low)...")
                    arr.flush()
                    del arr # Explicit delete for safety

                    new_size = int(idx + len(token_ids) + 1000)
                    new_bytes = new_size * np.dtype(np.uint32).itemsize

                    with open(self.bin_path, "a") as file_handle:
                        os.ftruncate(file_handle.fileno(), new_bytes)

                    arr = np.memmap(self.bin_path, dtype=np.uint32, mode='r+', shape=(new_size,))
                    alloc_size = new_size

                arr[idx : idx + len(token_ids)] = token_ids
                idx += len(token_ids)

        print(f"✂️ Finalizing: Trimming allocation excess. Real total: {idx}")
        arr.flush()
        del arr # Explicit delete

        final_bytes = idx * np.dtype(np.uint32).itemsize
        with open(self.bin_path, "a") as file_handle:
             os.ftruncate(file_handle.fileno(), final_bytes)

def test_packer_resize(setup_packer_data):
    # Using gpt2 as it is smaller and standard
    packer = MockDataStarPacker(TXT_PATH, BIN_PATH, "gpt2")
    packer.run()

    assert os.path.exists(BIN_PATH)
    file_size = os.path.getsize(BIN_PATH)
    assert file_size > 0
    assert file_size % 4 == 0 # uint32 is 4 bytes

    arr = np.memmap(BIN_PATH, dtype=np.uint32, mode='r')
    assert len(arr) > 0
