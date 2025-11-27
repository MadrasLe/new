import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
import shutil

class DataStarPacker:
    """
    DataStar Packer.
    Transforms raw text into contiguous binary (uint32) via Memory Mapping.
    Zero risk of RAM overflow, even with Terabyte datasets.
    """
    def __init__(self, txt_path, bin_path, tokenizer_name):
        self.txt_path = txt_path
        self.bin_path = bin_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def run(self):
        print(f"📦 DataStar Packer: Starting packing of {self.txt_path}...")

        # 1. First pass: Count exact tokens to allocate binary file
        print("🔍 Phase 1: Calculating final size (Pre-alloc)...")
        total_tokens = 0
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Counting"):
                if line:
                    # Quick estimate based on characters to avoid double tokenization
                    # Mistral average: ~3.5 chars per token
                    total_tokens += len(line) // 3

        # Safety margin of 10%
        alloc_size = int(total_tokens * 1.1)
        print(f"💾 Allocating virtual file of ~{alloc_size} tokens...")

        # 2. Create memory mapped file
        # Check if file exists to avoid error if we want to overwrite?
        # memmap w+ creates or overwrites.
        arr = np.memmap(self.bin_path, dtype=np.uint32, mode='w+', shape=(alloc_size,))

        # 3. Phase 2: Real Tokenization and Direct Disk Write
        print("⚡ Phase 2: Tokenizing and writing directly to disk...")
        idx = 0
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=alloc_size, unit="tok", desc="Packing")

            for line in f:
                try:
                    text = json.loads(line)
                    # If json is an object, assume we want a specific field?
                    if isinstance(text, dict):
                        # Fallback heuristic: look for 'text', 'content', or just str(text)
                        if 'text' in text:
                            text = text['text']
                        elif 'content' in text:
                            text = text['content']
                        else:
                            text = str(text)
                except:
                    text = line

                # Add EOS token if available
                eos = self.tokenizer.eos_token if self.tokenizer.eos_token is not None else ""
                text = str(text) + eos

                token_ids = self.tokenizer.encode(text, add_special_tokens=False)

                # Check space
                if idx + len(token_ids) > alloc_size:
                    print("⚠️ Expanding memmap (Estimate was low)...")
                    arr.flush() # Flush changes to disk

                    # Explicitly delete array to close handle (best effort for Windows/compat)
                    del arr

                    # Calculate new size
                    new_size = int(idx + len(token_ids) + 1e6)
                    new_bytes = new_size * np.dtype(np.uint32).itemsize

                    # Physically extend the file
                    with open(self.bin_path, "a") as file_handle:
                        os.ftruncate(file_handle.fileno(), new_bytes)

                    # Re-open memmap in r+ mode (read/write existing)
                    arr = np.memmap(self.bin_path, dtype=np.uint32, mode='r+', shape=(new_size,))
                    alloc_size = new_size

                # Write to array (SSD)
                arr[idx : idx + len(token_ids)] = token_ids
                idx += len(token_ids)
                pbar.update(len(token_ids))

        # 4. Finalize: Cut excess
        print(f"✂️ Finalizing: Trimming allocation excess. Real total: {idx}")
        arr.flush() # Ensure everything is written
        del arr # Close handle

        # Truncate file to exact size
        final_bytes = idx * np.dtype(np.uint32).itemsize
        with open(self.bin_path, "a") as file_handle:
             os.ftruncate(file_handle.fileno(), final_bytes)

        print(f"✅ Success! File '{self.bin_path}' ready for training.")
        print(f"📊 Final size: {os.path.getsize(self.bin_path) / 1024**2:.2f} MB")
