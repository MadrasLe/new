import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
import shutil
import struct
import multiprocessing as mp
from functools import partial

# Global tokenizer for workers
_tokenizer = None

def _init_worker(tokenizer_name):
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

def _process_batch(lines):
    """
    Process a batch of lines using the global tokenizer.
    Returns a numpy array of tokens (uint32).
    """
    global _tokenizer
    all_tokens = []
    eos = _tokenizer.eos_token if _tokenizer.eos_token is not None else ""

    for line in lines:
        try:
            text = json.loads(line)
            if isinstance(text, dict):
                if 'text' in text: text = text['text']
                elif 'content' in text: text = text['content']
                else: text = str(text)
        except:
            text = line

        text = str(text) + eos
        token_ids = _tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(token_ids)

    return np.array(all_tokens, dtype=np.uint32)

class DataStarPacker:
    """
    DataStar Packer (Parallel).
    Transforms raw text into a DataStar (.star) binary file.
    Structure: [HeaderSize u64][Header JSON][Data]
    """
    def __init__(self, txt_path, bin_path, tokenizer_name, n_workers=None):
        self.txt_path = txt_path
        self.bin_path = bin_path
        self.tokenizer_name = tokenizer_name
        self.n_workers = n_workers or min(os.cpu_count(), 8)

    def run(self):
        print(f"📦 DataStar Packer: Starting parallel packing of {self.txt_path}...")

        # We need to accumulate tokens into a temporary raw binary file first
        temp_bin_path = self.bin_path + ".tmp.raw"
        with open(temp_bin_path, "wb") as f:
            pass

        pool = mp.Pool(processes=self.n_workers, initializer=_init_worker, initargs=(self.tokenizer_name,))

        chunk_size = 1000
        batch = []
        futures = []

        print(f"🚀 Launching {self.n_workers} workers...")

        total_tokens = 0

        with open(self.txt_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading"):
                batch.append(line)
                if len(batch) >= chunk_size:
                    futures.append(pool.apply_async(_process_batch, (batch,)))
                    batch = []

                    if len(futures) > self.n_workers * 4:
                        total_tokens += self._collect_and_write(futures, temp_bin_path)
                        futures = []

            if batch:
                futures.append(pool.apply_async(_process_batch, (batch,)))

            total_tokens += self._collect_and_write(futures, temp_bin_path)

        pool.close()
        pool.join()

        # Now create the final .star file with header
        print("🔨 Finalizing .star format...")
        self._finalize_star_file(temp_bin_path, total_tokens)

        # Cleanup
        if os.path.exists(temp_bin_path):
            os.remove(temp_bin_path)

        print(f"✅ Success! File '{self.bin_path}' ready.")
        print(f"📊 Final size: {os.path.getsize(self.bin_path) / 1024**2:.2f} MB")

    def _collect_and_write(self, futures, output_path):
        """Collects results and appends to temp file."""
        count = 0
        with open(output_path, "ab") as f:
            for future in futures:
                try:
                    token_array = future.get()
                    f.write(token_array.tobytes())
                    count += len(token_array)
                except Exception as e:
                    print(f"❌ Worker Error: {e}")
        return count

    def _finalize_star_file(self, raw_data_path, total_tokens):
        """Prepend header to the raw data."""
        # Header structure
        header = {
            "tokens": {
                "dtype": "uint32",
                "shape": (total_tokens,),
                "offset": 0,
                "nbytes": total_tokens * 4
            }
        }
        header_bytes = json.dumps(header).encode('utf-8')

        with open(self.bin_path, 'wb') as f_out:
            # Write Header Size
            f_out.write(struct.pack('<Q', len(header_bytes)))
            # Write Header
            f_out.write(header_bytes)

            # Write Body (Copy from temp file)
            # Efficient copy using shutil.copyfileobj would be better but we need to append.
            # We can stream read/write.
            chunk_size = 1024 * 1024 * 64 # 64MB chunks
            with open(raw_data_path, 'rb') as f_in:
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
