# DataStar 🌟

**DataStar** is a high-performance, GPU-first data loading library designed for massive Machine Learning datasets. It combines a Python-based tiered caching system with a blazing fast C++ multi-threaded loader to keep your GPU fed without bottlenecks.

## 🚀 Key Features

*   **GPU-First Design**: `DataStarGPUFrame` manages data directly on the GPU, supporting high-speed filtering, selecting, and batching without CPU roundtrips.
*   **C++ Acceleration**: Includes a custom C++ extension (`datastar.loader_cpp`) that bypasses the Python GIL to parse headers and load tensors via `mmap` in background threads.
*   **Systolic Streaming**: `TieredCacheManager` automatically moves data from slow cold storage (S3, Network Drives) to fast local NVMe before the training loop needs it.
*   **Optimized Packing**: `DataStarPacker` compresses text datasets into compact `uint16` or `uint32` binary shards (`.star` format) based on vocabulary size.
*   **Zero-Copy Logic**: Utilizes memory mapping for efficient file I/O.

## 📦 Installation

DataStar requires **PyTorch** and a C++17 compatible compiler (e.g., `g++`).

```bash
# Clone the repository
git clone https://github.com/your-repo/datastar.git
cd datastar

# Install in editable mode (builds the C++ extension)
pip install -v -e .
```

## 🛠️ Usage

### 1. Ingest & Pack Data

Convert your raw text datasets (e.g., `.txt`, `.jsonl`) into the optimized `.star` binary format.

```python
from datastar.ingest.packer import DataStarPacker

# Automatically selects uint16 or uint32 based on vocab size
packer = DataStarPacker(
    txt_path="raw_data.jsonl",
    bin_path="data_shard_01.star",
    tokenizer_name="mistralai/Mistral-7B-v0.1"
)
packer.run()
```

### 2. High-Performance Loading (C++ Path)

For maximum performance, use the C++ loader path. This bypasses Python overhead for file reading and parsing.

```python
import torch
from datastar.io.stream import DataStarStream, TieredCacheManager

# Setup cache manager (optional for C++ path but good for file management)
manager = TieredCacheManager(
    drive_pattern="data/*.star",
    local_cache_dir="/tmp/hot_cache",
    max_cache_files=3
)

stream = DataStarStream(manager, device="cuda", use_cpp=True)

# List of files to process
files = ["data/shard_01.star", "data/shard_02.star"]

# Iterate over batches directly on GPU
for batch in stream.stream_cpp(files, batch_size=4096):
    # 'batch' is a DataStarGPUFrame
    input_ids = batch["tokens"]  # torch.Tensor on CUDA

    # Ready for model forward pass
    # logits = model(input_ids)
```

### 3. Flexible Streaming (Python Path)

Use the Python path if you need complex tiered caching logic or Parquet support.

```python
# Enable looping for multi-epoch training
manager = TieredCacheManager(
    drive_pattern="s3_mount/data/*.star",
    local_cache_dir="/local/nvme/cache",
    loop=True
)
manager.start()

stream = DataStarStream(manager, device="cuda")

for batch in stream.iter_batches(batch_size=2048):
    # Training loop
    pass

manager.stop_event.set()
```

### 4. GPU Dataframe Operations

`DataStarGPUFrame` allows you to manipulate loaded batches directly on the device.

```python
from datastar.core.frame import colg

# ... inside training loop ...
batch_frame = next(stream_iterator)

# Filter: Keep only samples where 'age' > 30 (executed on GPU)
filtered_batch = batch_frame.filter(colg("age") > 30)

# Select specific columns
tensors = filtered_batch.to_tensor(["age", "score"])
```

## 🏗️ Architecture

*   **Format (`.star`)**: A 64-bit size-prefixed JSON header followed by raw contiguous binary data.
*   **Loader (C++)**:
    *   Reads the header using `nlohmann/json`.
    *   Maps the file into memory.
    *   Creates PyTorch tensors from raw bytes (handling `uint16` -> `int32` promotion automatically).
    *   Slices tensors into batches and pushes them to a thread-safe queue.

## 🧪 Running Tests

```bash
python -m pytest datastar/tests/
```
