import os
import glob
import time
import shutil
import queue
import threading
import torch
from typing import Optional, Iterable
from .frame import DataStarGPUFrame
from .format import DataStarFormat

class TieredCacheManager:
    """
    Manages data flow Cold Storage (Drive/S3) -> Hot Storage (Local NVMe).
    Runs in background to ensure local disk always has files ready.
    """
    def __init__(self, drive_pattern: str, local_cache_dir: str, max_cache_files: int = 3):
        self.drive_files = sorted(glob.glob(drive_pattern))
        self.local_dir = local_cache_dir
        self.max_cache_files = max_cache_files
        self.ready_queue = queue.Queue()
        self.stop_event = threading.Event()

        # Ensure local directory exists
        os.makedirs(self.local_dir, exist_ok=True)

    def _worker(self):
        files_iter = iter(self.drive_files)

        while not self.stop_event.is_set():
            # Backpressure: Monitor local disk for .star files
            current_local_files = glob.glob(os.path.join(self.local_dir, "*.star"))

            # If local buffer is full, wait for consumer
            if len(current_local_files) >= self.max_cache_files:
                time.sleep(0.1)
                continue

            try:
                # Get next file from remote list
                drive_path = next(files_iter)
                filename = os.path.basename(drive_path)
                local_path = os.path.join(self.local_dir, filename)
                temp_path = local_path + ".tmp"

                # Atomic Copy: Copy to temp then rename
                shutil.copy2(drive_path, temp_path)
                os.rename(temp_path, local_path)

                # Make available to consumer
                self.ready_queue.put(local_path)

            except StopIteration:
                # End of file list
                self.ready_queue.put(None)
                break
            except Exception as e:
                print(f"[DataStar Cache Error] {e}")
                break

    def start(self):
        """Starts the background thread."""
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        return self

    def get_next_file(self) -> Optional[str]:
        """Blocks until a file is available locally."""
        return self.ready_queue.get()

    def cleanup(self, filepath):
        """Removes file from local disk to free space."""
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


class DataStarStream:
    """
    Systolic Orchestrator.
    Reads from Cache Manager -> Decodes (.star mmap) -> Moves (GPU) -> Serves Batches.
    """
    def __init__(self, cache_manager: TieredCacheManager, device="cuda", queue_size=50):
        self.manager = cache_manager
        self.device = torch.device(device)
        self.batch_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

    def _processor_worker(self, batch_size):
        while not self.stop_event.is_set():
            # 1. Request local file (Hot Tier)
            local_path = self.manager.get_next_file()

            if local_path is None: # End of stream signal
                break

            try:
                # 2. Read and Convert using Native Format (DataStarFormat)
                # Replaces previous Parquet loading logic
                raw_data = DataStarFormat.load(local_path)
                gpu_frame = DataStarGPUFrame(raw_data, self.device)

                # 3. Batching and Enqueuing
                for batch in gpu_frame.batch(batch_size):
                    if self.stop_event.is_set(): break
                    self.batch_queue.put(batch)

            except Exception as e:
                print(f"[DataStar Stream Error] Failed processing {local_path}: {e}")

            finally:
                # 4. Automatic Cleanup (Space Management)
                self.manager.cleanup(local_path)

        # Signal end to iterator
        self.batch_queue.put(None)

    def iter_batches(self, batch_size: int) -> Iterable[DataStarGPUFrame]:
        """
        Main generator for the training loop.
        """
        self.stop_event.clear()

        # Start processing worker
        t = threading.Thread(target=self._processor_worker, args=(batch_size,), daemon=True)
        t.start()

        try:
            while True:
                batch = self.batch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            # Ensure clean shutdown if user breaks the loop
            self.stop_event.set()
            t.join(timeout=1.0)
