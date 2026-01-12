from __future__ import annotations
import os
import glob
import time
import shutil
import queue
import threading
from typing import Optional, Iterable
import pyarrow.parquet as pq
import torch

from .gpu_frame import DataStarGPUFrame

class TieredCacheManager:
    """
    Cold storage (S3/Drive/parquet_dir) -> hot NVMe local.
    Roda em background e entrega paths prontos via queue.
    """
    def __init__(self, drive_pattern: str, local_cache_dir: str, max_cache_files: int = 3):
        self.drive_files = sorted(glob.glob(drive_pattern))
        self.local_dir = local_cache_dir
        self.max_cache_files = max_cache_files
        self.ready_queue: queue.Queue[Optional[str]] = queue.Queue()
        self.stop_event = threading.Event()
        os.makedirs(self.local_dir, exist_ok=True)

    def _worker(self) -> None:
        files_iter = iter(self.drive_files)
        while not self.stop_event.is_set():
            current_local_files = glob.glob(os.path.join(self.local_dir, "*.parquet"))
            if len(current_local_files) >= self.max_cache_files:
                time.sleep(0.1)
                continue
            try:
                drive_path = next(files_iter)
                filename = os.path.basename(drive_path)
                local_path = os.path.join(self.local_dir, filename)
                shutil.copy2(drive_path, local_path)
                self.ready_queue.put(local_path)
            except StopIteration:
                self.ready_queue.put(None)
                break
            except Exception as e:
                print(f"[DataStar Cache Error] {e}")
                break

    def start(self) -> "TieredCacheManager":
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        return self

    def get_next_file(self) -> Optional[str]:
        return self.ready_queue.get()

    def cleanup(self, filepath: str) -> None:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


class DataStarStream:
    """
    Lê Parquet -> Arrow -> DataStarGPUFrame -> mini-batches em GPU.
    Pensado para laços de treino longos com disco grande.
    """
    def __init__(self, cache_manager: TieredCacheManager, device: str = "cuda", queue_size: int = 50):
        self.manager = cache_manager
        self.device = torch.device(device)
        self.batch_queue: queue.Queue[Optional[DataStarGPUFrame]] = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

    def _processor_worker(self, batch_size: int) -> None:
        while not self.stop_event.is_set():
            local_path = self.manager.get_next_file()
            if local_path is None:
                break
            try:
                table = pq.read_table(local_path)
                gpu_frame = DataStarGPUFrame.from_arrow_table(table, device=self.device, numeric_only=True)
                for batch in gpu_frame.batch(batch_size):
                    if self.stop_event.is_set():
                        break
                    self.batch_queue.put(batch)
            except Exception as e:
                print(f"[DataStar Stream Error] {e}")
            finally:
                self.manager.cleanup(local_path)
        self.batch_queue.put(None)

    def iter_batches(self, batch_size: int) -> Iterable[DataStarGPUFrame]:
        self.stop_event.clear()
        t = threading.Thread(target=self._processor_worker, args=(batch_size,), daemon=True)
        t.start()
        try:
            while True:
                batch = self.batch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            self.stop_event.set()
            t.join(timeout=1.0)
