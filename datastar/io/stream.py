import os
import glob
import time
import shutil
import queue
import threading
import torch
import pyarrow.parquet as pq
from typing import Optional, Iterable, Union, Callable

from datastar.core.frame import DataStarGPUFrame
from datastar.io.format import DataStarFormat

class TieredCacheManager:
    """
    Gerencia o fluxo de dados Cold Storage (Drive/S3) -> Hot Storage (Local NVMe).
    Roda em background para garantir que o disco local sempre tenha arquivos prontos.
    """
    def __init__(self, drive_pattern: str, local_cache_dir: str, max_cache_files: int = 3, extension: str = ".star", loop: bool = False):
        self.drive_files = sorted(glob.glob(drive_pattern))
        self.local_dir = local_cache_dir
        self.max_cache_files = max_cache_files
        self.ready_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.extension = extension
        self.loop = loop

        # Garante que o diretório local existe
        os.makedirs(self.local_dir, exist_ok=True)

    def _worker(self):
        while not self.stop_event.is_set():
            files_iter = iter(self.drive_files)

            for drive_path in files_iter:
                if self.stop_event.is_set():
                    break

                # Backpressure: Monitora o disco local
                while not self.stop_event.is_set():
                    current_local_files = glob.glob(os.path.join(self.local_dir, f"*{self.extension}"))
                    if len(current_local_files) < self.max_cache_files:
                        break
                    time.sleep(0.1)

                if self.stop_event.is_set():
                    break

                try:
                    filename = os.path.basename(drive_path)
                    local_path = os.path.join(self.local_dir, filename)

                    # Cópia/Download (se não for o mesmo caminho)
                    if os.path.abspath(drive_path) != os.path.abspath(local_path):
                        shutil.copy2(drive_path, local_path)

                    # Disponibiliza para o consumidor
                    self.ready_queue.put(local_path)

                except Exception as e:
                    print(f"[DataStar Cache Error] {e}")
                    break

            if not self.loop:
                break

        # Fim da lista de arquivos
        self.ready_queue.put(None)

    def start(self):
        """Inicia a thread de background."""
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()
        return self

    def get_next_file(self) -> Optional[str]:
        """Bloqueia até ter um arquivo disponível localmente."""
        return self.ready_queue.get()

    def cleanup(self, filepath):
        """Remove o arquivo do disco local para liberar espaço."""
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


class DataStarStream:
    """
    Orquestrador Sistólico.
    Lê do Cache Manager -> Decodifica (CPU) -> Move (GPU) -> Serve Batches.
    Suporta .star (DataStarFormat) e .parquet.
    """
    def __init__(self, cache_manager: TieredCacheManager, device: Union[str, torch.device] = None, queue_size=50):
        self.manager = cache_manager
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.batch_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()

    def _processor_worker(self, batch_size):
        while not self.stop_event.is_set():
            # 1. Pede arquivo local (Hot Tier)
            local_path = self.manager.get_next_file()

            if local_path is None: # Sinal de fim de stream
                break

            try:
                # 2. Leitura e Conversão
                if local_path.endswith(".star"):
                    raw_data = DataStarFormat.load(local_path)
                    gpu_frame = DataStarGPUFrame(raw_data, self.device)
                elif local_path.endswith(".parquet"):
                    table = pq.read_table(local_path)
                    gpu_frame = DataStarGPUFrame.from_arrow_table(table, self.device)
                else:
                    print(f"[DataStar Stream] Formato desconhecido: {local_path}")
                    self.manager.cleanup(local_path)
                    continue

                # 3. Batching e Enfileiramento
                for batch in gpu_frame.batch(batch_size):
                    if self.stop_event.is_set(): break
                    self.batch_queue.put(batch)

            except Exception as e:
                print(f"[DataStar Stream Error] {e}")

            finally:
                # 4. Limpeza Automática (Space Management)
                self.manager.cleanup(local_path)

        # Sinaliza fim para o iterador
        self.batch_queue.put(None)

    def iter_batches(self, batch_size: int) -> Iterable[DataStarGPUFrame]:
        """
        Gerador principal para o loop de treino.
        """
        self.stop_event.clear()

        # Inicia worker de processamento
        t = threading.Thread(target=self._processor_worker, args=(batch_size,), daemon=True)
        t.start()

        try:
            while True:
                batch = self.batch_queue.get()
                if batch is None:
                    break
                yield batch
        finally:
            # Garante shutdown limpo se o usuário quebrar o loop
            self.stop_event.set()
            t.join(timeout=1.0)
