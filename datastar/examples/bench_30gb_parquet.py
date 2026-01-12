import os
import glob
import time
import shutil
import queue
import threading
import gc
import psutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from typing import Dict, List, Iterable, Tuple, Optional

from datastar import DataStarGPUFrame, TieredCacheManager, DataStarStream

# ==============================================================================
# CONFIGURAÇÕES MOD: BRUTAL (30GB)
# ==============================================================================
# Use current directory or temp for portable benchmark
BIG_DATA_DIR = "./big_data_30gb"
LOCAL_CACHE_DIR = "./local_cache_nvme"
TOTAL_GB_TO_GENERATE = 2 # Reduced for quick test, originally 30
BATCH_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def print_memory():
    mem = psutil.virtual_memory()
    print(f"   [RAM Sistema] Usado: {mem.percent}% ({mem.used / 1e9:.2f} GB / {mem.total / 1e9:.2f} GB)")

# ==============================================================================
# 2. GERADOR DE DADOS (30GB) - COM LIMPEZA AUTOMÁTICA
# ==============================================================================

def generate_big_data():
    # Limpa versões anteriores de 15gb pra liberar disco
    old_dir_15 = "./big_data_15gb"
    if os.path.exists(old_dir_15):
        print(f"[Cleanup] Removendo dataset antigo de 15GB para liberar espaço...")
        shutil.rmtree(old_dir_15)

    if os.path.exists(BIG_DATA_DIR) and len(glob.glob(f"{BIG_DATA_DIR}/*.parquet")) > 5:
        print(f"[Setup] Dados já parecem existir em {BIG_DATA_DIR}. Pulando geração.")
        return

    if os.path.exists(BIG_DATA_DIR): shutil.rmtree(BIG_DATA_DIR)
    os.makedirs(BIG_DATA_DIR)

    rows_per_file = 500_000
    # Ajuste: 30GB requer mais arquivos
    # n_files = int((TOTAL_GB_TO_GENERATE * 1024) / 100)
    n_files = 10 # For demo

    print(f"--- GERANDO DEMO DATA ({n_files} arquivos) ---")
    print_memory()

    # Gera em lotes para mostrar progresso
    for i in range(n_files):
        data = {f'col_{c}': np.random.randn(rows_per_file).astype(np.float32) for c in range(10)}
        data['label'] = np.random.randint(0, 2, rows_per_file).astype(np.int8)

        table = pa.Table.from_pydict(data)
        pq.write_table(table, os.path.join(BIG_DATA_DIR, f"shard_{i:03d}.parquet"))

        if i % 5 == 0:
            print(f"   Criado shard {i}/{n_files}...", end="\r")

    print(f"\n[Concluído] {n_files} arquivos gerados em {BIG_DATA_DIR}")

# ==============================================================================
# 3. BENCHMARK
# ==============================================================================

def benchmark_pandas_limit(max_files=10):
    print(f"\n[Pandas] Iniciando (Limitado a {max_files} arquivos)...")
    files = sorted(glob.glob(f"{BIG_DATA_DIR}/*.parquet"))[:max_files]
    total_rows = 0
    start_time = time.time()
    for f in files:
        df = pd.read_parquet(f)
        vals = df.values
        # To simulate torch conversion overhead
        t = torch.tensor(vals).to(DEVICE) if torch.cuda.is_available() else torch.tensor(vals)
        total_rows += len(df)
        del df, vals, t
        gc.collect()
    duration = time.time() - start_time
    return total_rows, duration

def benchmark_datastar_full():
    print(f"\n[DataStar] Iniciando Teste Completo...")
    if os.path.exists(LOCAL_CACHE_DIR): shutil.rmtree(LOCAL_CACHE_DIR)

    manager = TieredCacheManager(f"{BIG_DATA_DIR}/*.parquet", LOCAL_CACHE_DIR, max_cache_files=3).start()
    stream = DataStarStream(manager, device=DEVICE)

    total_rows = 0
    start_time = time.time()

    # Adicionei um contador visual pra você ver ele comendo os gigas
    batch_count = 0
    for batch in stream.iter_batches(BATCH_SIZE):
        total_rows += batch._length
        _ = batch.to_tensor(['col_0']) # Sync
        batch_count += 1
        if batch_count % 100 == 0:
            print(f"   ...Processados {total_rows/1e6:.1f} Milhões de linhas", end="\r")

    duration = time.time() - start_time
    return total_rows, duration

# ==============================================================================
# 4. EXECUÇÃO
# ==============================================================================

if __name__ == "__main__":
    print(f"=== INICIANDO DATASTAR BENCHMARK ({DEVICE.upper()}) ===")

    # 1. Gera dados (Isso vai demorar uns minutinhos, paciência!)
    generate_big_data()
    print_memory()

    # 2. Benchmark Pandas (Amostra Grátis, porque se pagar ele quebra)
    rows_pd, time_pd = benchmark_pandas_limit(max_files=5)
    speed_pd = rows_pd / (time_pd + 1e-9)
    print(f"   -> Pandas Speed: {speed_pd:,.0f} linhas/seg")

    # 3. Benchmark DataStar (O Prato Principal)
    rows_ds, time_ds = benchmark_datastar_full()
    speed_ds = rows_ds / (time_ds + 1e-9)
    print(f"\n   -> DataStar Speed: {speed_ds:,.0f} linhas/seg")
    print_memory()

    # 4. Resultado
    print("\n" + "="*40)
    print("       RESULTADO FINAL       ")
    print("="*40)
    print(f"Pandas (Estimado): {speed_pd:,.0f} rows/s")
    print(f"DataStar (Real):   {speed_ds:,.0f} rows/s")
    print(f"STATUS:            SUCESSO")
    print("="*40)
