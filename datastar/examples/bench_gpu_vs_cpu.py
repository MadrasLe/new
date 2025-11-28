import os
import time
import numpy as np
import pandas as pd
import pyarrow.csv as pacsv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datastar import DataStarGPUFrame, colg

CSV_PATH = "benchmark_data.csv"

# =========================
# Geração do CSV (se precisar)
# =========================

def generate_csv_if_needed(
    path: str,
    n_rows: int = 5_000_000,
    seed: int = 42,
) -> None:
    if os.path.exists(path):
        print(f"[gen] CSV já existe em '{path}', pulando geração.")
        return

    print(f"[gen] Gerando CSV sintético com {n_rows} linhas em '{path}'...")
    rng = np.random.default_rng(seed)

    ages = rng.integers(18, 80, size=n_rows, dtype=np.int32)
    scores = rng.random(size=n_rows, dtype=np.float32)
    labels = rng.integers(0, 2, size=n_rows, dtype=np.int8)

    df = pd.DataFrame(
        {
            "age": ages,
            "score": scores,
            "label": labels,
        }
    )

    df.to_csv(path, index=False)
    print("[gen] CSV criado.")


# =========================
# Loaders / pipelines
# =========================

def make_cpu_loader(csv_path: str, batch_size: int):
    """
    Pipeline clássico:
        pandas.read_csv -> filtro -> numpy -> tensor CPU -> DataLoader
    """
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    df = df[df["age"] > 30][["age", "score", "label"]]

    X = np.stack(
        [
            df["age"].to_numpy(dtype=np.float32),
            df["score"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    )
    y = df["label"].to_numpy(dtype=np.int64)

    X_t = torch.from_numpy(X)   # CPU
    y_t = torch.from_numpy(y)   # CPU

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    prep_time = time.perf_counter() - t0
    return loader, prep_time, len(dataset)


def make_datastar_gpu_view(csv_path: str, device: torch.device):
    """
    DataStarGPU:
        pacsv.read_csv -> Arrow Table -> tensores no device
        -> filtro + select
    """
    t0 = time.perf_counter()

    table = pacsv.read_csv(csv_path)
    ds = DataStarGPUFrame.from_arrow_table(table, device=device, numeric_only=True)

    ds2 = (
        ds.filter(colg("age") > 30)
          .select(["age", "score", "label"])
    )

    prep_time = time.perf_counter() - t0
    return ds2, prep_time, len(ds2)


# =========================
# Modelo e treino
# =========================

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_epoch_cpu(model, loader: DataLoader, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    t0 = time.perf_counter()
    for X_cpu, y_cpu in loader:
        X = X_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    return dt


def train_epoch_datastar(model, ds_gpu: DataStarGPUFrame, batch_size: int, device: torch.device) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)

    t0 = time.perf_counter()
    for batch in ds_gpu.batch(batch_size, drop_last=False):
        t = batch.to_tensor(["age", "score", "label"])
        X = torch.stack(
            [
                t["age"].float(),
                t["score"].float(),
            ],
            dim=-1,
        )
        y = t["label"].long()

        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    dt = time.perf_counter() - t0
    return dt


# =========================
# Main de teste
# =========================

def main_train_bench():
    # Generate smaller csv for quick test
    generate_csv_if_needed(CSV_PATH, n_rows=100_000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Usando device: {device}")

    batch_size = 4096

    # ---------- CPU pipeline ----------
    print("\n[train] Preparando pipeline CPU (pandas + DataLoader)...")
    cpu_loader, prep_cpu, n_rows_cpu = make_cpu_loader(CSV_PATH, batch_size)
    print(f"[train] Prep CPU: {prep_cpu:.3f} s | n_rows={n_rows_cpu}")

    model_cpu = TinyMLP().to(device)
    t_epoch_cpu = train_epoch_cpu(model_cpu, cpu_loader, device)
    print(f"[train] 1ª época (CPU DataLoader): {t_epoch_cpu:.3f} s")

    # ---------- DataStarGPU pipeline ----------
    print("\n[train] Preparando DataStarGPUFrame (Arrow -> tensores GPU)...")
    ds_gpu, prep_gpu, n_rows_gpu = make_datastar_gpu_view(CSV_PATH, device)
    print(f"[train] Prep DataStarGPU: {prep_gpu:.3f} s | n_rows={n_rows_gpu}")

    model_gpu = TinyMLP().to(device)

    t_epoch_gpu1 = train_epoch_datastar(model_gpu, ds_gpu, batch_size, device)
    t_epoch_gpu2 = train_epoch_datastar(model_gpu, ds_gpu, batch_size, device)

    print(f"[train] 1ª época (DataStarGPU): {t_epoch_gpu1:.3f} s")
    print(f"[train] 2ª época (DataStarGPU): {t_epoch_gpu2:.3f} s")

    print("\n===== RESUMO TREINO =====")
    print(f"Prep CPU loader           : {prep_cpu:.3f} s")
    print(f"Prep DataStarGPU          : {prep_gpu:.3f} s")
    print(f"1ª época CPU DataLoader   : {t_epoch_cpu:.3f} s")
    print(f"1ª época DataStarGPU      : {t_epoch_gpu1:.3f} s")
    print(f"2ª época DataStarGPU      : {t_epoch_gpu2:.3f} s")


if __name__ == "__main__":
    main_train_bench()
