import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt

from TRM_ARC.src.dataset import ARCDataset
from TRM_ARC.src.model import TinyRecursiveModel

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset (No augmentation for eval)
    dataset = ARCDataset(args.data_path, mode='eval', augment=False)
    # Just take a few examples
    indices = range(min(5, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1)

    # Model
    # Need to instantiate with same num_tasks.
    # We cheat a bit and reload the dataset to get ID mapping or save it.
    # For this script, we assume the same data path implies same task IDs order.
    num_tasks = len(dataset.task_ids)
    model = TinyRecursiveModel(num_tasks=num_tasks, d_model=args.d_model, n_layers=args.n_layers).to(device)

    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Loaded checkpoint.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}. Running with random weights.")

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch['input'].to(device)
            y_true = batch['output'].to(device)
            task_idx = batch['task_idx'].to(device)

            B, H, W = x.shape
            x_flat = x.view(B, -1)

            y_emb, z_emb = model.init_states(B, H*W, device)

            # Run for N_sup steps (test time: usually full steps)
            final_pred = None

            for step in range(args.n_sup):
                (y_emb, z_emb), y_logits, q_val = model.deep_recursion(x_flat, y_emb, z_emb, task_idx, n=args.n_recur, T=args.T_recur)
                pred = y_logits.argmax(dim=-1).view(H, W).cpu().numpy()
                final_pred = pred

                # Check halting
                if q_val.item() > 0:
                    print(f"Example {i}: Halted at step {step}")
                    break

            # Visualize
            input_grid = x.view(H, W).cpu().numpy()
            target_grid = y_true.view(H, W).cpu().numpy()

            print(f"Example {i} Accuracy: {(final_pred == target_grid).mean():.2f}")

            # Simple Text Visualization
            # print("Input:\n", input_grid)
            # print("Target:\n", target_grid)
            # print("Pred:\n", final_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="TRM_ARC/data_repo/data/arc-prize-2024/arc-agi_training_challenges.json")
    parser.add_argument("--checkpoint", type=str, default="TRM_ARC/checkpoint_epoch_1.pt")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_sup", type=int, default=16)
    parser.add_argument("--n_recur", type=int, default=2)
    parser.add_argument("--T_recur", type=int, default=2)

    args = parser.parse_args()
    evaluate(args)
