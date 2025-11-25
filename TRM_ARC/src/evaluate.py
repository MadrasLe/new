import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import matplotlib.pyplot as plt

from TRM_ARC.src.dataset import ARCDataset, collate_fn, PADDING_VAL
from TRM_ARC.src.model import TinyRecursiveModel

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset (No augmentation for eval)
    dataset = ARCDataset(args.data_path, mode='eval', augment=False)
    # Just take a few examples
    indices = range(min(5, len(dataset)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, collate_fn=collate_fn)

    # Model
    model = TinyRecursiveModel(d_model=args.d_model, n_layers=args.n_layers).to(device)

    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print("Loaded checkpoint.")
    except Exception as e:
        print(f"Could not load checkpoint: {e}. Running with random weights.")

    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            # Batch size 1, but batch is list of dicts
            item = batch[0]
            x = item['query_input'].unsqueeze(0).to(device)
            y_true = item['query_output'].unsqueeze(0).to(device)
            support_sets = [item['support']]

            B, H, W = x.shape
            x_flat = x.view(B, -1)

            y_emb, z_emb = model.init_states(B, H*W, device)

            # Run for N_sup steps (test time: usually full steps)
            final_pred = None

            for step in range(args.n_sup):
                (y_emb, z_emb), y_logits, q_val = model.deep_recursion(x_flat, y_emb, z_emb, support_sets, n=args.n_recur, T=args.T_recur)
                pred = y_logits.argmax(dim=-1).view(H, W).cpu().numpy()
                final_pred = pred

                # Check halting
                if q_val.item() > 0:
                    print(f"Example {i}: Halted at step {step}")
                    break

            # Visualize
            target_grid = y_true.view(H, W).cpu().numpy()

            # Calculate accuracy on non-padded region
            mask = (target_grid != PADDING_VAL)
            if mask.sum() > 0:
                acc = (final_pred[mask] == target_grid[mask]).mean()
            else:
                acc = 0.0

            print(f"Example {i} Accuracy (Valid Pixels): {acc:.2f}")

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
