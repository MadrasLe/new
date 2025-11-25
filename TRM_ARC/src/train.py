import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np

# Import our modules
from TRM_ARC.src.dataset import ARCDataset, collate_fn, PADDING_VAL
from TRM_ARC.src.model import TinyRecursiveModel

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(args):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    train_dataset = ARCDataset(args.data_path, mode='train', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    print(f"Training tasks: {len(train_dataset)}")

    # Model
    # No num_tasks needed anymore!
    model = TinyRecursiveModel(d_model=args.d_model, n_layers=args.n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2000)

    # 12 classes (0-9 colors, 10 padding, 11 unused)
    # Ignore index PADDING_VAL (10) for loss calculation
    criterion_ce = nn.CrossEntropyLoss(ignore_index=PADDING_VAL)
    criterion_bce = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            # Batch is a list of dicts (from collate_fn)
            # We process them together but supports are variable length.
            # Ideally we pad support sets or use nested tensors, but for simplicity let's stack if possible or handle list.
            # But deep_recursion expects batched tensors.
            # Since collate_fn returned a list of dicts, let's construct the batch.

            # Extract queries
            query_inputs = torch.stack([item['query_input'] for item in batch]).to(device)
            query_outputs = torch.stack([item['query_output'] for item in batch]).to(device)

            # Support sets is a list of lists of dicts.
            # model.deep_recursion expects support_sets as list of lists.
            support_sets = [item['support'] for item in batch]

            B, H, W = query_inputs.shape
            x_flat = query_inputs.view(B, -1)
            y_true_flat = query_outputs.view(B, -1)

            # Initialize latent states
            y_emb, z_emb = model.init_states(B, H*W, device)

            optimizer.zero_grad()
            batch_loss = 0

            # Deep Supervision Loop
            for step in range(args.n_sup):
                (y_emb, z_emb), y_logits, q_val = model.deep_recursion(x_flat, y_emb, z_emb, support_sets, n=args.n_recur, T=args.T_recur)

                # Losses
                # y_logits: [B, L, Vocab]
                loss_ce = criterion_ce(y_logits.view(-1, 12), y_true_flat.view(-1))

                # Halt target: 1 if prediction is correct (ignoring padding), 0 otherwise
                y_pred = y_logits.argmax(dim=-1)

                # Mask out padding for accuracy check
                mask = (y_true_flat != PADDING_VAL)
                correct = (y_pred == y_true_flat) & mask
                # Check if all non-padding pixels are correct
                # Sum of correct masked pixels == Sum of mask
                is_correct = (correct.sum(dim=1) == mask.sum(dim=1)).float().unsqueeze(1) # [B, 1]

                loss_bce = criterion_bce(q_val, is_correct)

                loss = loss_ce + loss_bce
                loss.backward()

                batch_loss += loss.item()

                y_emb = y_emb.detach()
                z_emb = z_emb.detach()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if (q_val > 0).all():
                    break

            total_loss += batch_loss / (step + 1)
            pbar.set_postfix({'loss': batch_loss / (step + 1)})

        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(train_loader)}")

        # Save checkpoint
        torch.save(model.state_dict(), f"TRM_ARC/checkpoint_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="TRM_ARC/data_repo/data/arc-prize-2024/arc-agi_training_challenges.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_sup", type=int, default=4)
    parser.add_argument("--n_recur", type=int, default=2)
    parser.add_argument("--T_recur", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)

    args = parser.parse_args()
    train(args)
