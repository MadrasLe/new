import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import argparse
import numpy as np

# Import our modules
from TRM_ARC.src.dataset import ARCDataset
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # workers=0 for simplicity

    num_tasks = len(train_dataset.task_ids)
    print(f"Number of tasks: {num_tasks}")
    print(f"Training examples: {len(train_dataset)}")

    # Model
    model = TinyRecursiveModel(num_tasks=num_tasks, d_model=args.d_model, n_layers=args.n_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=2000) # Warmup

    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            x = batch['input'].to(device) # [B, 30, 30] -> flatten to [B, 900]
            y_true = batch['output'].to(device)
            task_idx = batch['task_idx'].to(device)

            B, H, W = x.shape
            x_flat = x.view(B, -1)
            y_true_flat = y_true.view(B, -1)

            # Initialize latent states
            y_emb, z_emb = model.init_states(B, H*W, device)

            optimizer.zero_grad()
            batch_loss = 0

            # Deep Supervision Loop
            for step in range(args.n_sup):
                (y_emb, z_emb), y_logits, q_val = model.deep_recursion(x_flat, y_emb, z_emb, task_idx, n=args.n_recur, T=args.T_recur)

                # Losses
                loss_ce = criterion_ce(y_logits.view(-1, 11), y_true_flat.view(-1))

                # Halt target: 1 if prediction is correct, 0 otherwise
                y_pred = y_logits.argmax(dim=-1)
                is_correct = (y_pred == y_true_flat).all(dim=1).float().unsqueeze(1) # [B, 1]
                loss_bce = criterion_bce(q_val, is_correct)

                loss = loss_ce + loss_bce
                loss.backward() # Backprop through the one recursion step

                batch_loss += loss.item()

                # Detach for next step (already done in deep_recursion but good to be explicit mentally)
                y_emb = y_emb.detach()
                z_emb = z_emb.detach()

                # Early stopping for efficient training (simulated)
                # In practice we process the whole batch in parallel, so we can't easily "break" for some samples and not others without complex masking.
                # The paper says: "ACT greatly diminishes the time spent per example ... allowing more coverage"
                # Here we just train for fixed N_sup or until batch average suggests halting?
                # For simplicity in this script, we run full N_sup but gradients accumulate?
                # Wait, "loss.backward()" is inside the loop.
                # "opt.step(), opt.zero_grad()" is inside the loop in Figure 3.
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                # Check halting condition (if all halted)
                # q_val is logit. if sigmoid(q_val) > 0.5 (q_val > 0) -> halt
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
    parser.add_argument("--batch_size", type=int, default=4) # Small batch for demo/CPU
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=64) # Tiny for demo
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_sup", type=int, default=4) # Reduced for demo
    parser.add_argument("--n_recur", type=int, default=2) # Reduced for demo
    parser.add_argument("--T_recur", type=int, default=2) # Reduced for demo
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)

    args = parser.parse_args()
    train(args)
