#!/bin/bash
set -e

echo "Starting TRM Setup and Demo..."

# Check if data exists
DATA_DIR="TRM_ARC/data_repo"
if [ ! -d "$DATA_DIR" ]; then
    echo "Downloading ARC-AGI Data..."
    git clone https://github.com/TrelisResearch/arc-agi-2025.git "$DATA_DIR"
else
    echo "Data directory found."
fi

# Install dependencies if needed (assuming pip is available)
# In this environment I assume they are installed, but for plug and play:
# pip install torch numpy tqdm matplotlib

echo "Starting Training..."
# Run a quick training
python -m TRM_ARC.src.train --epochs 1 --batch_size 2 --d_model 64 --n_sup 2 --n_recur 2 --T_recur 2 --data_path "$DATA_DIR/data/arc-prize-2024/arc-agi_training_challenges.json"

echo "Training complete. Running evaluation..."
python -m TRM_ARC.src.evaluate --d_model 64 --n_sup 2 --n_recur 2 --T_recur 2 --data_path "$DATA_DIR/data/arc-prize-2024/arc-agi_training_challenges.json" --checkpoint TRM_ARC/checkpoint_epoch_1.pt
