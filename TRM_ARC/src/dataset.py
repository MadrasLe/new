import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random

def pad_grid(grid, max_h=30, max_w=30):
    grid = np.array(grid)
    h, w = grid.shape
    pad_h = max_h - h
    pad_w = max_w - w
    if pad_h < 0 or pad_w < 0:
        # Crop if too big (should not happen for ARC standard, but safe to handle)
        return grid[:max_h, :max_w]
    return np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

class ARCDataset(Dataset):
    def __init__(self, json_path, mode='train', augment=True):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.mode = mode
        self.augment = augment
        self.examples = []

        # Flatten structure: key -> train/test -> examples
        for task_id, task_data in self.data.items():
            # In training mode, we use the 'train' pairs
            # In evaluation, we might use 'test' pairs, but typically for training we just iterate over all 'train' examples
            # The paper says: "Each puzzle ... is given a specific embedding"
            # So we need to index tasks

            # For this simple implementation, let's just grab all training pairs
            # Paper mentions ARC-AGI-1 contains 800 tasks.

            train_pairs = task_data.get('train', [])
            for pair in train_pairs:
                self.examples.append({
                    'task_id': task_id, # We might need to map this to an integer index if we use task embeddings
                    'input': pair['input'],
                    'output': pair['output']
                })

        # Create task ID mapping
        self.task_ids = sorted(list(set(e['task_id'] for e in self.examples)))
        self.task_to_idx = {t_id: i for i, t_id in enumerate(self.task_ids)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        task_idx = self.task_to_idx[ex['task_id']]

        input_grid = np.array(ex['input'])
        output_grid = np.array(ex['output'])

        # Augmentations
        if self.augment:
            # Color permutation
            if random.random() < 0.5:
                perm = list(range(10))
                random.shuffle(perm)
                map_perm = np.array(perm)
                input_grid = map_perm[input_grid]
                output_grid = map_perm[output_grid]

            # Dihedral (Rotations/Flips)
            k = random.randint(0, 3)
            input_grid = np.rot90(input_grid, k)
            output_grid = np.rot90(output_grid, k)

            if random.random() < 0.5:
                input_grid = np.flipud(input_grid)
                output_grid = np.flipud(output_grid)

            # Translation? Paper mentions translation.
            # ARC grids are often object based, simple translation inside the 30x30 canvas might break things
            # if the grid size is dynamic.
            # Since we pad to 30x30, we can translate *within* the 30x30 window if the grid is smaller.
            # For simplicity, I will stick to rigid transforms for now.

        # Pad to 30x30
        input_padded = pad_grid(input_grid)
        output_padded = pad_grid(output_grid)

        return {
            'input': torch.tensor(input_padded, dtype=torch.long),
            'output': torch.tensor(output_padded, dtype=torch.long),
            'task_idx': torch.tensor(task_idx, dtype=torch.long)
        }
