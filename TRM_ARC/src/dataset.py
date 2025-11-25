import json
import torch
import numpy as np
from torch.utils.data import Dataset
import random

# 0-9 are colors. 10 is padding.
PADDING_VAL = 10

def pad_grid(grid, max_h=30, max_w=30, pad_val=PADDING_VAL):
    grid = np.array(grid)
    h, w = grid.shape
    pad_h = max_h - h
    pad_w = max_w - w
    if pad_h < 0 or pad_w < 0:
        return grid[:max_h, :max_w]
    return np.pad(grid, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=pad_val)

class ARCDataset(Dataset):
    def __init__(self, json_path, mode='train', augment=True):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.mode = mode
        self.augment = augment
        self.tasks = []

        # Structure: Each item is a Task, containing support set (train pairs) and query set (one train pair or test pair)
        # For training, we sample one pair from 'train' as query, and use the rest as support.
        # This is standard "Episodic Training" for Few-Shot.

        for task_id, task_data in self.data.items():
            train_pairs = task_data.get('train', [])
            test_pairs = task_data.get('test', [])

            if mode == 'train':
                # In training, we can generate multiple examples per task by rotating the query
                # If a task has N train pairs, we can create N examples where one is query and N-1 are support.
                # If N < 2, we can't really do few-shot, but ARC usually has 2-5.
                if len(train_pairs) > 1:
                    for i in range(len(train_pairs)):
                        self.tasks.append({
                            'task_id': task_id,
                            'support': [p for j, p in enumerate(train_pairs) if j != i],
                            'query': train_pairs[i]
                        })
                else:
                    # Fallback for single example tasks (rare but possible) - duplicate as support
                     self.tasks.append({
                            'task_id': task_id,
                            'support': train_pairs,
                            'query': train_pairs[0]
                        })
            else:
                # Evaluation: use all train pairs as support, and iterate over test pairs as queries
                for test_pair in test_pairs:
                    self.tasks.append({
                        'task_id': task_id,
                        'support': train_pairs,
                        'query': test_pair
                    })

    def __len__(self):
        return len(self.tasks)

    def apply_augment(self, grid, perm, k, flip):
        # Color perm
        grid = perm[grid]
        # Rot
        grid = np.rot90(grid, k)
        # Flip
        if flip:
            grid = np.flipud(grid)
        return grid

    def __getitem__(self, idx):
        task = self.tasks[idx]

        support_raw = task['support']
        query_raw = task['query']

        # Prepare augmentations (consistent across support and query for valid task logic)
        perm = np.arange(11) # 0-9 colors + 10 padding (identity for padding)
        # Padding should not be permuted usually, but if we map 0-9 -> 0-9, 10 stays 10.
        if self.augment:
            color_perm = np.random.permutation(10)
            perm[:10] = color_perm

            k = random.randint(0, 3)
            flip = random.random() < 0.5
        else:
            perm = np.arange(11)
            k = 0
            flip = False

        support_tensors = []
        for pair in support_raw:
            inp = self.apply_augment(np.array(pair['input']), perm, k, flip)
            out = self.apply_augment(np.array(pair['output']), perm, k, flip)

            support_tensors.append({
                'input': torch.tensor(pad_grid(inp), dtype=torch.long),
                'output': torch.tensor(pad_grid(out), dtype=torch.long)
            })

        q_inp = self.apply_augment(np.array(query_raw['input']), perm, k, flip)
        # Handle test query output (might not exist or we ignore it for inference, but exist for training)
        if 'output' in query_raw:
            q_out = self.apply_augment(np.array(query_raw['output']), perm, k, flip)
            q_out_t = torch.tensor(pad_grid(q_out), dtype=torch.long)
        else:
            q_out_t = torch.tensor(pad_grid(np.zeros((1,1))), dtype=torch.long) # Dummy

        return {
            'task_id': task['task_id'],
            'support': support_tensors,
            'query_input': torch.tensor(pad_grid(q_inp), dtype=torch.long),
            'query_output': q_out_t
        }

def collate_fn(batch):
    # Batch is a list of dicts.
    # Support sets have different lengths.
    # We can pad support sets to max length in batch, or just return list.
    # Returning list is easier for model to handle (process per task).

    return batch
