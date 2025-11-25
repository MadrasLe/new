import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING_VAL = 10

class TaskEncoder(nn.Module):
    def __init__(self, d_model, embedding_layer, pos_embedding, transformer_encoder):
        super().__init__()
        self.d_model = d_model
        self.embedding = embedding_layer
        self.pos_embedding = pos_embedding
        self.encoder = transformer_encoder

    def forward(self, support_sets):
        # support_sets: list of length B, where each element is a list of tensors (pairs)
        # We need to process each task's support set to get a single task_emb [D]

        batch_task_embs = []

        for task_supports in support_sets:
            # task_supports: list of dicts {'input': [L], 'output': [L]}
            if not task_supports:
                # No support? Should not happen in training, but maybe in extreme cases.
                batch_task_embs.append(torch.zeros(self.d_model, device=self.pos_embedding.device))
                continue

            # We process each pair.
            # Strategy: Embed input + output. Concatenate or add?
            # Let's add them to represent the "transformation" in the same space.
            # Then run through encoder. Then Mean Pool.

            pair_embs = []
            for pair in task_supports:
                inp = pair['input'].to(self.pos_embedding.device)
                out = pair['output'].to(self.pos_embedding.device)

                # Flatten grids: [H, W] -> [H*W] = [L]
                inp = inp.view(-1)
                out = out.view(-1)

                # Embed
                inp_emb = self.embedding(inp) # [L, D]
                out_emb = self.embedding(out) # [L, D]

                # Combine: Simple addition to represent "Input -> Output" mapping in latent space
                # Add position embedding
                combined = inp_emb + out_emb
                # combined is [L, D]. pos_embedding is [1, MaxLen, D].
                combined = combined.unsqueeze(0) + self.pos_embedding[:, :inp.size(0), :] # [1, L, D]

                # Mask
                # 10 is padding. If both are padding, we mask.
                # Actually, we should mask if ANY is padding or just if the position is invalid?
                # The grid is padded. So positions > H*W are padding.
                # Let's create mask based on PADDING_VAL
                is_padding = (inp == PADDING_VAL) # [L]

                # Encoder
                # Transformer expects src_key_padding_mask as [B, L]
                # Here B=1 for this pair
                enc_out = self.encoder(combined, src_key_padding_mask=is_padding.unsqueeze(0))

                # Global Mean Pooling of non-padding tokens
                # mask: [1, L]
                mask = ~is_padding.unsqueeze(0) # 1 valid, 0 padding
                mask_float = mask.float().unsqueeze(-1)

                pooled = (enc_out * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
                pair_embs.append(pooled)

            # Average over all pairs in the support set
            task_emb = torch.stack(pair_embs).mean(dim=0) # [1, D]
            batch_task_embs.append(task_emb)

        return torch.cat(batch_task_embs, dim=0) # [B, D]

class TinyRecursiveModel(nn.Module):
    def __init__(self, num_tasks=None, d_model=512, n_layers=2, max_len=900, vocab_size=12):
        super().__init__()
        # vocab_size 12: 0-9 colors, 10 padding, 11 (optional extra)
        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PADDING_VAL)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Task Encoder (In-Context Learning)
        # We reuse the main embedding and transformer to save parameters (Tiny Network philosophy)
        self.task_encoder = TaskEncoder(d_model, self.embedding, self.pos_embedding, self.transformer)

        # Output Heads
        self.output_head = nn.Linear(d_model, vocab_size)
        self.q_head = nn.Linear(d_model, 1) # For ACT halting probability (BCE)

        self.norm = nn.LayerNorm(d_model)

    def forward_network(self, combined_embedding, src_key_padding_mask=None):
        # combined_embedding: [B, L, D]
        # Add position embedding
        x = combined_embedding + self.pos_embedding[:, :combined_embedding.size(1), :]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def latent_recursion(self, x_emb, y_emb, z_emb, task_emb, mask, n=6):
        # mask: [B, L] boolean (True = padding)

        # Iterate on z
        for _ in range(n):
            # Input is x + y + z + task_emb
            combined = x_emb + y_emb + z_emb + task_emb
            z_emb = self.forward_network(combined, src_key_padding_mask=mask)

        # Update y
        # Input is y + z + task_emb (No x!)
        combined = y_emb + z_emb + task_emb
        y_emb = self.forward_network(combined, src_key_padding_mask=mask)

        return y_emb, z_emb

    def deep_recursion(self, x, y_emb, z_emb, support_sets, n=6, T=3):
        # x: [B, L] long tensor
        # y_emb, z_emb: [B, L, D]
        # support_sets: list of pairs

        B, L = x.shape
        x_emb = self.embedding(x) # [B, L, D]

        # Generate Task Embedding from Support Set
        # [B, D] -> [B, 1, D]
        task_emb = self.task_encoder(support_sets).unsqueeze(1)

        # Create Padding Mask
        # PADDING_VAL is 10.
        # nn.Transformer src_key_padding_mask: True for padded elements.
        padding_mask = (x == PADDING_VAL) # [B, L]

        # Recurse T-1 times without gradients
        with torch.no_grad():
            for _ in range(T - 1):
                 y_emb, z_emb = self.latent_recursion(x_emb, y_emb, z_emb, task_emb, padding_mask, n)

        # Recurse once with gradients
        y_emb, z_emb = self.latent_recursion(x_emb, y_emb, z_emb, task_emb, padding_mask, n)

        # Output predictions
        y_logits = self.output_head(self.norm(y_emb)) # [B, L, Vocab]

        # Halt probability
        # Pool (mean over non-padded tokens)
        mask_float = (~padding_mask).float().unsqueeze(-1)
        pooled_y = (y_emb * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-9)
        q_val = self.q_head(self.norm(pooled_y)) # [B, 1]

        return (y_emb.detach(), z_emb.detach()), y_logits, q_val

    def init_states(self, batch_size, seq_len, device):
        y_init = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        z_init = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        return y_init, z_init
