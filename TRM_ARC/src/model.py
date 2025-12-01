import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyRecursiveModel(nn.Module):
    def __init__(self, num_tasks, d_model=512, n_layers=2, max_len=900, vocab_size=11):
        super().__init__()
        # ARC has 10 colors (0-9). 11th token could be padding or special, but 0 is usually black/background.
        # Paper says "embedding of shape [0, 1, D] ... added to the input".
        # Also "input x, current solution y, current latent z".
        # x and y are grids.

        self.d_model = d_model

        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Task embedding (Paper: "Each puzzle ... is given a specific embedding")
        self.task_embedding = nn.Embedding(num_tasks, d_model)

        # The paper describes "Recursive Reasoning with Tiny Networks" using a single network.
        # "z <- net(x, y, z)" (latent reasoning)
        # "y <- net(y, z)" (refine answer)
        # Actually Section 4.3 says "Single network... z <- fL(z+z+x) ... y <- fH(z+z)".
        # And then: "We considered the possibility that both networks could be replaced by a single network... It turns out that a single network is enough."
        # Input to the network is concatenation or sum?
        # "z <- f(x, y, z)" implies inputs are combined.
        # Paper implies sum in section 2.2 for HRM: "fL(zL + zH + x)".
        # For TRM single network, let's assume summing embeddings is the way (standard Transformer practice).

        # Transformer Layer
        # Paper: "2 layers... TRM with self-attention generalizes better".
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2048, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output Heads
        self.output_head = nn.Linear(d_model, vocab_size)
        self.q_head = nn.Linear(d_model, 1) # For ACT halting probability (BCE)

        self.norm = nn.LayerNorm(d_model)

    def forward_network(self, combined_embedding):
        # combined_embedding: [B, L, D]
        # Add position embedding
        x = combined_embedding + self.pos_embedding[:, :combined_embedding.size(1), :]
        x = self.transformer(x)
        return x

    def latent_recursion(self, x_emb, y_emb, z_emb, task_emb, n=6):
        # x_emb: Input grid embedding
        # y_emb: Current answer embedding (initially z_init or previous y)
        # z_emb: Latent reasoning embedding

        # "Given the input question x, current solution y, and current latent reasoning z, the model recursively improves its latent z."
        # "Then, given the current latent z and the previous solution y, the model proposes a new solution y"

        # Paper pseudocode (Figure 3):
        # def latent_recursion(x, y, z, n=6):
        #   for i in range(n):
        #     z = net(x, y, z)
        #   y = net(y, z)
        #   return y, z

        # Wait, the pseudocode in Figure 3 says:
        # z = net(x, y, z)
        # y = net(y, z)

        # Are `net` the same network? Yes, Section 4.3 "Single network".
        # How does `net` distinguish between updating z and updating y?
        # Section 4.3: "since z <- fL(x+y+z) contains x but y <- fH(y+z) does not contain x, the task ... is directly specified by the inclusion or lack of x".

        for _ in range(n):
            # Update z: input is x + y + z + task_emb
            combined = x_emb + y_emb + z_emb + task_emb
            z_emb = self.forward_network(combined)

        # Update y: input is y + z + task_emb (No x!)
        combined = y_emb + z_emb + task_emb
        y_emb = self.forward_network(combined)

        return y_emb, z_emb

    def deep_recursion(self, x, y_emb, z_emb, task_idx, n=6, T=3):
        # x: [B, L] long tensor
        # y_emb, z_emb: [B, L, D]

        B, L = x.shape
        x_emb = self.embedding(x) # [B, L, D]
        task_emb = self.task_embedding(task_idx).unsqueeze(1) # [B, 1, D] (broadcastable)

        # Recurse T-1 times without gradients
        with torch.no_grad():
            for _ in range(T - 1):
                 y_emb, z_emb = self.latent_recursion(x_emb, y_emb, z_emb, task_emb, n)

        # Recurse once with gradients
        y_emb, z_emb = self.latent_recursion(x_emb, y_emb, z_emb, task_emb, n)

        # Output predictions
        # y_emb is the refined latent "answer". We map it to logits.
        y_logits = self.output_head(self.norm(y_emb)) # [B, L, Vocab]

        # Halt probability
        # "Q-learning objective ... decides when to halt ... requires passing the zH (y) through an additional head"
        # "only learn a halting probability through a Binary-Cross-Entropy loss"
        # We pool y_emb or take the max/mean? The paper says "passing the zH through an additional head".
        # Usually for sequence tasks, we might take the mean or first token.
        # Or maybe the head projects [B, L, D] -> [B, L, 1] and we pool?
        # The paper doesn't specify pooling. Given it's a grid, maybe global average pooling?
        # Let's assume global average pooling for the halt signal.

        q_val = self.q_head(self.norm(y_emb)).mean(dim=1) # [B, 1]

        return (y_emb.detach(), z_emb.detach()), y_logits, q_val

    def forward(self, x, task_idx, y_true=None, n_sup=16, n=6, T=3):
        # Full training loop logic embedded in forward?
        # Or step-by-step?
        # The paper's pseudocode shows the loop outside.
        # "Deep Supervision ... for step in range(N_sup)..."

        # We will return the necessary components for the loop in `train.py`.
        # But to be cleaner, I can implement the initial embeddings here.
        pass

    def init_states(self, batch_size, seq_len, device):
        y_init = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        z_init = torch.zeros(batch_size, seq_len, self.d_model, device=device)
        return y_init, z_init
