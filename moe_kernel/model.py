import torch
import torch.nn as nn
import triton
import triton.language as tl

# Optimized FFN expert with SwiGLU activation
class Expert(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.activation = nn.SiLU() # SiLU (Sigmoid Linear Unit) is equivalent to swish

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)) * self.w_gate(x))

class MoELayer(nn.Module):
    def __init__(self, num_experts, in_features, out_features, hidden_features, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(in_features, out_features, hidden_features) for _ in range(num_experts)])
        self.gate = nn.Linear(in_features, num_experts, bias=False)

        # Stack weights and register them as buffers
        weights1 = torch.stack([expert.w1.weight.T for expert in self.experts])
        weights_gate = torch.stack([expert.w_gate.weight.T for expert in self.experts])
        weights2 = torch.stack([expert.w2.weight.T for expert in self.experts])

        self.register_buffer('weights1', weights1)
        self.register_buffer('weights_gate', weights_gate)
        self.register_buffer('weights2', weights2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.in_features)

        gate_logits = self.gate(x_flat)
        gate_weights, topk_indices = torch.topk(torch.softmax(gate_logits, dim=-1), self.top_k, dim=-1)

        output_flat = self.moe_dispatch_triton(x_flat, topk_indices, gate_weights)

        output = output_flat.view(batch_size, seq_len, self.out_features)
        return output

    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_FF': 32}, num_stages=3, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_FF': 64}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_FF': 32}, num_stages=4, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_FF': 32}, num_stages=4, num_warps=4),
        ],
        key=['d_model', 'd_ff'],
    )
    @triton.jit
    def fused_swiglu_kernel(
        x_ptr, expert_indices_ptr, gate_weights_ptr,
        weights1_ptr, weights_gate_ptr, weights2_ptr,
        output_ptr,
        n_tokens, d_model, d_ff,
        TOP_K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_FF: tl.constexpr
    ):
        pid = tl.program_id(0)
        # Each program computes a BLOCK_SIZE_M block of tokens
        token_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        token_mask = token_offsets < n_tokens

        final_output = tl.zeros((BLOCK_SIZE_M, d_model), dtype=tl.float32)

        for k in range(TOP_K):
            # Load expert indices and gate weights for the block
            expert_indices = tl.load(expert_indices_ptr + token_offsets * TOP_K + k, mask=token_mask)
            gate_weights = tl.load(gate_weights_ptr + token_offsets * TOP_K + k, mask=token_mask)

            # --- Tiled GEMM for W1 and W_gate ---
            acc1 = tl.zeros((BLOCK_SIZE_M, d_ff), dtype=tl.float32)
            acc_gate = tl.zeros((BLOCK_SIZE_M, d_ff), dtype=tl.float32)

            for i in range(0, d_model, BLOCK_SIZE_K):
                x_offsets = token_offsets[:, None] * d_model + (i + tl.arange(0, BLOCK_SIZE_K))[None, :]
                x_chunk = tl.load(x_ptr + x_offsets, mask=token_mask[:, None])

                # Gather weights for the block of tokens
                w1_offsets = expert_indices[:, None, None] * d_model * d_ff + (i + tl.arange(0, BLOCK_SIZE_K))[None, :, None] * d_ff + tl.arange(0, d_ff)[None, None, :]
                w1_chunk = tl.load(weights1_ptr + w1_offsets, mask=token_mask[:, None, None])

                wg_offsets = expert_indices[:, None, None] * d_model * d_ff + (i + tl.arange(0, BLOCK_SIZE_K))[None, :, None] * d_ff + tl.arange(0, d_ff)[None, None, :]
                wg_chunk = tl.load(weights_gate_ptr + wg_offsets, mask=token_mask[:, None, None])

                acc1 += tl.dot(x_chunk, w1_chunk)
                acc_gate += tl.dot(x_chunk, wg_chunk)

            # Apply SwiGLU activation
            silu_acc1 = acc1 * tl.sigmoid(acc1)
            fused_hidden = silu_acc1 * acc_gate

            # --- Tiled GEMM for W2 ---
            acc2 = tl.zeros((BLOCK_SIZE_M, d_model), dtype=tl.float32)
            for i in range(0, d_ff, BLOCK_SIZE_FF):
                hidden_offsets = tl.arange(0, BLOCK_SIZE_FF) + i
                hidden_chunk = tl.load(fused_hidden[:, hidden_offsets]) # Simplified load

                w2_offsets = expert_indices[:, None, None] * d_ff * d_model + hidden_offsets[None, :, None] * d_model + tl.arange(0, d_model)[None, None, :]
                w2_chunk = tl.load(weights2_ptr + w2_offsets, mask=token_mask[:, None, None])

                acc2 += tl.dot(hidden_chunk, w2_chunk)

            final_output += acc2 * gate_weights[:, None]

        # Store the result
        output_offsets = token_offsets[:, None] * d_model + tl.arange(0, d_model)[None, :]
        tl.store(output_ptr + output_offsets, final_output, mask=token_mask[:, None])

    def moe_dispatch_triton(self, x, topk_indices, gate_weights):
        output = torch.empty_like(x)
        n_tokens, d_model = x.shape
        d_ff = self.experts[0].w1.out_features

        grid = lambda meta: (triton.cdiv(n_tokens, meta['BLOCK_SIZE_M']),)

        self.fused_swiglu_kernel[grid](
            x, topk_indices, gate_weights,
            self.weights1, self.weights_gate, self.weights2,
            output,
            n_tokens=n_tokens, d_model=d_model, d_ff=d_ff,
            TOP_K=self.top_k,
        )
        return output

    def dispatch_and_execute_original(self, x_flat, topk_indices, gate_weights):
        output_flat = torch.zeros_like(x_flat)
        for i in range(x_flat.size(0)):
            for j in range(self.top_k):
                expert_idx = topk_indices[i, j].item()
                weight = gate_weights[i, j]
                output_flat[i] += weight * self.experts[expert_idx](x_flat[i])
        return output_flat

if __name__ == '__main__':
    # Example usage
    num_experts = 8
    d_model = 32
    d_ff = d_model * 4
    batch_size = 4
    seq_len = 10

    # Ensure tensors are on CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MoELayer(num_experts, d_model, d_model, d_ff).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    try:
        output = model(x)
        print("Model forward pass successful.")
        print("Output shape:", output.shape)
        assert output.shape == (batch_size, seq_len, d_model)
    except Exception as e:
        print(f"An error occurred during model execution: {e}")
