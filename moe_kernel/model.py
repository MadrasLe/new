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
    @triton.jit
    def fused_swiglu_kernel(
        x_ptr, expert_indices_ptr, gate_weights_ptr,
        weights1_ptr, weights_gate_ptr, weights2_ptr,
        output_ptr,
        n_tokens,
        stride_x_n, stride_x_d,
        stride_ei_n, stride_ei_k,
        stride_gw_n, stride_gw_k,
        stride_w1_e, stride_w1_d, stride_w1_f,
        stride_wg_e, stride_wg_d, stride_wg_f,
        stride_w2_e, stride_w2_f, stride_w2_d,
        stride_o_n, stride_o_d,
        D_MODEL: tl.constexpr, D_FF: tl.constexpr, TOP_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr
    ):
        pid = tl.program_id(0)

        for i in range(BLOCK_SIZE_N):
            # Calculate the token index directly, avoiding tensor indexing
            current_token_idx = pid * BLOCK_SIZE_N + i
            if current_token_idx < n_tokens:
                # Load data for the current token
                x_offsets = current_token_idx * stride_x_n + tl.arange(0, D_MODEL) * stride_x_d
                x_token = tl.load(x_ptr + x_offsets) # Shape: (D_MODEL,)

                output_token_accumulator = tl.zeros((D_MODEL,), dtype=tl.float32)

                for k in range(TOP_K):
                    expert_idx = tl.load(expert_indices_ptr + current_token_idx * stride_ei_n + k)
                    gate_weight = tl.load(gate_weights_ptr + current_token_idx * stride_gw_n + k)

                    # Load expert weights (2D matrices)
                    w1_offsets = expert_idx * stride_w1_e + tl.arange(0, D_MODEL)[:, None] * stride_w1_d + tl.arange(0, D_FF)[None, :] * stride_w1_f
                    w1 = tl.load(weights1_ptr + w1_offsets)

                    wg_offsets = expert_idx * stride_wg_e + tl.arange(0, D_MODEL)[:, None] * stride_wg_d + tl.arange(0, D_FF)[None, :] * stride_wg_f
                    w_gate = tl.load(weights_gate_ptr + wg_offsets)

                    w2_offsets = expert_idx * stride_w2_e + tl.arange(0, D_FF)[:, None] * stride_w2_f + tl.arange(0, D_MODEL)[None, :] * stride_w2_d
                    w2 = tl.load(weights2_ptr + w2_offsets)

                    # FFN computation using 2D dot products
                    hidden1 = tl.dot(x_token[None, :], w1)
                    gate_val = tl.dot(x_token[None, :], w_gate)

                    # Correct SwiGLU logic
                    activated_hidden = tl.sigmoid(hidden1) * hidden1 # SiLU

                    fused_result = activated_hidden * gate_val
                    expert_output = tl.dot(fused_result, w2)

                    output_token_accumulator += expert_output[0] * gate_weight

                # Store the final result for the token
                output_offsets = current_token_idx * stride_o_n + tl.arange(0, D_MODEL) * stride_o_d
                tl.store(output_ptr + output_offsets, output_token_accumulator)

    def moe_dispatch_triton(self, x, topk_indices, gate_weights):
        output = torch.empty_like(x)
        n_tokens, d_model = x.shape
        d_ff = self.experts[0].w1.out_features

        grid = lambda meta: (triton.cdiv(n_tokens, meta['BLOCK_SIZE_N']),)

        self.fused_swiglu_kernel[grid](
            x, topk_indices, gate_weights,
            self.weights1, self.weights_gate, self.weights2,
            output,
            n_tokens=n_tokens,
            stride_x_n=x.stride(0), stride_x_d=x.stride(1),
            stride_ei_n=topk_indices.stride(0), stride_ei_k=topk_indices.stride(1),
            stride_gw_n=gate_weights.stride(0), stride_gw_k=gate_weights.stride(1),
            stride_w1_e=self.weights1.stride(0), stride_w1_d=self.weights1.stride(1), stride_w1_f=self.weights1.stride(2),
            stride_wg_e=self.weights_gate.stride(0), stride_wg_d=self.weights_gate.stride(1), stride_wg_f=self.weights_gate.stride(2),
            stride_w2_e=self.weights2.stride(0), stride_w2_f=self.weights2.stride(1), stride_w2_d=self.weights2.stride(2),
            stride_o_n=output.stride(0), stride_o_d=output.stride(1),
            D_MODEL=d_model, D_FF=d_ff, TOP_K=self.top_k,
            BLOCK_SIZE_N=64, # Or another suitable block size
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
