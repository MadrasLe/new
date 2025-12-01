"""
MoE Layer com Kernels CUDA customizados para treino eficiente.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Tuple, Dict, Any
import math

# Importar extensão CUDA
try:
    import moe_dispatch._C as moe_ops
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("Warning: CUDA extension not found. Using fallback implementation.")


# ============================================================================
# Autograd Functions
# ============================================================================

class MoERouterFunction(Function):
    """Autograd function para Router com kernels CUDA."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, router_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: [num_tokens, hidden_dim]
            router_weights: [hidden_dim, num_experts]
        Returns:
            router_logits: [num_tokens, num_experts]
        """
        if HAS_CUDA_EXT and input.is_cuda:
            router_logits = moe_ops.router_forward(input, router_weights)
        else:
            router_logits = torch.matmul(input, router_weights)

        ctx.save_for_backward(input, router_weights)
        return router_logits

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input, router_weights = ctx.saved_tensors

        # Gradiente para input: grad_output @ router_weights.T
        grad_input = torch.matmul(grad_output, router_weights.t())

        # Gradiente para weights: input.T @ grad_output
        grad_weights = torch.matmul(input.t(), grad_output)

        return grad_input, grad_weights


class TopKGatingFunction(Function):
    """Autograd function para Top-K Gating."""

    @staticmethod
    def forward(
        ctx,
        router_logits: torch.Tensor,
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            router_logits: [num_tokens, num_experts]
            top_k: número de experts por token
        Returns:
            expert_indices: [num_tokens, top_k]
            expert_weights: [num_tokens, top_k]
            softmax_out: [num_tokens, num_experts]
        """
        if HAS_CUDA_EXT and router_logits.is_cuda:
            expert_indices, expert_weights, softmax_out = moe_ops.topk_gating_forward(
                router_logits.float(), top_k
            )
        else:
            # Fallback PyTorch
            softmax_out = F.softmax(router_logits, dim=-1)
            expert_weights, expert_indices = torch.topk(softmax_out, top_k, dim=-1)
            # Renormalizar
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
            expert_indices = expert_indices.int()

        ctx.save_for_backward(softmax_out, expert_indices, expert_weights)
        ctx.top_k = top_k

        return expert_indices, expert_weights, softmax_out

    @staticmethod
    def backward(
        ctx,
        grad_indices: Optional[torch.Tensor],
        grad_weights: torch.Tensor,
        grad_softmax: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, None]:
        softmax_out, expert_indices, expert_weights = ctx.saved_tensors

        if HAS_CUDA_EXT and grad_weights.is_cuda:
            grad_logits = moe_ops.topk_gating_backward(
                grad_weights.float(),
                softmax_out,
                expert_indices,
                expert_weights
            )
        else:
            # Fallback: aproximação do gradiente
            num_tokens, num_experts = softmax_out.shape
            top_k = ctx.top_k

            grad_logits = torch.zeros_like(softmax_out)

            # Scatter gradientes para posições top-k
            for k in range(top_k):
                idx = expert_indices[:, k].long()
                w = expert_weights[:, k]
                g = grad_weights[:, k]

                # Gradiente do softmax normalizado
                grad_logits.scatter_add_(
                    1,
                    idx.unsqueeze(-1),
                    (g * w * (1 - w)).unsqueeze(-1)
                )

        return grad_logits, None


class MoEDispatchFunction(Function):
    """Autograd function para dispatch completo de tokens."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,           # [num_tokens, hidden_dim]
        expert_indices: torch.Tensor,  # [num_tokens, top_k]
        expert_weights: torch.Tensor,  # [num_tokens, top_k]
        num_experts: int,
        expert_capacity: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens para experts.

        Returns:
            permuted_input: [total_active_tokens, hidden_dim]
            expert_counts: [num_experts]
            expert_offsets: [num_experts + 1]
            token_positions: [num_tokens, top_k] - posição no buffer ou -1 se dropped
        """
        num_tokens, hidden_dim = input.shape
        top_k = expert_indices.shape[1]

        device = input.device
        dtype = input.dtype

        if HAS_CUDA_EXT and input.is_cuda:
            # Use CUDA implementation for assignment and sorting
            # sorted_encoded_indices stores `token_idx * top_k + k` sorted by expert
            expert_counts, expert_offsets, sorted_encoded_indices, sorted_expert_ids = \
                moe_ops.compute_expert_assignment(expert_indices, num_experts, expert_capacity)

            total_tokens = expert_offsets[-1].item()

            permuted_input = moe_ops.permute_tokens(
                input, sorted_encoded_indices, total_tokens, top_k
            )

            # Reconstruct token_positions for backward/combine
            # token_positions maps [token_idx, k] -> permuted_index
            # We use the fact that sorted_encoded_indices[permuted_index] = original_flat_index
            token_positions = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=device)

            # We scatter the permuted position back to the original location.
            # Only valid indices (>=0) are considered.
            valid_mask = sorted_encoded_indices >= 0
            if valid_mask.any():
                valid_indices = sorted_encoded_indices[valid_mask].long()
                # Create a range of indices [0, 1, 2, ...] corresponding to permuted positions
                permuted_positions = torch.arange(sorted_encoded_indices.size(0), device=device, dtype=torch.int32)[valid_mask]

                # Scatter: token_positions.view(-1)[original_flat_index] = permuted_position
                token_positions.view(-1).scatter_(0, valid_indices, permuted_positions)

        else:
             # Fallback
            expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
            token_positions = torch.full((num_tokens, top_k), -1, dtype=torch.int32, device=device)
            expert_indices_flat = expert_indices.view(-1).cpu().numpy()
            expert_counts_cpu = expert_counts.cpu().numpy()
            token_positions_cpu = token_positions.cpu().numpy().reshape(-1)
            for i, expert_id in enumerate(expert_indices_flat):
                if expert_counts_cpu[expert_id] < expert_capacity:
                    token_positions_cpu[i] = expert_counts_cpu[expert_id]
                    expert_counts_cpu[expert_id] += 1
            expert_counts = torch.from_numpy(expert_counts_cpu).to(device)
            token_positions = torch.from_numpy(token_positions_cpu.reshape(num_tokens, top_k)).to(device)
            expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
            expert_offsets[1:] = torch.cumsum(torch.clamp(expert_counts, max=expert_capacity), dim=0)
            total_tokens = expert_offsets[-1].item()
            permuted_input = torch.zeros(total_tokens, hidden_dim, dtype=dtype, device=device)
            for token_idx in range(num_tokens):
                for k in range(top_k):
                    pos = token_positions[token_idx, k].item()
                    if pos >= 0:
                        expert_id = expert_indices[token_idx, k].item()
                        global_pos = expert_offsets[expert_id].item() + pos
                        permuted_input[global_pos] = input[token_idx]

        ctx.save_for_backward(expert_indices, expert_weights, token_positions, expert_offsets)
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.top_k = top_k
        ctx.dtype = dtype

        return permuted_input, expert_counts, expert_offsets, token_positions

    @staticmethod
    def backward(ctx, grad_permuted, grad_counts, grad_offsets, grad_positions):
        expert_indices, expert_weights, token_positions, expert_offsets = ctx.saved_tensors
        num_tokens = ctx.num_tokens
        hidden_dim = ctx.hidden_dim
        top_k = ctx.top_k
        dtype = ctx.dtype

        device = grad_permuted.device

        # Unpermute gradientes
        grad_input = torch.zeros(num_tokens, hidden_dim, dtype=dtype, device=device)

        if HAS_CUDA_EXT and grad_permuted.is_cuda:
             # Use kernel for unpermute (scatter back)
             # Note: Unpermute combines with weights usually, but for gradient of dispatch,
             # we just move gradients back. The weight is applied in the MoECombineFunction backward.
             # So we pass weights=1.0.
             ones_weights = torch.ones_like(expert_weights)
             grad_input = moe_ops.unpermute_and_combine(
                 grad_permuted, token_positions, ones_weights, num_tokens
             )
        else:
            for token_idx in range(num_tokens):
                for k in range(top_k):
                    pos = token_positions[token_idx, k].item()
                    if pos >= 0:
                        expert_id = expert_indices[token_idx, k].item()
                        global_pos = expert_offsets[expert_id].item() + pos
                        grad_input[token_idx] += grad_permuted[global_pos]

        return grad_input, None, None, None, None


class MoECombineFunction(Function):
    """Autograd function para combinar outputs dos experts."""

    @staticmethod
    def forward(
        ctx,
        expert_output: torch.Tensor,    # [total_tokens, hidden_dim]
        expert_indices: torch.Tensor,   # [num_tokens, top_k]
        expert_weights: torch.Tensor,   # [num_tokens, top_k]
        token_positions: torch.Tensor,  # [num_tokens, top_k]
        expert_offsets: torch.Tensor,   # [num_experts + 1]
        num_tokens: int
    ) -> torch.Tensor:
        """Combina outputs dos experts ponderados."""
        hidden_dim = expert_output.shape[1]
        top_k = expert_indices.shape[1]
        device = expert_output.device
        dtype = expert_output.dtype

        output = torch.zeros(num_tokens, hidden_dim, dtype=dtype, device=device)

        if HAS_CUDA_EXT and expert_output.is_cuda:
            output = moe_ops.unpermute_and_combine(
                expert_output,
                token_positions,
                expert_weights.float(),
                num_tokens
            )
        else:
            for token_idx in range(num_tokens):
                for k in range(top_k):
                    pos = token_positions[token_idx, k].item()
                    if pos >= 0:
                        expert_id = expert_indices[token_idx, k].item()
                        global_pos = expert_offsets[expert_id].item() + pos
                        weight = expert_weights[token_idx, k]
                        output[token_idx] += weight * expert_output[global_pos]

        ctx.save_for_backward(expert_output, expert_indices, expert_weights,
                             token_positions, expert_offsets)
        ctx.num_tokens = num_tokens

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (expert_output, expert_indices, expert_weights,
         token_positions, expert_offsets) = ctx.saved_tensors
        num_tokens = ctx.num_tokens

        total_tokens, hidden_dim = expert_output.shape
        top_k = expert_indices.shape[1]
        device = grad_output.device
        dtype = grad_output.dtype

        # Gradiente para expert_output
        grad_expert_output = torch.zeros_like(expert_output)

        # Gradiente para expert_weights
        grad_expert_weights = torch.zeros_like(expert_weights)

        # We perform the backward pass for combine manually or via kernel?
        # The backward of "unpermute_and_combine" is a gather operation + dot product.
        # Kernel support for backward of unpermute_and_combine is not explicitly implemented in C++,
        # but we can implement it using basic operations or fallback.
        # Given the previous review, I should ensure correctness.
        # But wait, MoECombineFunction backward is tricky.
        # grad_output [num_tokens, hidden]
        # We need grad_expert_output [total_tokens, hidden]
        # and grad_expert_weights [num_tokens, top_k].

        # CPU Fallback logic provided for reference:
        for token_idx in range(num_tokens):
            for k in range(top_k):
                pos = token_positions[token_idx, k].item()
                if pos >= 0:
                    expert_id = expert_indices[token_idx, k].item()
                    global_pos = expert_offsets[expert_id].item() + pos
                    weight = expert_weights[token_idx, k]

                    # Gradiente para output do expert
                    # grad_expert[pos] = weight * grad_out[token]
                    grad_expert_output[global_pos] += weight * grad_output[token_idx]

                    # Gradiente para weight
                    # grad_weight = dot(grad_out[token], expert_out[pos])
                    grad_expert_weights[token_idx, k] = torch.dot(
                        grad_output[token_idx].flatten(),
                        expert_output[global_pos].flatten()
                    )

        return grad_expert_output, None, grad_expert_weights, None, None, None


# ============================================================================
# Expert FFN
# ============================================================================

class ExpertFFN(nn.Module):
    """Feed-forward network para um expert individual."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim, bias=True)
        self.w2 = nn.Linear(ffn_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish' or activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        return x


class GroupedExpertFFN(nn.Module):
    """
    Experts agrupados para processamento eficiente em batch.
    Usa grouped GEMM quando possível.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        ffn_dim: int,
        dropout: float = 0.0,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # Pesos agrupados [num_experts, in_features, out_features]
        self.w1_weight = nn.Parameter(torch.empty(num_experts, hidden_dim, ffn_dim))
        self.w1_bias = nn.Parameter(torch.empty(num_experts, ffn_dim))
        self.w2_weight = nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_dim))
        self.w2_bias = nn.Parameter(torch.empty(num_experts, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self._init_weights()

    def _init_weights(self):
        for w in [self.w1_weight, self.w2_weight]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))

        for b in [self.w1_bias, self.w2_bias]:
            fan_in = self.hidden_dim
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,              # [total_tokens, hidden_dim]
        expert_offsets: torch.Tensor, # [num_experts + 1]
        expert_counts: torch.Tensor   # [num_experts]
    ) -> torch.Tensor:
        """
        Forward pass com grouped computation.

        Args:
            x: Tokens permutados agrupados por expert
            expert_offsets: Offset inicial de cada expert no buffer
            expert_counts: Número de tokens por expert
        """
        output = torch.zeros_like(x)

        # Process each expert's tokens
        for expert_id in range(self.num_experts):
            start = expert_offsets[expert_id].item()
            count = min(expert_counts[expert_id].item(),
                       expert_offsets[expert_id + 1].item() - start)

            if count == 0:
                continue

            end = start + count
            expert_input = x[start:end]  # [count, hidden_dim]

            # FFN forward
            h = F.linear(expert_input, self.w1_weight[expert_id].t(), self.w1_bias[expert_id])
            h = self.activation(h)
            h = self.dropout(h)
            h = F.linear(h, self.w2_weight[expert_id].t(), self.w2_bias[expert_id])

            output[start:end] = h

        return output


# ============================================================================
# MoE Layer Principal
# ============================================================================

class MoELayer(nn.Module):
    """
    Mixture of Experts Layer com kernels CUDA otimizados.

    Suporta:
    - FP16/BF16 para treino misto
    - Load balancing loss
    - Z-loss para estabilidade
    - Token dropping com capacity factor
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        activation: str = 'gelu',
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        use_grouped_experts: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        self.z_loss_weight = z_loss_weight

        # Router
        self.router_weights = nn.Parameter(torch.empty(hidden_dim, num_experts))
        nn.init.kaiming_uniform_(self.router_weights, a=math.sqrt(5))

        # Experts
        if use_grouped_experts:
            self.experts = GroupedExpertFFN(
                num_experts, hidden_dim, ffn_dim, dropout, activation
            )
        else:
            self.experts = nn.ModuleList([
                ExpertFFN(hidden_dim, ffn_dim, dropout, activation)
                for _ in range(num_experts)
            ])
            self.use_grouped_experts = False
        self.use_grouped_experts = use_grouped_experts

        # Stats para logging
        self.register_buffer('expert_counts_ema', torch.zeros(num_experts))
        self.register_buffer('tokens_dropped_ema', torch.tensor(0.0))

    def compute_capacity(self, num_tokens: int) -> int:
        """Computa capacidade do expert baseado no número de tokens."""
        tokens_per_expert = num_tokens * self.top_k / self.num_experts
        return int(self.capacity_factor * tokens_per_expert)

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass do MoE.

        Args:
            x: [batch_size, seq_len, hidden_dim] ou [num_tokens, hidden_dim]
            return_aux_loss: Se deve retornar losses auxiliares

        Returns:
            Dict com:
                - output: Tensor de saída
                - aux_loss: Load balancing loss (se return_aux_loss=True)
                - z_loss: Router z-loss (se return_aux_loss=True)
                - expert_counts: Contagem por expert
        """
        original_shape = x.shape

        # Flatten para [num_tokens, hidden_dim]
        if x.dim() == 3:
            batch_size, seq_len, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)

        num_tokens = x.shape[0]
        expert_capacity = self.compute_capacity(num_tokens)

        # 1. Router forward
        router_logits = MoERouterFunction.apply(x, self.router_weights)

        # 2. Top-K gating
        expert_indices, expert_weights, softmax_probs = TopKGatingFunction.apply(
            router_logits, self.top_k
        )

        # 3. Dispatch tokens para experts
        permuted_input, expert_counts, expert_offsets, token_positions = \
            MoEDispatchFunction.apply(
                x, expert_indices, expert_weights,
                self.num_experts, expert_capacity
            )

        # 4. Expert computation
        if self.use_grouped_experts:
            expert_output = self.experts(permuted_input, expert_offsets, expert_counts)
        else:
            # Process com experts individuais
            expert_output = torch.zeros_like(permuted_input)
            for expert_id in range(self.num_experts):
                start = expert_offsets[expert_id].item()
                count = min(expert_counts[expert_id].item(),
                           expert_offsets[expert_id + 1].item() - start)
                if count > 0:
                    end = start + count
                    expert_output[start:end] = self.experts[expert_id](permuted_input[start:end])

        # 5. Combine outputs
        output = MoECombineFunction.apply(
            expert_output, expert_indices, expert_weights,
            token_positions, expert_offsets, num_tokens
        )

        # Reshape para formato original
        if len(original_shape) == 3:
            output = output.view(original_shape)

        result = {'output': output, 'expert_counts': expert_counts}

        # Compute auxiliary losses
        if return_aux_loss:
            # Load balance loss
            if HAS_CUDA_EXT and x.is_cuda:
                aux_loss, expert_fraction, expert_prob_mean = moe_ops.load_balance_loss(
                    softmax_probs, expert_indices
                )
                aux_loss = aux_loss.squeeze()
            else:
                # Fallback
                expert_fraction = expert_counts.float() / (num_tokens * self.top_k)
                expert_prob_mean = softmax_probs.mean(dim=0)
                aux_loss = (expert_fraction * expert_prob_mean).sum() * self.num_experts

            result['aux_loss'] = self.aux_loss_weight * aux_loss

            # Z-loss para estabilidade do router
            if HAS_CUDA_EXT and x.is_cuda:
                z_loss = moe_ops.z_loss(router_logits).squeeze()
            else:
                log_sum_exp = torch.logsumexp(router_logits, dim=-1)
                z_loss = (log_sum_exp ** 2).mean()

            result['z_loss'] = self.z_loss_weight * z_loss

            # Tokens dropped
            tokens_dropped = (token_positions < 0).sum().float() / (num_tokens * self.top_k)
            result['tokens_dropped'] = tokens_dropped

        # Update EMA stats
        if self.training:
            with torch.no_grad():
                self.expert_counts_ema = 0.9 * self.expert_counts_ema + 0.1 * expert_counts.float()
                if 'tokens_dropped' in result:
                    self.tokens_dropped_ema = 0.9 * self.tokens_dropped_ema + 0.1 * result['tokens_dropped']

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do MoE."""
        return {
            'expert_counts_ema': self.expert_counts_ema.cpu().numpy(),
            'tokens_dropped_ema': self.tokens_dropped_ema.item(),
            'load_balance': (self.expert_counts_ema.std() / self.expert_counts_ema.mean()).item()
        }


# ============================================================================
# MoE Transformer Block
# ============================================================================

class MoETransformerBlock(nn.Module):
    """Bloco Transformer com MoE no lugar do FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = True
    ):
        super().__init__()

        self.pre_norm = pre_norm

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

        # MoE FFN
        self.moe = MoELayer(
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            dropout=dropout,
            aux_loss_weight=aux_loss_weight,
            z_loss_weight=z_loss_weight
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] ou [batch_size, seq_len, seq_len]
            return_aux_loss: Se deve computar losses auxiliares
        """
        # Self-Attention
        if self.pre_norm:
            residual = x
            x = self.attn_norm(x)
            x, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
            x = self.attn_dropout(x)
            x = residual + x
        else:
            residual = x
            x, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
            x = self.attn_dropout(x)
            x = self.attn_norm(residual + x)

        # MoE FFN
        if self.pre_norm:
            residual = x
            x = self.ffn_norm(x)
            moe_output = self.moe(x, return_aux_loss=return_aux_loss)
            x = self.ffn_dropout(moe_output['output'])
            x = residual + x
        else:
            residual = x
            moe_output = self.moe(x, return_aux_loss=return_aux_loss)
            x = self.ffn_dropout(moe_output['output'])
            x = self.ffn_norm(residual + x)

        result = {'output': x}

        if return_aux_loss:
            result['aux_loss'] = moe_output.get('aux_loss', 0)
            result['z_loss'] = moe_output.get('z_loss', 0)
            result['expert_counts'] = moe_output.get('expert_counts')
            result['tokens_dropped'] = moe_output.get('tokens_dropped', 0)

        return result


# ============================================================================
# Modelo MoE Completo
# ============================================================================

class MoETransformer(nn.Module):
    """Transformer completo com camadas MoE."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        moe_frequency: int = 2,  # MoE a cada N camadas
        aux_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        tie_embeddings: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.moe_frequency = moe_frequency

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)

        # Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_moe = (i + 1) % moe_frequency == 0

            if use_moe:
                layer = MoETransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout,
                    capacity_factor=capacity_factor,
                    aux_loss_weight=aux_loss_weight,
                    z_loss_weight=z_loss_weight
                )
            else:
                # Camada densa regular
                layer = DenseTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout
                )

            self.layers.append(layer)

        # Output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] para language modeling
            return_aux_loss: Se deve computar losses auxiliares
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        # Converter mask se necessário
        if attention_mask is not None:
            # Inverter: 1 = attend, 0 = mask -> True = mask, False = attend
            # Assuming standard padding mask where 0 is padding
            # MultiheadAttention expects True for ignored elements if key_padding_mask is used
            attention_mask = attention_mask == 0

        # Forward through layers
        total_aux_loss = 0.0
        total_z_loss = 0.0
        all_expert_counts = []
        total_tokens_dropped = 0.0
        num_moe_layers = 0

        for layer in self.layers:
            if isinstance(layer, MoETransformerBlock):
                layer_output = layer(x, attention_mask, return_aux_loss=return_aux_loss)
                x = layer_output['output']

                if return_aux_loss:
                    total_aux_loss += layer_output.get('aux_loss', 0)
                    total_z_loss += layer_output.get('z_loss', 0)
                    if layer_output.get('expert_counts') is not None:
                        all_expert_counts.append(layer_output['expert_counts'])
                    total_tokens_dropped += layer_output.get('tokens_dropped', 0)
                    num_moe_layers += 1
            else:
                x = layer(x, attention_mask)

        # Final norm and output
        x = self.final_norm(x)
        logits = self.lm_head(x)

        result = {'logits': logits}

        # Compute LM loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            result['lm_loss'] = lm_loss

            # Total loss com auxiliares
            if return_aux_loss and num_moe_layers > 0:
                result['aux_loss'] = total_aux_loss / num_moe_layers
                result['z_loss'] = total_z_loss / num_moe_layers
                result['loss'] = lm_loss + result['aux_loss'] + result['z_loss']
                result['tokens_dropped'] = total_tokens_dropped / num_moe_layers
            else:
                result['loss'] = lm_loss

        if all_expert_counts:
            result['expert_counts'] = torch.stack(all_expert_counts)

        return result


class DenseTransformerBlock(nn.Module):
    """Bloco Transformer denso padrão (sem MoE)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-Attention
        residual = x
        x = self.attn_norm(x)
        x, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = self.attn_dropout(x)
        x = residual + x

        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x = residual + x

        return x
