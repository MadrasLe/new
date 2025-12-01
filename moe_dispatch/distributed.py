"""
Suporte para treinamento distribuído de MoE.
Implementa Expert Parallelism e Data Parallelism híbrido.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List, Dict, Any, Tuple
import os


def setup_distributed():
    """Inicializa ambiente distribuído."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        if not dist.is_initialized():
             dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Limpa ambiente distribuído."""
    if dist.is_initialized():
        dist.destroy_process_group()


class ExpertParallelGroup:
    """Gerencia grupos de processos para Expert Parallelism."""

    def __init__(
        self,
        world_size: int,
        num_experts: int,
        expert_parallel_size: Optional[int] = None
    ):
        self.world_size = world_size
        self.num_experts = num_experts

        # Determinar tamanho do grupo de experts
        if expert_parallel_size is None:
            # Dividir experts igualmente entre GPUs
            expert_parallel_size = min(world_size, num_experts)

        self.expert_parallel_size = expert_parallel_size
        self.data_parallel_size = world_size // expert_parallel_size

        assert world_size % expert_parallel_size == 0, \
            f"world_size ({world_size}) must be divisible by expert_parallel_size ({expert_parallel_size})"

        self.experts_per_rank = num_experts // expert_parallel_size

        # Criar grupos de processos
        self._create_groups()

    def _create_groups(self):
        """Cria grupos de processos para EP e DP."""
        self.expert_parallel_groups = []
        self.data_parallel_groups = []

        # Expert Parallel groups: ranks que compartilham tokens mas têm experts diferentes
        for dp_idx in range(self.data_parallel_size):
            ranks = list(range(dp_idx * self.expert_parallel_size,
                              (dp_idx + 1) * self.expert_parallel_size))
            group = dist.new_group(ranks)
            self.expert_parallel_groups.append(group)

        # Data Parallel groups: ranks que têm os mesmos experts mas dados diferentes
        for ep_idx in range(self.expert_parallel_size):
            ranks = list(range(ep_idx, self.world_size, self.expert_parallel_size))
            group = dist.new_group(ranks)
            self.data_parallel_groups.append(group)

    def get_expert_parallel_group(self, rank: int):
        """Retorna grupo EP para o rank."""
        dp_idx = rank // self.expert_parallel_size
        return self.expert_parallel_groups[dp_idx]

    def get_data_parallel_group(self, rank: int):
        """Retorna grupo DP para o rank."""
        ep_idx = rank % self.expert_parallel_size
        return self.data_parallel_groups[ep_idx]

    def get_expert_ids_for_rank(self, rank: int) -> List[int]:
        """Retorna IDs dos experts atribuídos a este rank."""
        ep_idx = rank % self.expert_parallel_size
        start = ep_idx * self.experts_per_rank
        return list(range(start, start + self.experts_per_rank))


class AllToAll(torch.autograd.Function):
    """All-to-All coletivo para Expert Parallelism."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group

        # Preparar tensores de saída
        total_output = sum(output_splits)
        output = torch.empty(
            total_output, *input.shape[1:],
            dtype=input.dtype, device=input.device
        )

        # Executar all-to-all
        input_list = list(input.split(input_splits))
        output_list = list(output.split(output_splits))

        dist.all_to_all(output_list, input_list, group=group)

        return torch.cat(output_list)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        # Backward é all-to-all com splits invertidos
        return AllToAll.apply(
            grad_output,
            ctx.output_splits,
            ctx.input_splits,
            ctx.group
        ), None, None, None


class DistributedMoELayer(nn.Module):
    """
    MoE Layer com suporte a Expert Parallelism.

    Cada rank mantém apenas um subconjunto dos experts.
    Tokens são roteados entre ranks usando all-to-all.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        dropout: float = 0.0,
        expert_parallel_group: Optional[ExpertParallelGroup] = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Configuração distribuída
        if expert_parallel_group is not None and dist.is_initialized():
            self.ep_group = expert_parallel_group
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_expert_ids = expert_parallel_group.get_expert_ids_for_rank(self.rank)
            self.num_local_experts = len(self.local_expert_ids)
            self.ep_process_group = expert_parallel_group.get_expert_parallel_group(self.rank)
            self.use_ep = True
        else:
            self.local_expert_ids = list(range(num_experts))
            self.num_local_experts = num_experts
            self.use_ep = False

        # Router (todos os ranks têm o router completo)
        self.router_weights = nn.Parameter(torch.empty(hidden_dim, num_experts))
        nn.init.kaiming_uniform_(self.router_weights, a=1.41421)

        # Apenas experts locais
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, hidden_dim)
            )
            for _ in range(self.num_local_experts)
        ])

    def _compute_routing(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computa roteamento dos tokens."""
        # Router logits
        router_logits = torch.matmul(x, self.router_weights)

        # Softmax e top-k
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Renormalizar
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        return expert_indices, expert_weights, router_probs

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward com Expert Parallelism.

        Args:
            x: [batch_size, seq_len, hidden_dim]
        """
        original_shape = x.shape
        if x.dim() == 3:
            x = x.view(-1, self.hidden_dim)

        num_tokens = x.shape[0]

        # 1. Routing
        expert_indices, expert_weights, router_probs = self._compute_routing(x)

        if self.use_ep:
            # 2. Preparar para all-to-all
            output = self._forward_with_ep(x, expert_indices, expert_weights)
        else:
            # Fallback para single GPU
            output = self._forward_local(x, expert_indices, expert_weights)

        # Reshape
        if len(original_shape) == 3:
            output = output.view(original_shape)

        # Compute aux loss
        expert_counts = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                0,
                expert_indices[:, k].long(),
                torch.ones(num_tokens, device=x.device)
            )

        fraction = expert_counts / (num_tokens * self.top_k)
        prob_mean = router_probs.mean(dim=0)
        aux_loss = (fraction * prob_mean).sum() * self.num_experts

        return {
            'output': output,
            'aux_loss': aux_loss * 0.01,
            'expert_counts': expert_counts
        }

    def _forward_local(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward local sem EP."""
        num_tokens = x.shape[0]
        output = torch.zeros_like(x)

        for k in range(self.top_k):
            for expert_id in range(self.num_experts):
                mask = expert_indices[:, k] == expert_id
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weights[mask, k:k+1] * expert_output

        return output

    def _forward_with_ep(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Forward com Expert Parallelism usando all-to-all."""
        num_tokens = x.shape[0]
        ep_size = dist.get_world_size(self.ep_process_group)

        # Contar tokens para cada rank
        tokens_per_rank = [0] * ep_size
        for k in range(self.top_k):
            for token_idx in range(num_tokens):
                expert_id = expert_indices[token_idx, k].item()
                dest_rank = expert_id // self.num_local_experts
                tokens_per_rank[dest_rank] += 1

        # Ordenar tokens por destino
        sorted_indices = []
        sorted_weights = []
        sorted_expert_local_ids = []

        for dest_rank in range(ep_size):
            for k in range(self.top_k):
                for token_idx in range(num_tokens):
                    expert_id = expert_indices[token_idx, k].item()
                    if expert_id // self.num_local_experts == dest_rank:
                        sorted_indices.append(token_idx)
                        sorted_weights.append(expert_weights[token_idx, k].item())
                        sorted_expert_local_ids.append(expert_id % self.num_local_experts)

        # Permutar input
        sorted_indices_t = torch.tensor(sorted_indices, device=x.device, dtype=torch.long)
        permuted_input = x[sorted_indices_t]

        # All-to-all para enviar tokens aos experts corretos
        recv_counts_tensor = torch.zeros(ep_size, dtype=torch.long, device=x.device)
        dist.all_to_all_single(
            recv_counts_tensor,
            torch.tensor(tokens_per_rank, device=x.device),
            group=self.ep_process_group
        )
        recv_counts = recv_counts_tensor.tolist()

        received_input = AllToAll.apply(
            permuted_input,
            tokens_per_rank,
            recv_counts,
            self.ep_process_group
        )

        # Processar com experts locais
        received_expert_ids = torch.tensor(sorted_expert_local_ids, device=x.device)

        # All-to-all para receber IDs dos experts
        received_expert_ids = AllToAll.apply(
            received_expert_ids.float().unsqueeze(-1),
            tokens_per_rank,
            recv_counts,
            self.ep_process_group
        ).squeeze(-1).long()

        # Computar outputs
        local_output = torch.zeros_like(received_input)
        for local_id in range(self.num_local_experts):
            mask = received_expert_ids == local_id
            if mask.any():
                local_output[mask] = self.experts[local_id](received_input[mask])

        # All-to-all reverso para retornar outputs
        output_permuted = AllToAll.apply(
            local_output,
            recv_counts,
            tokens_per_rank,
            self.ep_process_group
        )

        # Unpermute e combine
        output = torch.zeros_like(x)
        weights_t = torch.tensor(sorted_weights, device=x.device, dtype=x.dtype)

        for i, (token_idx, weight) in enumerate(zip(sorted_indices, sorted_weights)):
            output[token_idx] += weight * output_permuted[i]

        return output


# ============================================================================
# Trainer Distribuído
# ============================================================================

class DistributedMoETrainer:
    """Trainer para MoE com suporte distribuído."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16
    ):
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}')

        # Mover modelo para GPU
        model = model.to(self.device)

        # Wrap com DDP se distribuído
        if self.world_size > 1:
            self.model = DDP(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=True
            )
        else:
            self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # AMP
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, float]:
        """Executa um passo de treino."""
        self.model.train()

        # Mover batch para device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward com AMP
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                labels=batch.get('labels'),
                return_aux_loss=True
            )

            loss = outputs['loss']

            # Escalar para gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient step
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.optimizer.zero_grad()

        # Coletar métricas
        metrics = {
            'loss': outputs['loss'].item(),
            'lm_loss': outputs.get('lm_loss', outputs['loss']).item(),
        }

        if 'aux_loss' in outputs:
            metrics['aux_loss'] = outputs['aux_loss'].item()
        if 'z_loss' in outputs:
            metrics['z_loss'] = outputs['z_loss'].item()
        if 'tokens_dropped' in outputs:
            metrics['tokens_dropped'] = outputs['tokens_dropped'].item() if torch.is_tensor(outputs['tokens_dropped']) else outputs['tokens_dropped']

        return metrics

    def save_checkpoint(self, path: str, step: int):
        """Salva checkpoint."""
        if self.rank == 0:
            state = {
                'step': step,
                'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            if self.scheduler is not None:
                state['scheduler_state_dict'] = self.scheduler.state_dict()
            if self.scaler is not None:
                state['scaler_state_dict'] = self.scaler.state_dict()

            torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        """Carrega checkpoint. Retorna step."""
        state = torch.load(path, map_location=self.device)

        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state['model_state_dict'])

        self.optimizer.load_state_dict(state['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in state:
            self.scaler.load_state_dict(state['scaler_state_dict'])

        return state['step']
