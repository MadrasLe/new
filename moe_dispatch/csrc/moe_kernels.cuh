#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Constants
#define WARP_SIZE 32
#define MAX_EXPERTS 128
#define BLOCK_SIZE 256
#define TILE_DIM 32

// Configuration structure
struct MoEConfig {
    int num_tokens;
    int hidden_dim;
    int num_experts;
    int top_k;
    int expert_capacity;
    int ffn_hidden_dim;
};

// Kernel declarations
template<typename T>
void launch_router_forward(
    const T* input,
    const T* router_weights,
    float* router_logits,
    int num_tokens,
    int hidden_dim,
    int num_experts,
    cudaStream_t stream
);

template<typename T>
void launch_router_backward(
    const T* input,
    const float* grad_logits,
    const T* router_weights,
    T* grad_input,
    T* grad_weights,
    int num_tokens,
    int hidden_dim,
    int num_experts,
    cudaStream_t stream
);

void launch_topk_softmax_forward(
    const float* router_logits,
    float* softmax_out,
    int* expert_indices,
    float* expert_weights,
    int num_tokens,
    int num_experts,
    int top_k,
    cudaStream_t stream
);

void launch_topk_softmax_backward(
    const float* grad_weights,
    const float* softmax_out,
    const int* expert_indices,
    const float* expert_weights,
    float* grad_logits,
    int num_tokens,
    int num_experts,
    int top_k,
    cudaStream_t stream
);

template<typename T>
void launch_permute_tokens(
    const T* input,
    T* permuted_input,
    const int* sorted_indices,
    int num_tokens,
    int hidden_dim,
    int total_expert_tokens,
    cudaStream_t stream
);

template<typename T>
void launch_unpermute_and_combine(
    const T* expert_output,
    T* output,
    const int* token_positions,
    const float* expert_weights,
    int num_tokens,
    int hidden_dim,
    int top_k,
    cudaStream_t stream
);

void launch_compute_expert_assignment(
    const int* expert_indices,
    int* expert_counts,
    int* unsorted_token_indices,
    int* unsorted_expert_ids,
    int num_tokens,
    int num_experts,
    int top_k,
    int expert_capacity,
    cudaStream_t stream
);

void launch_radix_sort_pairs(
    const int* keys_in,
    const int* values_in,
    int* keys_out,
    int* values_out,
    int num_items,
    int num_bits,
    cudaStream_t stream
);

void launch_compute_expert_offsets(
    const int* expert_counts,
    int* expert_offsets,
    int num_experts,
    int expert_capacity,
    cudaStream_t stream
);

void launch_load_balance_loss(
    const float* router_probs,
    const int* expert_indices,
    float* aux_loss,
    float* expert_fraction,
    float* expert_prob_mean,
    int num_tokens,
    int num_experts,
    int top_k,
    cudaStream_t stream
);

void launch_z_loss(
    const float* router_logits,
    float* z_loss,
    int num_tokens,
    int num_experts,
    cudaStream_t stream
);
