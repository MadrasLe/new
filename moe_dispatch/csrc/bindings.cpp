#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "moe_kernels.cuh"
#include <cmath>

// ============================================================================
// Dispatch Functions by Dtype
// ============================================================================

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...)                    \
    switch (TYPE) {                                              \
        case at::ScalarType::Float: {                            \
            using scalar_t = float;                              \
            __VA_ARGS__();                                       \
            break;                                               \
        }                                                        \
        case at::ScalarType::Half: {                             \
            using scalar_t = __half;                             \
            __VA_ARGS__();                                       \
            break;                                               \
        }                                                        \
        case at::ScalarType::BFloat16: {                         \
            using scalar_t = __nv_bfloat16;                      \
            __VA_ARGS__();                                       \
            break;                                               \
        }                                                        \
        default:                                                 \
            TORCH_CHECK(false, "Unsupported dtype for " #NAME);  \
    }

// ============================================================================
// Router Forward
// ============================================================================

torch::Tensor moe_router_forward(
    torch::Tensor input,          // [num_tokens, hidden_dim]
    torch::Tensor router_weights  // [hidden_dim, num_experts]
) {
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(router_weights.is_cuda(), "Router weights must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(router_weights.dim() == 2, "Router weights must be 2D");

    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_tokens = input.size(0);
    int hidden_dim = input.size(1);
    int num_experts = router_weights.size(1);

    TORCH_CHECK(router_weights.size(0) == hidden_dim, "Dimension mismatch");

    // Output always in float32 for numerical stability
    auto router_logits = torch::empty(
        {num_tokens, num_experts},
        input.options().dtype(torch::kFloat32)
    );

    DISPATCH_FLOAT_TYPES(input.scalar_type(), "router_forward", [&] {
        launch_router_forward<scalar_t>(
            reinterpret_cast<const scalar_t*>(input.data_ptr()),
            reinterpret_cast<const scalar_t*>(router_weights.data_ptr()),
            router_logits.data_ptr<float>(),
            num_tokens, hidden_dim, num_experts,
            stream
        );
    });

    return router_logits;
}

// ============================================================================
// Top-K Gating Forward
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_topk_gating_forward(
    torch::Tensor router_logits,  // [num_tokens, num_experts]
    int top_k
) {
    TORCH_CHECK(router_logits.is_cuda(), "Router logits must be on CUDA");
    TORCH_CHECK(router_logits.dim() == 2, "Router logits must be 2D");
    TORCH_CHECK(router_logits.scalar_type() == torch::kFloat32, "Router logits must be float32");

    const at::cuda::CUDAGuard device_guard(router_logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_tokens = router_logits.size(0);
    int num_experts = router_logits.size(1);

    auto softmax_out = torch::empty_like(router_logits);
    auto expert_indices = torch::empty(
        {num_tokens, top_k},
        router_logits.options().dtype(torch::kInt32)
    );
    auto expert_weights = torch::empty(
        {num_tokens, top_k},
        router_logits.options()
    );

    launch_topk_softmax_forward(
        router_logits.data_ptr<float>(),
        softmax_out.data_ptr<float>(),
        expert_indices.data_ptr<int>(),
        expert_weights.data_ptr<float>(),
        num_tokens, num_experts, top_k,
        stream
    );

    return std::make_tuple(expert_indices, expert_weights, softmax_out);
}

// ============================================================================
// Top-K Gating Backward
// ============================================================================

torch::Tensor moe_topk_gating_backward(
    torch::Tensor grad_expert_weights,  // [num_tokens, top_k]
    torch::Tensor softmax_out,          // [num_tokens, num_experts]
    torch::Tensor expert_indices,       // [num_tokens, top_k]
    torch::Tensor expert_weights        // [num_tokens, top_k]
) {
    const at::cuda::CUDAGuard device_guard(grad_expert_weights.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_tokens = softmax_out.size(0);
    int num_experts = softmax_out.size(1);
    int top_k = expert_indices.size(1);

    auto grad_logits = torch::zeros_like(softmax_out);

    launch_topk_softmax_backward(
        grad_expert_weights.data_ptr<float>(),
        softmax_out.data_ptr<float>(),
        expert_indices.data_ptr<int>(),
        expert_weights.data_ptr<float>(),
        grad_logits.data_ptr<float>(),
        num_tokens, num_experts, top_k,
        stream
    );

    return grad_logits;
}

// ============================================================================
// Expert Assignment
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_compute_expert_assignment(
    torch::Tensor expert_indices,  // [num_tokens, top_k]
    int num_experts,
    int expert_capacity
) {
    const at::cuda::CUDAGuard device_guard(expert_indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_tokens = expert_indices.size(0);
    int top_k = expert_indices.size(1);
    int total_pairs = num_tokens * top_k;

    auto expert_counts = torch::zeros(
        {num_experts},
        expert_indices.options().dtype(torch::kInt32)
    );
    auto expert_offsets = torch::empty(
        {num_experts + 1},
        expert_indices.options().dtype(torch::kInt32)
    );

    // Buffers for sorting pairs (expert_id, encoded_index)
    auto unsorted_expert_ids = torch::empty(
        {total_pairs},
        expert_indices.options().dtype(torch::kInt32)
    );
    auto unsorted_encoded_indices = torch::empty(
        {total_pairs},
        expert_indices.options().dtype(torch::kInt32)
    );

    auto sorted_expert_ids = torch::empty_like(unsorted_expert_ids);
    auto sorted_encoded_indices = torch::empty_like(unsorted_encoded_indices);

    // 1. Count experts and prepare unsorted pairs (using encoded indices)
    launch_compute_expert_assignment(
        expert_indices.data_ptr<int>(),
        expert_counts.data_ptr<int>(),
        unsorted_encoded_indices.data_ptr<int>(),
        unsorted_expert_ids.data_ptr<int>(),
        num_tokens, num_experts, top_k, expert_capacity,
        stream
    );

    // 2. Compute offsets (prefix sum of counts)
    launch_compute_expert_offsets(
        expert_counts.data_ptr<int>(),
        expert_offsets.data_ptr<int>(),
        num_experts, expert_capacity,
        stream
    );

    // 3. Sort pairs by expert_id
    int num_bits = 0;
    if (num_experts > 0) {
        num_bits = static_cast<int>(std::ceil(std::log2(num_experts + 2)));
    }

    launch_radix_sort_pairs(
        unsorted_expert_ids.data_ptr<int>(),    // keys in
        unsorted_encoded_indices.data_ptr<int>(), // values in (encoded idx)
        sorted_expert_ids.data_ptr<int>(),      // keys out
        sorted_encoded_indices.data_ptr<int>(),   // values out (encoded idx)
        total_pairs,
        num_bits,
        stream
    );

    // Return sorted encoded indices, which contains (token_idx * top_k + k)
    return std::make_tuple(expert_counts, expert_offsets, sorted_encoded_indices, sorted_expert_ids);
}

// ============================================================================
// Permute Tokens
// ============================================================================

torch::Tensor moe_permute_tokens(
    torch::Tensor input,              // [num_tokens, hidden_dim]
    torch::Tensor sorted_encoded_ids, // [total_expert_tokens]
    int total_expert_tokens,
    int top_k
) {
    const at::cuda::CUDAGuard device_guard(input.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int hidden_dim = input.size(1);

    auto permuted = torch::empty(
        {total_expert_tokens, hidden_dim},
        input.options()
    );

    DISPATCH_FLOAT_TYPES(input.scalar_type(), "permute_tokens", [&] {
        launch_permute_tokens<scalar_t>(
            reinterpret_cast<const scalar_t*>(input.data_ptr()),
            reinterpret_cast<scalar_t*>(permuted.data_ptr()),
            sorted_encoded_ids.data_ptr<int>(),
            total_expert_tokens,
            hidden_dim,
            top_k,
            stream
        );
    });

    return permuted;
}

// ============================================================================
// Unpermute and Combine
// ============================================================================

torch::Tensor moe_unpermute_and_combine(
    torch::Tensor expert_output,     // [total_expert_tokens, hidden_dim]
    torch::Tensor token_positions,   // [num_tokens, top_k]
    torch::Tensor expert_weights,    // [num_tokens, top_k]
    int num_tokens
) {
    const at::cuda::CUDAGuard device_guard(expert_output.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int hidden_dim = expert_output.size(1);
    int top_k = token_positions.size(1);

    auto output = torch::zeros(
        {num_tokens, hidden_dim},
        expert_output.options()
    );

    DISPATCH_FLOAT_TYPES(expert_output.scalar_type(), "unpermute_combine", [&] {
        launch_unpermute_and_combine<scalar_t>(
            reinterpret_cast<const scalar_t*>(expert_output.data_ptr()),
            reinterpret_cast<scalar_t*>(output.data_ptr()),
            token_positions.data_ptr<int>(),
            expert_weights.data_ptr<float>(),
            num_tokens, hidden_dim, top_k,
            stream
        );
    });

    return output;
}

// ============================================================================
// Load Balance Loss
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_load_balance_loss(
    torch::Tensor router_probs,     // [num_tokens, num_experts]
    torch::Tensor expert_indices    // [num_tokens, top_k]
) {
    const at::cuda::CUDAGuard device_guard(router_probs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_tokens = router_probs.size(0);
    int num_experts = router_probs.size(1);
    int top_k = expert_indices.size(1);

    auto aux_loss = torch::zeros({1}, router_probs.options());
    auto expert_fraction = torch::zeros({num_experts}, router_probs.options());
    auto expert_prob_mean = torch::zeros({num_experts}, router_probs.options());

    launch_load_balance_loss(
        router_probs.data_ptr<float>(),
        expert_indices.data_ptr<int>(),
        aux_loss.data_ptr<float>(),
        expert_fraction.data_ptr<float>(),
        expert_prob_mean.data_ptr<float>(),
        num_tokens, num_experts, top_k,
        stream
    );

    return std::make_tuple(aux_loss, expert_fraction, expert_prob_mean);
}

// ============================================================================
// Z-Loss
// ============================================================================

torch::Tensor moe_z_loss(torch::Tensor router_logits) {
    const at::cuda::CUDAGuard device_guard(router_logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int num_tokens = router_logits.size(0);
    int num_experts = router_logits.size(1);

    auto z_loss = torch::zeros({1}, router_logits.options());

    launch_z_loss(
        router_logits.data_ptr<float>(),
        z_loss.data_ptr<float>(),
        num_tokens, num_experts,
        stream
    );

    return z_loss;
}

// ============================================================================
// Pybind11 Module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("router_forward", &moe_router_forward, "MoE Router Forward");
    m.def("topk_gating_forward", &moe_topk_gating_forward, "MoE Top-K Gating Forward");
    m.def("topk_gating_backward", &moe_topk_gating_backward, "MoE Top-K Gating Backward");
    m.def("compute_expert_assignment", &moe_compute_expert_assignment, "Compute Expert Assignment");
    m.def("permute_tokens", &moe_permute_tokens, "Permute Tokens for Experts");
    m.def("unpermute_and_combine", &moe_unpermute_and_combine, "Unpermute and Combine Expert Outputs");
    m.def("load_balance_loss", &moe_load_balance_loss, "Compute Load Balance Loss");
    m.def("z_loss", &moe_z_loss, "Compute Z-Loss for Router Stability");
}
