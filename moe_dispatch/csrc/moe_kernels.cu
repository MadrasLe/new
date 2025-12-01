#include "moe_kernels.cuh"
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================================
// Utilities
// ============================================================================

template<typename T>
__device__ __forceinline__ float to_float(T val);

template<>
__device__ __forceinline__ float to_float<float>(float val) { return val; }

template<>
__device__ __forceinline__ float to_float<__half>(__half val) { return __half2float(val); }

template<>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<typename T>
__device__ __forceinline__ T from_float(float val);

template<>
__device__ __forceinline__ float from_float<float>(float val) { return val; }

template<>
__device__ __forceinline__ __half from_float<__half>(float val) { return __float2half(val); }

template<>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

// ============================================================================
// Kernel: Router Forward
// ============================================================================

template<typename T, int BLOCK_M = 32, int BLOCK_N = 32, int BLOCK_K = 32>
__global__ void router_forward_kernel(
    const T* __restrict__ input,           // [num_tokens, hidden_dim]
    const T* __restrict__ router_weights,  // [hidden_dim, num_experts]
    float* __restrict__ router_logits,     // [num_tokens, num_experts]
    int num_tokens,
    int hidden_dim,
    int num_experts
) {
    __shared__ float smem_input[BLOCK_M][BLOCK_K + 1];
    __shared__ float smem_weights[BLOCK_K][BLOCK_N + 1];

    int bx = blockIdx.x;  // expert block
    int by = blockIdx.y;  // token block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int token_idx = by * BLOCK_M + ty;
    int expert_idx = bx * BLOCK_N + tx;

    float acc = 0.0f;

    // Loop over hidden dimension tiles
    for (int k = 0; k < hidden_dim; k += BLOCK_K) {
        // Load input tile
        int input_k = k + tx;
        if (token_idx < num_tokens && input_k < hidden_dim) {
            smem_input[ty][tx] = to_float(input[token_idx * hidden_dim + input_k]);
        } else {
            smem_input[ty][tx] = 0.0f;
        }

        // Load weights tile
        int weight_k = k + ty;
        if (weight_k < hidden_dim && expert_idx < num_experts) {
            smem_weights[ty][tx] = to_float(router_weights[weight_k * num_experts + expert_idx]);
        } else {
            smem_weights[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial product
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk++) {
            acc += smem_input[ty][kk] * smem_weights[kk][tx];
        }

        __syncthreads();
    }

    // Write result
    if (token_idx < num_tokens && expert_idx < num_experts) {
        router_logits[token_idx * num_experts + expert_idx] = acc;
    }
}

// ============================================================================
// Kernel: Router Backward
// ============================================================================

template<typename T, int BLOCK = 256>
__global__ void router_backward_input_kernel(
    const float* __restrict__ grad_logits,   // [num_tokens, num_experts]
    const T* __restrict__ router_weights,    // [hidden_dim, num_experts]
    T* __restrict__ grad_input,              // [num_tokens, hidden_dim]
    int num_tokens,
    int hidden_dim,
    int num_experts
) {
    int token_idx = blockIdx.x;
    int hidden_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_idx >= num_tokens || hidden_idx >= hidden_dim) return;

    float grad = 0.0f;

    for (int e = 0; e < num_experts; e++) {
        float g = grad_logits[token_idx * num_experts + e];
        float w = to_float(router_weights[hidden_idx * num_experts + e]);
        grad += g * w;
    }

    grad_input[token_idx * hidden_dim + hidden_idx] = from_float<T>(grad);
}

template<typename T, int BLOCK = 256>
__global__ void router_backward_weights_kernel(
    const float* __restrict__ grad_logits,   // [num_tokens, num_experts]
    const T* __restrict__ input,             // [num_tokens, hidden_dim]
    float* __restrict__ grad_weights,        // [hidden_dim, num_experts]
    int num_tokens,
    int hidden_dim,
    int num_experts
) {
    int hidden_idx = blockIdx.x;
    int expert_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (hidden_idx >= hidden_dim || expert_idx >= num_experts) return;

    float grad = 0.0f;

    // Note: This loop over all tokens can be slow for large batch sizes.
    // Optimization: Parallelize over tokens (tiling) and use atomicAdd for accumulation.
    // Currently, each thread computes one weight gradient element fully.
    for (int t = 0; t < num_tokens; t++) {
        float g = grad_logits[t * num_experts + expert_idx];
        float x = to_float(input[t * hidden_dim + hidden_idx]);
        grad += g * x;
    }

    // Since grid guarantees unique thread per weight element (hidden, expert),
    // atomicAdd is not strictly necessary unless multiple blocks map to same weight.
    // However, keeping atomicAdd makes it safe for future tiling optimizations.
    atomicAdd(&grad_weights[hidden_idx * num_experts + expert_idx], grad);
}

// ============================================================================
// Kernel: Top-K Softmax Forward (Gating)
// ============================================================================

__global__ void topk_gating_forward_kernel(
    const float* __restrict__ router_logits,  // [num_tokens, num_experts]
    float* __restrict__ softmax_out,          // [num_tokens, num_experts]
    int* __restrict__ expert_indices,         // [num_tokens, top_k]
    float* __restrict__ expert_weights,       // [num_tokens, top_k]
    int num_tokens,
    int num_experts,
    int top_k
) {
    // Shared memory layout
    extern __shared__ char shared_mem[];
    float* s_logits = (float*)shared_mem;
    float* s_topk_vals = s_logits + num_experts;
    int* s_topk_ids = (int*)(s_topk_vals + top_k);

    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= num_tokens) return;

    const float* logits = router_logits + token_idx * num_experts;

    // Load logits
    for (int i = tid; i < num_experts; i += blockDim.x) {
        s_logits[i] = logits[i];
    }
    __syncthreads();

    // Initialize top-k
    if (tid < top_k) {
        s_topk_vals[tid] = -INFINITY;
        s_topk_ids[tid] = -1;
    }
    __syncthreads();

    // CUB BlockReduce for finding max
    typedef cub::KeyValuePair<int, float> KeyValuePair;
    typedef cub::BlockReduce<KeyValuePair, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // Find top-k with partial selection sort
    for (int k = 0; k < top_k; k++) {
        KeyValuePair local_pair(-1, -INFINITY);

        // Each thread finds local max
        for (int i = tid; i < num_experts; i += blockDim.x) {
            bool selected = false;
            for (int j = 0; j < k; j++) {
                if (s_topk_ids[j] == i) {
                    selected = true;
                    break;
                }
            }
            if (!selected && s_logits[i] > local_pair.value) {
                local_pair.value = s_logits[i];
                local_pair.key = i;
            }
        }

        // Reduce to find global max
        // Use ArgMax which is (value, key) usually? No, KeyValuePair is (key, value)
        // cub::ArgMax expects Key to be index/identifier and Value to be the metric.
        // Wait, cub::BlockReduce::Reduce accepts a functor.

        // Let's use a simple comparison functor.
        KeyValuePair block_max_pair = BlockReduce(temp_storage).Reduce(local_pair, cub::ArgMax());

        if (tid == 0) {
            s_topk_vals[k] = block_max_pair.value;
            s_topk_ids[k] = block_max_pair.key;
        }
        __syncthreads();
    }

    // Softmax over top-k
    __shared__ float max_val;
    __shared__ float sum_exp;

    if (tid == 0) {
        max_val = s_topk_vals[0];
        for (int i = 1; i < top_k; i++) {
            max_val = fmaxf(max_val, s_topk_vals[i]);
        }
    }
    __syncthreads();

    if (tid < top_k) {
        s_topk_vals[tid] = expf(s_topk_vals[tid] - max_val);
    }
    __syncthreads();

    if (tid == 0) {
        sum_exp = 0.0f;
        for (int i = 0; i < top_k; i++) {
            sum_exp += s_topk_vals[i];
        }
    }
    __syncthreads();

    // Write results
    if (tid < top_k) {
        expert_indices[token_idx * top_k + tid] = s_topk_ids[tid];
        expert_weights[token_idx * top_k + tid] = s_topk_vals[tid] / sum_exp;
    }

    // Full Softmax for backward
    if (tid == 0) {
        max_val = s_logits[0];
        for (int i = 1; i < num_experts; i++) {
            max_val = fmaxf(max_val, s_logits[i]);
        }
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < num_experts; i += blockDim.x) {
        s_logits[i] = expf(s_logits[i] - max_val);
        local_sum += s_logits[i];
    }

    // Reduce sum - Using CUB BlockReduceSum is overkill but consistent,
    // or just atomicAdd since performance of this part is not the main bottleneck (shared mem).
    // Using atomicAdd to shared float is fine.

    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    atomicAdd(&block_sum, local_sum);
    __syncthreads();

    for (int i = tid; i < num_experts; i += blockDim.x) {
        softmax_out[token_idx * num_experts + i] = s_logits[i] / block_sum;
    }
}

// ============================================================================
// Kernel: Top-K Softmax Backward
// ============================================================================

__global__ void topk_gating_backward_kernel(
    const float* __restrict__ grad_expert_weights, // [num_tokens, top_k]
    const float* __restrict__ softmax_out,         // [num_tokens, num_experts]
    const int* __restrict__ expert_indices,        // [num_tokens, top_k]
    const float* __restrict__ expert_weights,      // [num_tokens, top_k]
    float* __restrict__ grad_logits,               // [num_tokens, num_experts]
    int num_tokens,
    int num_experts,
    int top_k
) {
    int token_idx = blockIdx.x;
    int expert_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_idx >= num_tokens || expert_idx >= num_experts) return;

    // Check if this expert is in top-k
    int k_idx = -1;
    for (int k = 0; k < top_k; k++) {
        if (expert_indices[token_idx * top_k + k] == expert_idx) {
            k_idx = k;
            break;
        }
    }

    float grad = 0.0f;

    if (k_idx >= 0) {
        // This expert is in top-k
        float w = expert_weights[token_idx * top_k + k_idx];
        float g = grad_expert_weights[token_idx * top_k + k_idx];

        // Softmax gradient: g * w * (1 - w) for diagonal
        // and -g * w_i * w_j for off-diagonal
        for (int k = 0; k < top_k; k++) {
            float w_k = expert_weights[token_idx * top_k + k];
            float g_k = grad_expert_weights[token_idx * top_k + k];

            if (k == k_idx) {
                grad += g_k * w * (1.0f - w);
            } else {
                grad -= g_k * w_k * w;
            }
        }
    }

    grad_logits[token_idx * num_experts + expert_idx] = grad;
}

// ============================================================================
// Kernel: Expert Assignment and Sorting
// ============================================================================

// OPTIMIZATION: Using Shared Memory Atomics for Expert Counts
// This reduces global atomic contention significantly.
__global__ void compute_expert_counts_kernel(
    const int* __restrict__ expert_indices,     // [num_tokens, top_k]
    int* __restrict__ expert_counts,            // [num_experts]
    int* __restrict__ unsorted_encoded_indices, // [num_tokens * top_k] - Stores idx = token_idx * top_k + k
    int* __restrict__ unsorted_expert_ids,      // [num_tokens * top_k]
    int num_tokens,
    int num_experts,
    int top_k,
    int expert_capacity
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_tokens * top_k;

    if (idx >= total_pairs) return;

    int token_idx = idx / top_k;
    int k = idx % top_k;
    int expert_id = expert_indices[token_idx * top_k + k];

    int position = atomicAdd(&expert_counts[expert_id], 1);

    // Store pair (expert, encoded_index) for sorting
    // We populate separate arrays for keys (expert_id) and values (encoded_idx)
    // CUB will sort these.
    if (position < expert_capacity) {
        unsorted_encoded_indices[idx] = idx; // Store encoded index
        unsorted_expert_ids[idx] = expert_id;
    } else {
        // Overflow - mark as invalid.
        unsorted_encoded_indices[idx] = -1;
        unsorted_expert_ids[idx] = num_experts + 1; // Beyond valid range to sort to end
    }
}

__global__ void compute_expert_offsets_kernel(
    const int* __restrict__ expert_counts,  // [num_experts]
    int* __restrict__ expert_offsets,       // [num_experts + 1]
    int num_experts,
    int expert_capacity
) {
    extern __shared__ int shared_counts[];

    int tid = threadIdx.x;

    // Load counts (capped by capacity)
    if (tid < num_experts) {
        shared_counts[tid] = min(expert_counts[tid], expert_capacity);
    }
    __syncthreads();

    // Simple prefix sum (for few experts)
    if (tid == 0) {
        expert_offsets[0] = 0;
        for (int i = 0; i < num_experts; i++) {
            expert_offsets[i + 1] = expert_offsets[i] + shared_counts[i];
        }
    }
}

// ============================================================================
// Kernel: Permute Tokens for Experts
// ============================================================================

template<typename T>
__global__ void permute_tokens_kernel(
    const T* __restrict__ input,              // [num_tokens, hidden_dim]
    T* __restrict__ permuted_input,           // [total_expert_tokens, hidden_dim]
    const int* __restrict__ sorted_encoded_ids, // [total_expert_tokens] - Encoded indices
    int total_tokens,
    int hidden_dim,
    int top_k
) {
    int permuted_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (permuted_idx >= total_tokens || dim_idx >= hidden_dim) return;

    int encoded_idx = sorted_encoded_ids[permuted_idx];
    if (encoded_idx < 0) {
        permuted_input[permuted_idx * hidden_dim + dim_idx] = from_float<T>(0.0f);
        return;
    }

    int token_idx = encoded_idx / top_k; // Decode token_idx

    permuted_input[permuted_idx * hidden_dim + dim_idx] = input[token_idx * hidden_dim + dim_idx];
}

// ============================================================================
// Kernel: Unpermute and Combine Outputs
// ============================================================================

template<typename T>
__global__ void unpermute_and_combine_optimized_kernel(
    const T* __restrict__ expert_output,         // [total_expert_tokens, hidden_dim]
    T* __restrict__ output,                      // [num_tokens, hidden_dim]
    const int* __restrict__ token_positions,     // [num_tokens, top_k] - position in buffer
    const float* __restrict__ expert_weights,    // [num_tokens, top_k]
    int num_tokens,
    int hidden_dim,
    int top_k
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (token_idx >= num_tokens || dim_idx >= hidden_dim) return;

    float result = 0.0f;

    for (int k = 0; k < top_k; k++) {
        int pos = token_positions[token_idx * top_k + k];
        if (pos >= 0) {
            float weight = expert_weights[token_idx * top_k + k];
            result += weight * to_float(expert_output[pos * hidden_dim + dim_idx]);
        }
    }

    output[token_idx * hidden_dim + dim_idx] = from_float<T>(result);
}

// ============================================================================
// Kernel: Load Balancing Loss
// ============================================================================

__global__ void load_balance_loss_kernel(
    const float* __restrict__ router_probs,    // [num_tokens, num_experts]
    const int* __restrict__ expert_indices,    // [num_tokens, top_k]
    float* __restrict__ aux_loss,              // [1]
    float* __restrict__ expert_fraction,       // [num_experts] - token fraction
    float* __restrict__ expert_prob_mean,      // [num_experts] - probability mean
    int num_tokens,
    int num_experts,
    int top_k
) {
    extern __shared__ float shared_mem[];
    float* s_counts = shared_mem;
    float* s_probs = shared_mem + num_experts;

    int tid = threadIdx.x;

    // Initialize
    for (int i = tid; i < num_experts; i += blockDim.x) {
        s_counts[i] = 0.0f;
        s_probs[i] = 0.0f;
    }
    __syncthreads();

    // Count tokens and sum probabilities
    for (int t = tid; t < num_tokens; t += blockDim.x) {
        // Count selected experts
        for (int k = 0; k < top_k; k++) {
            int expert_id = expert_indices[t * top_k + k];
            atomicAdd(&s_counts[expert_id], 1.0f);
        }

        // Sum probabilities
        for (int e = 0; e < num_experts; e++) {
            atomicAdd(&s_probs[e], router_probs[t * num_experts + e]);
        }
    }
    __syncthreads();

    // Normalize and compute loss
    if (tid == 0) {
        float loss = 0.0f;
        float total_assignments = (float)(num_tokens * top_k);

        for (int e = 0; e < num_experts; e++) {
            float fraction = s_counts[e] / total_assignments;
            float prob_mean = s_probs[e] / (float)num_tokens;

            expert_fraction[e] = fraction;
            expert_prob_mean[e] = prob_mean;

            loss += fraction * prob_mean;
        }

        *aux_loss = loss * (float)num_experts;
    }
}

// ============================================================================
// Kernel: Z-Loss (for stability)
// ============================================================================

__global__ void z_loss_kernel(
    const float* __restrict__ router_logits,  // [num_tokens, num_experts]
    float* __restrict__ z_loss,               // [1]
    int num_tokens,
    int num_experts
) {
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    float local_sum = 0.0f;

    for (int t = tid; t < num_tokens; t += blockDim.x) {
        float log_sum_exp = 0.0f;
        float max_val = router_logits[t * num_experts];

        // Find max
        for (int e = 1; e < num_experts; e++) {
            max_val = fmaxf(max_val, router_logits[t * num_experts + e]);
        }

        // Compute log-sum-exp
        for (int e = 0; e < num_experts; e++) {
            log_sum_exp += expf(router_logits[t * num_experts + e] - max_val);
        }
        log_sum_exp = logf(log_sum_exp) + max_val;

        local_sum += log_sum_exp * log_sum_exp;
    }

    // Reduce
    shared_mem[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *z_loss = shared_mem[0] / (float)num_tokens;
    }
}

// ============================================================================
// Launchers
// ============================================================================

template<typename T>
void launch_router_forward(
    const T* input,
    const T* router_weights,
    float* router_logits,
    int num_tokens,
    int hidden_dim,
    int num_experts,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;

    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid(
        (num_experts + BLOCK_N - 1) / BLOCK_N,
        (num_tokens + BLOCK_M - 1) / BLOCK_M
    );

    router_forward_kernel<T, BLOCK_M, BLOCK_N, 32><<<grid, block, 0, stream>>>(
        input, router_weights, router_logits,
        num_tokens, hidden_dim, num_experts
    );
}

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
) {
    // Gradient for input
    {
        int dim_blocks = (hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(num_tokens, dim_blocks);
        router_backward_input_kernel<T><<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_logits, router_weights, grad_input,
            num_tokens, hidden_dim, num_experts
        );
    }

    // Gradient for weights (accumulate)
    {
        int expert_blocks = (num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid(hidden_dim, expert_blocks);

        router_backward_weights_kernel<T><<<grid, BLOCK_SIZE, 0, stream>>>(
            grad_logits, input, (float*)grad_weights,
            num_tokens, hidden_dim, num_experts
        );
    }
}

void launch_topk_softmax_forward(
    const float* router_logits,
    float* softmax_out,
    int* expert_indices,
    float* expert_weights,
    int num_tokens,
    int num_experts,
    int top_k,
    cudaStream_t stream
) {
    // Shared size: logits buffer + topk vals + topk ids + temp storage for CUB reduce
    // CUB temp storage size depends on BLOCK_SIZE
    typedef cub::KeyValuePair<int, float> KeyValuePair;
    typedef cub::BlockReduce<KeyValuePair, BLOCK_SIZE> BlockReduce;

    int shared_size = (num_experts + top_k) * sizeof(float) + top_k * sizeof(int) + sizeof(typename BlockReduce::TempStorage);

    topk_gating_forward_kernel<<<num_tokens, BLOCK_SIZE, shared_size, stream>>>(
        router_logits, softmax_out, expert_indices, expert_weights,
        num_tokens, num_experts, top_k
    );
}

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
) {
    int expert_blocks = (num_experts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(num_tokens, expert_blocks);

    topk_gating_backward_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        grad_weights, softmax_out, expert_indices, expert_weights, grad_logits,
        num_tokens, num_experts, top_k
    );
}

void launch_compute_expert_assignment(
    const int* expert_indices,
    int* expert_counts,
    int* unsorted_encoded_indices,
    int* unsorted_expert_ids,
    int num_tokens,
    int num_experts,
    int top_k,
    int expert_capacity,
    cudaStream_t stream
) {
    int total_pairs = num_tokens * top_k;
    int block_size = 256;
    int grid_size = (total_pairs + block_size - 1) / block_size;

    compute_expert_counts_kernel<<<grid_size, block_size, 0, stream>>>(
        expert_indices,
        expert_counts,
        unsorted_encoded_indices,
        unsorted_expert_ids,
        num_tokens, num_experts, top_k, expert_capacity
    );
}

void launch_radix_sort_pairs(
    const int* keys_in,
    const int* values_in,
    int* keys_out,
    int* values_out,
    int num_items,
    int num_bits,
    cudaStream_t stream
) {
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // First call to determine size
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys_in, keys_out, values_in, values_out,
        num_items, 0, num_bits, stream
    );

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Actual sort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        keys_in, keys_out, values_in, values_out,
        num_items, 0, num_bits, stream
    );

    cudaFree(d_temp_storage);
}

void launch_compute_expert_offsets(
    const int* expert_counts,
    int* expert_offsets,
    int num_experts,
    int expert_capacity,
    cudaStream_t stream
) {
    int shared_size = num_experts * sizeof(int);
    compute_expert_offsets_kernel<<<1, 256, shared_size, stream>>>(
        expert_counts,
        expert_offsets,
        num_experts, expert_capacity
    );
}

template<typename T>
void launch_permute_tokens(
    const T* input,
    T* permuted_input,
    const int* sorted_encoded_ids,
    int total_tokens, // actually total_expert_tokens
    int hidden_dim,
    int top_k,
    cudaStream_t stream
) {
    int dim_blocks = (hidden_dim + 256 - 1) / 256;
    dim3 grid(total_tokens, dim_blocks);

    permute_tokens_kernel<T><<<grid, 256, 0, stream>>>(
        input, permuted_input, sorted_encoded_ids,
        total_tokens, hidden_dim, top_k
    );
}

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
) {
    int dim_blocks = (hidden_dim + 256 - 1) / 256;
    dim3 grid(num_tokens, dim_blocks);

    unpermute_and_combine_optimized_kernel<T><<<grid, 256, 0, stream>>>(
        expert_output, output, token_positions, expert_weights,
        num_tokens, hidden_dim, top_k
    );
}

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
) {
    int shared_size = 2 * num_experts * sizeof(float);
    load_balance_loss_kernel<<<1, 256, shared_size, stream>>>(
        router_probs, expert_indices, aux_loss, expert_fraction, expert_prob_mean,
        num_tokens, num_experts, top_k
    );
}

void launch_z_loss(
    const float* router_logits,
    float* z_loss,
    int num_tokens,
    int num_experts,
    cudaStream_t stream
) {
    int shared_size = 256 * sizeof(float);
    z_loss_kernel<<<1, 256, shared_size, stream>>>(
        router_logits, z_loss, num_tokens, num_experts
    );
}

// Explicit instantiations
template void launch_router_forward<float>(const float*, const float*, float*, int, int, int, cudaStream_t);
template void launch_router_forward<__half>(const __half*, const __half*, float*, int, int, int, cudaStream_t);
template void launch_router_forward<__nv_bfloat16>(const __nv_bfloat16*, const __nv_bfloat16*, float*, int, int, int, cudaStream_t);

template void launch_router_backward<float>(const float*, const float*, const float*, float*, float*, int, int, int, cudaStream_t);
template void launch_router_backward<__half>(const __half*, const float*, const __half*, __half*, __half*, int, int, int, cudaStream_t);
template void launch_router_backward<__nv_bfloat16>(const __nv_bfloat16*, const float*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, int, int, int, cudaStream_t);

template void launch_permute_tokens<float>(const float*, float*, const int*, int, int, int, cudaStream_t);
template void launch_permute_tokens<__half>(const __half*, __half*, const int*, int, int, int, cudaStream_t);
template void launch_permute_tokens<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, const int*, int, int, int, cudaStream_t);

template void launch_unpermute_and_combine<float>(const float*, float*, const int*, const float*, int, int, int, cudaStream_t);
template void launch_unpermute_and_combine<__half>(const __half*, __half*, const int*, const float*, int, int, int, cudaStream_t);
template void launch_unpermute_and_combine<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, const int*, const float*, int, int, int, cudaStream_t);
