import torch
import pytest
from model import MoELayer

# Define test parameters
NUM_EXPERTS = 8
D_MODEL = 32
D_FF = D_MODEL * 4
BATCH_SIZE = 4
SEQ_LEN = 10
TOP_K = 2

@pytest.fixture
def moe_model_and_input():
    """Pytest fixture to create a MoELayer model and a random input tensor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        pytest.skip("Test requires a CUDA-enabled GPU")

    model = MoELayer(
        num_experts=NUM_EXPERTS,
        in_features=D_MODEL,
        out_features=D_MODEL,
        hidden_features=D_FF,
        top_k=TOP_K
    ).to(device).eval() # Set to eval mode for consistent behavior

    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device)
    return model, x

def test_moe_kernel_correctness(moe_model_and_input):
    """
    Tests that the output of the Triton kernel is numerically close to the
    output of the naive, PyTorch-based implementation.
    """
    model, x = moe_model_and_input

    # --- Run Naive Implementation ---
    # We need to call the components of the forward pass manually to isolate the dispatch
    x_flat = x.view(-1, model.in_features)
    gate_logits = model.gate(x_flat)
    gate_weights, topk_indices = torch.topk(torch.softmax(gate_logits, dim=-1), model.top_k, dim=-1)

    output_original_flat = model.dispatch_and_execute_original(x_flat, topk_indices, gate_weights)
    output_original = output_original_flat.view(x.shape)

    # --- Run Triton Implementation ---
    # The standard forward pass now uses the Triton kernel
    with torch.no_grad():
        output_triton = model(x)

    # --- Compare Results ---
    assert output_triton.shape == output_original.shape, "Output shapes do not match"

    # Use torch.allclose for robust floating-point comparison
    is_close = torch.allclose(output_triton, output_original, atol=1e-5, rtol=1e-3)
    if not is_close:
        print("Triton output:", output_triton)
        print("Original output:", output_original)
        print("Difference:", torch.abs(output_triton - output_original).max())

    assert is_close, "Triton kernel output does not match the original implementation"
