import torch
from model import MoELayer

# Define model parameters
NUM_EXPERTS = 8
D_MODEL = 512
D_FF = D_MODEL * 4
BATCH_SIZE = 4
SEQ_LEN = 1024
TOP_K = 2

def profile_moe_layer():
    """
    Profiles the MoELayer using torch.profiler to measure performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Profiling requires a CUDA-enabled GPU. Aborting.")
        return

    print(f"Using device: {device}")
    print(f"Profiling with: BATCH_SIZE={BATCH_SIZE}, SEQ_LEN={SEQ_LEN}, D_MODEL={D_MODEL}")

    # Create model and input tensor
    model = MoELayer(
        num_experts=NUM_EXPERTS,
        in_features=D_MODEL,
        out_features=D_MODEL,
        hidden_features=D_FF,
        top_k=TOP_K
    ).to(device).eval()

    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device, dtype=torch.float16)

    # Use float16 for more realistic performance measurement
    model.half()

    # Warm-up runs to stabilize GPU clocks and cache
    print("Warming up...")
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize()

    # Profiling context
    print("Starting profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.Activity.CPU,
            torch.profiler.Activity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/moe_profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(5):
            with torch.profiler.record_function("model_inference"):
                _ = model(x)
            prof.step()

    torch.cuda.synchronize()
    print("Profiling complete.")

    # Print the results sorted by total CUDA time
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

if __name__ == "__main__":
    profile_moe_layer()
