import time
import numpy as np
from src.network import TwoHiddenLayerMLP

def benchmark_forward_backward_timing(model: TwoHiddenLayerMLP, X_batch: np.ndarray, Y_batch: np.ndarray, num_runs: int = 100):
    """
    Task 7: Execution Profiling
    Times forward and backward passes separately across 100 iterations.
    Computes median execution time and compares computational complexity.
    """
    print("==================================================================")
    print("   Task 7: Forward vs Backward Pass Latency Profiling (100 runs)  ")
    print("==================================================================")

    forward_times = []
    backward_times = []

    # Warmup runs
    for _ in range(10):
        model.forward(X_batch, Y_batch)
        model.backward()

    for _ in range(num_runs):
        # Time Forward Pass
        t0 = time.perf_counter()
        model.forward(X_batch, Y_batch)
        t_fwd = (time.perf_counter() - t0) * 1000.0  # in ms
        forward_times.append(t_fwd)

        # Time Backward Pass
        t1 = time.perf_counter()
        model.backward()
        t_bwd = (time.perf_counter() - t1) * 1000.0  # in ms
        backward_times.append(t_bwd)

    median_fwd = float(np.median(forward_times))
    median_bwd = float(np.median(backward_times))
    ratio = median_bwd / median_fwd if median_fwd > 0 else 0.0

    print(f"Median Forward Pass Duration  : {median_fwd:.4f} ms")
    print(f"Median Backward Pass Duration : {median_bwd:.4f} ms")
    print(f"Backward / Forward Ratio      : {ratio:.2f}x\n")

    explanation = (
        "Computational Complexity Explanation:\n"
        "The backward pass consistently dominates runtime because it requires roughly double the number of "
        "dense matrix multiplications (GEMM operations):\n"
        "1. Forward Pass (3 Matmuls):\n"
        "   - Z1 = X @ W1          (Shape: [B, 64] @ [64, W])\n"
        "   - Z2 = A1 @ W2         (Shape: [B, W] @ [W, W])\n"
        "   - Z3 = A2 @ W3         (Shape: [B, W] @ [W, 10])\n"
        "2. Backward Pass (5-6 Matmuls):\n"
        "   - dW3 = A2.T @ dZ3     (Shape: [W, B] @ [B, 10])   -> for weight gradient\n"
        "   - dA2 = dZ3 @ W3.T     (Shape: [B, 10] @ [10, W])  -> for activation gradient\n"
        "   - dW2 = A1.T @ dZ2     (Shape: [W, B] @ [B, W])    -> for weight gradient\n"
        "   - dA1 = dZ2 @ W2.T     (Shape: [B, W] @ [W, W])    -> for activation gradient\n"
        "   - dW1 = X.T @ dZ1      (Shape: [64, B] @ [B, W])   -> for weight gradient\n"
        "Because backpropagation must simultaneously calculate gradients with respect to both layer inputs (to propagate "
        "errors backward) and layer parameters (to update weights), it naturally demands approximately 2x the floating-point "
        "operations (FLOPs) of forward inference."
    )
    print(explanation)

    return {
        "median_forward_ms": median_fwd,
        "median_backward_ms": median_bwd,
        "ratio": ratio,
        "explanation": explanation
    }
