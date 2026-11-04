"""
PyTorch Graph Inspection, Eager vs. Compiled Profiling, and Tensor Contiguity Benchmarks
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 05)
"""

import time
import json
import os
import torch
import torch.nn as nn


class SystemsSmallNet(nn.Module):
    """
    Standard 2-layer perceptron matching the lab specification:
    Input: 2 features -> Linear(2, 8) -> ReLU -> Linear(8, 1) -> Sigmoid
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        out = self.sigmoid(self.fc2(h))
        return out


def inspect_graph_and_compile():
    print("[COMPILER] Instantiating SystemsSmallNet and tracing IR graph...")
    model = SystemsSmallNet().eval()
    dummy_input = torch.randn(64, 2, dtype=torch.float32)

    # 1. Traced Graph Inspection
    traced_model = torch.jit.trace(model, dummy_input)
    traced_graph_str = str(traced_model.inlined_graph)

    # 2. PyTorch 2.0 TorchDynamo/Inductor Compilation
    has_compile = hasattr(torch, 'compile')
    compiled_model = None
    if has_compile:
        try:
            compiled_model = torch.compile(model)
            # Run one warm-up step to trigger JIT compilation
            _ = compiled_model(dummy_input)
            compile_status = "Available and functional"
        except Exception as e:
            compile_status = f"Failed ({type(e).__name__}: {e})"
            compiled_model = None
    else:
        compile_status = "Unsupported on current torch build"

    # 3. Latency Benchmarking: Eager vs. Traced vs. Compiled
    warmup_iters = 100
    bench_iters = 1000

    # Warm-up phase
    for _ in range(warmup_iters):
        _ = model(dummy_input)
        _ = traced_model(dummy_input)
        if compiled_model is not None:
            _ = compiled_model(dummy_input)

    # Measure Eager Mode
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = model(dummy_input)
    t_eager_total = time.perf_counter() - t0
    eager_lat_us = (t_eager_total / bench_iters) * 1e6

    # Measure Traced Mode
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        _ = traced_model(dummy_input)
    t_traced_total = time.perf_counter() - t0
    traced_lat_us = (t_traced_total / bench_iters) * 1e6

    # Measure Compiled Mode (if available)
    if compiled_model is not None:
        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = compiled_model(dummy_input)
        t_compiled_total = time.perf_counter() - t0
        compiled_lat_us = (t_compiled_total / bench_iters) * 1e6
    else:
        compiled_lat_us = None

    print(f"  Eager Latency:    {eager_lat_us:.2f} µs/iter")
    print(f"  Traced Latency:   {traced_lat_us:.2f} µs/iter (Speedup: {eager_lat_us/traced_lat_us:.2f}x)")
    if compiled_lat_us:
        print(f"  Compiled Latency: {compiled_lat_us:.2f} µs/iter (Speedup: {eager_lat_us/compiled_lat_us:.2f}x)")

    # 4. Tensor Memory Layout & Contiguity Impact Analysis
    print("\n[MEMORY] Analyzing tensor layout, stride patterns, and non-contiguous overhead...")
    # Base matrix: 2048 x 2048 float32 (16 MB memory footprint, exceeds L1/L2 CPU caches)
    N = 2048
    A_contig = torch.randn(N, N, dtype=torch.float32)
    # Create transposed non-contiguous view
    A_non_contig = A_contig.t()
    # Explicitly verify memory layout invariants
    contig_meta = {
        'shape': list(A_contig.shape),
        'dtype': str(A_contig.dtype),
        'stride': list(A_contig.stride()),
        'is_contiguous': A_contig.is_contiguous()
    }
    non_contig_meta = {
        'shape': list(A_non_contig.shape),
        'dtype': str(A_non_contig.dtype),
        'stride': list(A_non_contig.stride()),
        'is_contiguous': A_non_contig.is_contiguous()
    }

    # Benchmark reduction operation: Row sum (dim=1)
    # For contiguous tensor, row elements are adjacent in physical memory (stride 1 across columns)
    # For transposed tensor, row elements are separated by N*4 bytes (stride 2048), causing L1/L2 cache misses
    ops_iters = 200

    # Warm-up
    for _ in range(20):
        _ = torch.sum(A_contig, dim=1)
        _ = torch.sum(A_non_contig, dim=1)

    t0 = time.perf_counter()
    for _ in range(ops_iters):
        _ = torch.sum(A_contig, dim=1)
    contig_sum_time = (time.perf_counter() - t0) / ops_iters

    t0 = time.perf_counter()
    for _ in range(ops_iters):
        _ = torch.sum(A_non_contig, dim=1)
    non_contig_sum_time = (time.perf_counter() - t0) / ops_iters

    # Cost of calling .contiguous() to restore locality
    t0 = time.perf_counter()
    for _ in range(ops_iters):
        _ = A_non_contig.contiguous()
    contiguous_alloc_time = (time.perf_counter() - t0) / ops_iters

    print(f"  Contiguous Row-Sum Latency:     {contig_sum_time * 1e3:.3f} ms")
    print(f"  Non-Contiguous Row-Sum Latency: {non_contig_sum_time * 1e3:.3f} ms")
    print(f"  Non-Contiguous Slowdown Factor: {non_contig_sum_time / contig_sum_time:.2f}x")
    print(f"  Explicit .contiguous() Copy:    {contiguous_alloc_time * 1e3:.3f} ms")

    results = {
        'student': "Tughan Bulut Kurtulus (73454)",
        'execution_modes': {
            'eager_us': eager_lat_us,
            'traced_us': traced_lat_us,
            'compiled_us': compiled_lat_us,
            'compile_status': compile_status,
            'speedup_traced_vs_eager': eager_lat_us / traced_lat_us,
            'speedup_compiled_vs_eager': (eager_lat_us / compiled_lat_us) if compiled_lat_us else None
        },
        'tensor_layout_analysis': {
            'contiguous_tensor': contig_meta,
            'non_contiguous_tensor': non_contig_meta,
            'contig_row_sum_ms': contig_sum_time * 1e3,
            'non_contig_row_sum_ms': non_contig_sum_time * 1e3,
            'non_contig_slowdown_ratio': non_contig_sum_time / contig_sum_time,
            'contiguous_reallocation_cost_ms': contiguous_alloc_time * 1e3
        },
        'traced_ir_graph': traced_graph_str[:1200]
    }

    os.makedirs('results', exist_ok=True)
    with open('results/compiler_contiguity_benchmarks.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    inspect_graph_and_compile()
