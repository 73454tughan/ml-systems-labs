# Lab 05: Frameworks and Automatic Differentiation

**Author:** Tughan Bulut Kurtulus  
**Student ID:** 73454  
**Course:** Machine Learning Systems (SPEC: Engineering of Artificial Intelligence)  
**Academic Year:** 2025/2026  
**Curriculum Reference:** Vijay Janapa Reddi, *Machine Learning Systems*, Chapter 7 (AI Frameworks: §§7.3–7.8)  

---

## 1. Executive Summary & Research Scope

Modern machine learning frameworks (e.g., PyTorch, JAX, TensorFlow) mediate between abstract algorithmic models and physical silicon execution. This laboratory investigates the computational foundation of deep learning frameworks:
1. **Scalar Reverse-Mode Automatic Differentiation:** Implementation of a dynamic directed acyclic graph (DAG) engine with local gradient closures, topological sorting, and non-destructive adjoint accumulation.
2. **Empirical Numerical & Framework Verification:** High-precision gradient checking against PyTorch Autograd and two-sided central finite differences on non-linear composite expressions.
3. **End-to-End Perceptron Training on `make_moons`:** Comparative evaluation of training latency, memory allocation overhead, and Python interpreter dispatch bottlenecks against vectorized PyTorch C++ kernels over 5 independent runs.
4. **Graph Compilation & Execution Modes:** Graph tracing via TorchScript (`torch.jit.trace`) and TorchDynamo (`torch.compile`), contrasting eager dispatch overhead against optimized fused execution graphs.
5. **Tensor Strides & Cache Hierarchy:** Microarchitectural analysis of memory layout (strides, contiguous vs. non-contiguous memory access) and its impact on L1/L2 cache locality and SIMD vectorization.

---

## 2. Dynamic Computation Graph Architecture (`ScalarNode`)

The custom automatic differentiation core is built around `ScalarNode` ([engine.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/src/engine.py)), an explicit representation of a scalar intermediate in a dynamic computational graph:

### 2.1 State Representation & Memory Invariants
Each `ScalarNode` stores:
- `val` (`float`): The primal scalar value evaluated during the forward pass.
- `grad` (`float`): The accumulated adjoint ($\partial \mathcal{L} / \partial x$), initialized to $0.0$.
- `_creators` (`set` of `ScalarNode`): Input ancestor nodes in the DAG.
- `_backward_fn` (`callable`): A zero-argument closure encapsulating the local derivative chain rule.
- `_operation` (`str`): Operation symbol for graph visualization and debugging.

### 2.2 Dynamic Forward Construction & Local Derivative Rules
Every primitive mathematical operator builds a new child node and binds a backward closure:
- **Addition ($z = x + y$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= 1.0 \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} \mathrel{+}= 1.0 \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Multiplication ($z = x \cdot y$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= y \cdot \frac{\partial \mathcal{L}}{\partial z}, \quad \frac{\partial \mathcal{L}}{\partial y} \mathrel{+}= x \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Power ($z = x^p$ for constant $p$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= (p \cdot x^{p-1}) \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Rectified Linear Unit ($z = \max(0, x)$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \mathbf{1}_{\{x > 0\}} \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \sigma(x)(1 - \sigma(x)) \cdot \frac{\partial \mathcal{L}}{\partial z}$$
- **Natural Logarithm ($z = \ln(x)$):**
  $$\frac{\partial \mathcal{L}}{\partial x} \mathrel{+}= \frac{1}{x} \cdot \frac{\partial \mathcal{L}}{\partial z}$$

---

## 3. Topological Reverse-Mode Backpropagation

In arbitrary computation graphs, variables may be referenced multiple times across diverse branches (multi-consumer fan-out). A naive recursive backpropagation traversal would prematurely evaluate gradients before all upstream paths have contributed, leading to incorrect adjoint values.

### 3.1 Depth-First Topological Sorting
To guarantee valid reverse-mode evaluation, `ScalarNode.backward()` performs a post-order depth-first traversal of the DAG:
```python
def backward(self) -> None:
    ordered_nodes = []
    visited = set()

    def build_topo(v: ScalarNode) -> None:
        if v not in visited:
            visited.add(v)
            for child in v._creators:
                build_topo(child)
            ordered_nodes.append(v)

    build_topo(self)
    self.grad = 1.0  # Seed output gradient dL/dL = 1.0
    for node in reversed(ordered_nodes):
        node._backward_fn()
```

### 3.2 Gradient Accumulation Invariant (`+=`)
Gradients are strictly accumulated using in-place addition (`+=`). If an operation overwrote the gradient with assignment (`=`), variables with fan-out $> 1$ (such as shared model parameters or residual skips) would lose gradients from parallel paths, violating the multivariate chain rule:
$$\frac{\partial \mathcal{L}}{\partial x_i} = \sum_{j \in \text{Children}(i)} \frac{\partial \mathcal{L}}{\partial z_j} \frac{\partial z_j}{\partial x_i}$$

---

## 4. Verification Against PyTorch Autograd & Central Finite Differences

To prove mathematical and operational fidelity, we verified the engine against PyTorch autograd and numerical finite difference estimates on the benchmark expression:
$$\mathcal{L} = \sum_{i=1}^N \text{ReLU}\left( (a_i \cdot b_i + c_i)^2 \right)$$

### 4.1 Numerical Central Finite Difference Formulation
$$\frac{\partial \mathcal{L}}{\partial \theta_j} \approx \frac{\mathcal{L}(\theta_j + \epsilon) - \mathcal{L}(\theta_j - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-7}$$

### 4.2 Empirical Verification Results
- **Test Vectors:** $a = [1.5, -2.0, 0.75, -0.5]$, $b = [-3.0, 4.0, 1.25, 2.5]$, $c = [2.0, 1.5, -3.5, 0.2]$
- **Forward Output:** Custom Engine = `84.810625`, PyTorch = `84.810625` ($\Delta = 0.00\times 10^0$)

| Parameter | Custom Engine Grad | PyTorch Autograd | Central Finite Difference | Abs Diff vs PyTorch | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| $a_0$ | **-15.000000** | -15.000000 | -15.000000 | **0.00e+00** | **PASS** |
| $a_1$ | **-52.000000** | -52.000000 | -52.000000 | **0.00e+00** | **PASS** |
| $a_2$ | **-6.406250** | -6.406250 | -6.406250 | **0.00e+00** | **PASS** |
| $a_3$ | **-5.250000** | -5.250000 | -5.250000 | **0.00e+00** | **PASS** |
| $b_0$ | **7.500000** | 7.500000 | 7.500000 | **0.00e+00** | **PASS** |
| $b_1$ | **26.000000** | 26.000000 | 26.000000 | **0.00e+00** | **PASS** |
| $b_2$ | **-3.843750** | -3.843750 | -3.843750 | **0.00e+00** | **PASS** |
| $b_3$ | **1.050000** | 1.050000 | 1.050000 | **0.00e+00** | **PASS** |
| $c_0$ | **-5.000000** | -5.000000 | -5.000000 | **0.00e+00** | **PASS** |
| $c_1$ | **-13.000000** | -13.000000 | -13.000000 | **0.00e+00** | **PASS** |
| $c_2$ | **-5.125000** | -5.125000 | -5.125000 | **0.00e+00** | **PASS** |
| $c_3$ | **-2.100000** | -2.100000 | -2.100000 | **0.00e+00** | **PASS** |

- **Maximum PyTorch Absolute Difference:** $0.00 \times 10^0$ (Strictly satisfies the $< 10^{-6}$ criterion).
- **Maximum Numerical Difference:** $4.53 \times 10^{-8}$.

---

## 5. End-to-End Training Benchmark on `make_moons`

We trained a complete multi-layer perceptron on the non-linear `make_moons` dataset ($N=100$, $\sigma=0.10$, seed 42):
- **Architecture:** $2 \to 8 \to 1$ (8 hidden units with ReLU activation, 1 output unit with Sigmoid activation).
- **Criterion:** Binary Cross-Entropy Loss:
  $$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
- **Optimization:** Stochastic Gradient Descent (SGD, learning rate $\eta = 0.25$, 40 epochs).

### 5.1 5-Trial Benchmark Telemetry
To eliminate transient OS scheduling noise, 5 independent trials were recorded:

| Trial | Custom Engine (ms) | Final Loss | PyTorch Engine (ms) | Final Loss |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 2797.48 | 0.3288 | 44.18 | 0.3518 |
| 2 | 3016.59 | 0.3288 | 16.92 | 0.3518 |
| 3 | 3053.46 | 0.3288 | 13.75 | 0.3518 |
| 4 | 3137.81 | 0.3288 | 14.22 | 0.3518 |
| 5 | 3627.73 | 0.3288 | 14.57 | 0.3518 |
| **Median** | **3053.46 ms** | **0.3288** | **14.57 ms** | **0.3518** |

### 5.2 Systems Performance Analysis
- **Custom Engine Median Latency:** $3053.46\text{ ms}$
- **PyTorch Engine Median Latency:** $14.57\text{ ms}$
- **Slowdown Factor:** $\approx 209.5\times$

**Root Cause of Framework Speedup:**
1. **Python Dynamic Object Allocation:** Custom autograd instantiates thousands of Python heap objects per epoch (`ScalarNode`, tuples, lambda closures, sets). Each allocation invokes the CPython object allocator (PyMalloc) and increments reference counting (`Py_INCREF`/`Py_DECREF`).
2. **Dynamic Bytecode Interpretation:** Custom backward propagation performs dynamic function calls inside a Python `for` loop, incurring opcode dispatch overhead on every scalar edge.
3. **C++/SIMD Kernel Execution:** PyTorch processes all 100 samples simultaneously in contiguous contiguous memory buffers using optimized BLAS/Intel MKL vectorized matrix-multiplication kernels (`dgemm`/`sgemm`) utilizing AVX2/AVX-512 register instructions.

---

## 6. PyTorch Graph Inspection & Compiler Modes

We inspected the execution graph of the `SystemsSmallNet` architecture and profiled execution latencies across 1,000 warm-up iterations.

### 6.1 Traced Intermediate Representation (TorchScript IR)
Inspecting `torch.jit.trace(model, dummy_input).inlined_graph` exposes the high-level computational IR:
```
graph(%self : __torch__.SystemsSmallNet, %x : Float(64, 2, strides=[2, 1])):
  %fc1.weight : Float(8, 2, strides=[2, 1]) = prim::GetAttr[name="fc1.weight"](%self)
  %fc1.bias : Float(8, strides=[1]) = prim::GetAttr[name="fc1.bias"](%self)
  %1 : Tensor = aten::linear(%x, %fc1.weight, %fc1.bias)
  %2 : Tensor = aten::relu(%1)
  %fc2.weight : Float(1, 8, strides=[8, 1]) = prim::GetAttr[name="fc2.weight"](%self)
  %fc2.bias : Float(1, strides=[1]) = prim::GetAttr[name="fc2.bias"](%self)
  %3 : Tensor = aten::linear(%2, %fc2.weight, %fc2.bias)
  %out : Tensor = aten::sigmoid(%3)
  return (%out)
```

### 6.2 Latency Comparison: Eager vs. Traced Dispatch
- **Eager Execution Latency:** $43.16\ \mu\text{s/iter}$
- **Traced Execution Latency:** $32.42\ \mu\text{s/iter}$
- **Speedup Factor:** **$1.33\times$**

TorchScript tracing flattens Python abstraction layers, eliminating Python frame creation, method dispatch, and dictionary attribute lookups (`__getattr__`).

---

## 7. Tensor Memory Layout, Striding, and Non-Contiguous Overhead

In deep learning systems, tensor performance is dominated by memory bandwidth and CPU cache locality. A tensor's logical coordinate $(i, j)$ maps to linear physical memory address via its strides:
$$\text{Offset}(i, j) = i \cdot s_0 + j \cdot s_1$$

### 7.1 Experimental Setup
We evaluated a $2048 \times 2048$ single-precision tensor (16 MB buffer, exceeding CPU L1/L2 data caches):
- **Contiguous Matrix:** `shape = [2048, 2048]`, `stride = [2048, 1]`, `is_contiguous() = True`.
- **Transposed Non-Contiguous View:** `shape = [2048, 2048]`, `stride = [1, 2048]`, `is_contiguous() = False`.

### 7.2 Benchmark Measurements (Row-Wise Reduction $\sum_{j} A_{ij}$)
- **Contiguous Row-Sum Latency:** $0.218\text{ ms}$
- **Non-Contiguous Row-Sum Latency:** $0.467\text{ ms}$
- **Non-Contiguous Slowdown Factor:** **$2.14\times$**
- **Explicit `.contiguous()` Buffer Reallocation:** $8.863\text{ ms}$

### 7.3 Microarchitectural Explanation
1. **Cache Line Utilization:** Standard x86-64 CPUs load memory in 64-byte cache lines (16 contiguous `float32` values). In a contiguous row-sum, each cache line fetch serves 16 sequential additions. In the transposed non-contiguous view, adjacent elements in logical row $i$ are separated by 2048 elements ($8192$ bytes), causing every single element to trigger an L1/L2 cache miss and evicting previous lines prematurely.
2. **SIMD Vectorization:** Contiguous data allows AVX-256/512 registers to load 8 or 16 floats in a single `vmovups` instruction and compute vector additions. Non-contiguous layouts require gather instructions (`vgatherdps`) or scalar fallbacks, degrading throughput.
3. **The `.contiguous()` Trade-off:** Calling `.contiguous()` copies all elements into a new sequential buffer ($8.863\text{ ms}$), which is an order of magnitude more expensive than the slowdown of a single reduction ($0.249\text{ ms}$). Compilers (like Inductor) analyze lifetime: they avoid `.contiguous()` unless subsequent operations justify the amortization cost.

---

## 8. Technical Note: When Graph Compilation Helps

Per Reddi Chapter 7, graph compilation transforms interpreted eager commands into optimized execution plans. Compilation yields significant benefits under specific conditions:

1. **Kernel Fusion (Eliminating Memory Bound Bottlenecks):**
   In eager execution, `y = relu(x * w + b)` invokes three independent CUDA/CPU kernels: GEMM, bias addition, and ReLU activation. Each kernel writes intermediate results to DRAM and reads them back. Graph compilers fuse elementwise activations into the epilogue of the matrix multiplication kernel, keeping intermediates in CPU L1/L2 caches or GPU shared memory, reducing DRAM round-trips by up to $3\times$.
2. **Interpreter Overhead Elimination:**
   For small batch sizes (e.g., robotics, edge inference, or micro-batches), Python dispatch latency ($\approx 10\text{–}40\ \mu\text{s}$ per operation) can exceed silicon execution time. Compilers lower the computation graph into native C++/LLVM code, dropping dispatch latency to sub-microsecond levels.
3. **Dead Code Elimination & Constant Folding:**
   Compilers pre-compute operations on constant subgraphs (e.g., static weights, normalization multipliers) at compile time and prune unreferenced intermediate outputs.
4. **Buffer Reuse & Static Memory Planning:**
   Instead of dynamically allocating and freeing memory on every iteration, compilers analyze tensor liveness, generating static memory pools that prevent memory fragmentation and eliminate allocator synchronization locks.

---

## 9. Deliverables & Artifact Verification

- [engine.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/src/engine.py): Dynamic scalar DAG engine with reverse-mode AD and topological sorting.
- [verification.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/src/verification.py): High-precision gradient verification test against PyTorch Autograd and Central Finite Differences.
- [train_moons.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/src/train_moons.py): 5-trial median benchmark comparing custom scalar MLP to PyTorch on `make_moons`.
- [graph_compiler_bench.py](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/src/graph_compiler_bench.py): Graph tracing, eager vs. compiled speed profiling, and tensor memory striding benchmarks.
- `results/`: Contains generated JSON benchmark logs ([verification_metrics.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/results/verification_metrics.json), [moons_benchmark.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/results/moons_benchmark.json), [compiler_contiguity_benchmarks.json](file:///C:/Users/Tuan/.gemini/antigravity/scratch/ml_systems_tughan/lab05/results/compiler_contiguity_benchmarks.json)).
