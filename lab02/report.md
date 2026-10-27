# Lab 02: A Neural Network from Scratch

**Student:** Tughan Bulut Kurtulus (ID: 73454)  
**Readings:** Reddi, Chapter 3 DL Primer: §§3.4–3.6  

---

## 1. Goal
The objective of this laboratory is to design and implement a two-hidden-layer fully connected artificial neural network entirely from first principles using pure NumPy. The implementation encompasses layer-by-layer forward propagation, numerically stable softmax cross-entropy loss, exact analytical backpropagation derived via the matrix calculus chain rule, finite-difference numerical gradient checking (+/- 1e-5), latency profiling comparing forward and backward passes, and hyperparameter grid training across 6 configurations on the Scikit-Learn `load_digits` dataset.

---

## 2. Method
1. **Dataset Ingestion & Preprocessing:** Loaded the 8x8 handwritten digits dataset (`load_digits`), normalized pixel intensities by dividing by 16.0, one-hot encoded the 10 target classes, and generated a reproducible 70/30 stratified split pinned to `random_state=42` (1,257 training, 540 testing vectors).
2. **Network Topology & Parameter Initialization:**
   - Architecture: 64 inputs -> Dense Layer 1 (W1: 64 x W, b1: W) -> ReLU -> Dense Layer 2 (W2: W x W, b2: W) -> ReLU -> Dense Layer 3 (W3: W x 10, b3: 10) -> Softmax.
   - Evaluated hidden dimensions: W in [16, 64].
   - Weights initialized from Gaussian N(0, 0.01^2) and biases set to zero.
3. **Forward Pass Integrity & Loss Sanity Check:**
   - Validated that softmax output rows sum to 1.00000000 within tolerance (< 1e-6).
   - Confirmed untrained cross-entropy loss approximates theoretical ln(10) = 2.302585. Observed: **`2.302557`**.
4. **Analytical Gradient Derivation & Numerical Verification:**
   - Derived parameter gradients analytically using matrix calculus (detailed proofs in `src/DERIVATIONS.md`).
   - Nudged 10 randomly selected weights by +/- 1e-5 and calculated two-sided numerical slopes.
   - Evaluated relative error |analytic - numeric| / max(|analytic|, |numeric|, 1e-12). Verified relative error < 1e-6 for >= 9/10 weights.
5. **Mini-Batch SGD Grid Training:**
   - Mini-batch size 32, 50 epochs with epoch-wise shuffling, testing learning rates [0.001, 0.01, 0.1] across widths [16, 64].
   - Logged epoch trajectories and plotted convergence to `results/loss_curves.png`.
6. **Execution Profiling (100 Runs):** Measured forward and backward pass durations and analyzed computational complexity in terms of General Matrix Multiplication (GEMM) operations.

---

## 3. Results Table or Plot

### Analytical vs Numerical Gradient Checking Verification (10 Checks)

| Check # | Parameter Target | Analytical Derivative | Numerical Slope | Relative Error | Tolerance Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Layer 1 (W1) @ [37, 11] | `+2.39568450e-05` | `+2.39568587e-05` | `5.74e-07` | **PASS (< 1e-6)** |
| 2 | Layer 2 (W2) @ [12, 8]  | `+2.28132509e-05` | `+2.28132402e-05` | `4.72e-07` | **PASS (< 1e-6)** |
| 3 | Layer 3 (W3) @ [15, 9]  | `+7.01319405e-06` | `+7.01319003e-06` | `5.74e-07` | **PASS (< 1e-6)** |
| 4 | Layer 1 (W1) @ [11, 5]  | `-8.72545491e-05` | `-8.72545591e-05` | `1.15e-07` | **PASS (< 1e-6)** |
| 5 | Layer 2 (W2) @ [15, 0]  | `-3.05108401e-05` | `-3.05108383e-05` | `6.02e-08` | **PASS (< 1e-6)** |
| 6 | Layer 3 (W3) @ [0, 1]   | `-7.88078652e-05` | `-7.88078713e-05` | `7.77e-08` | **PASS (< 1e-6)** |
| 7 | Layer 1 (W1) @ [12, 7]  | `-9.07251729e-05` | `-9.07251829e-05` | `1.10e-07` | **PASS (< 1e-6)** |
| 8 | Layer 2 (W2) @ [13, 12] | `+0.00000000e+00` | `+0.00000000e+00` | `0.00e+00` | **PASS (< 1e-6)** |
| 9 | Layer 3 (W3) @ [6, 9]   | `+8.22058316e-05` | `+8.22058199e-05` | `1.42e-07` | **PASS (< 1e-6)** |
| 10 | Layer 1 (W1) @ [50, 4] | `-4.53115729e-05` | `-4.53115545e-05` | `4.06e-07` | **PASS (< 1e-6)** |

**Outcome:** **10 / 10 weights passed** with relative error strictly below 1e-6 (100% compliance).

### Hyperparameter Grid Evaluation Summary (50 Epochs)

| Configuration | Hidden Width (W) | Learning Rate (lr) | Final Train Loss | Final Test Loss | Final Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `W16_lr0.001` | 16 | 0.001 | 2.3026 | 2.3025 | 10.19% |
| `W16_lr0.01`  | 16 | 0.010 | 2.3026 | 2.3025 | 10.19% |
| `W16_lr0.1`   | 16 | 0.100 | 0.8444 | 0.9538 | 67.22% |
| `W64_lr0.001` | 64 | 0.001 | 2.3025 | 2.3025 | 10.19% |
| `W64_lr0.01`  | 64 | 0.010 | 2.3024 | 2.3023 | 10.19% |
| **`W64_lr0.1`** | **64** | **0.100** | **0.1371** | **0.2067** | **94.44%** |

### Training & Validation Loss Trajectories
![Loss Curves](results/loss_curves.png)

### Execution Latency & GEMM Analysis
- **Median Forward Pass Duration (100 runs):** `0.3586 ms`
- **Median Backward Pass Duration (100 runs):** `0.2266 ms`
- **Computational Complexity Rationale:** Forward evaluation requires **3 General Matrix Multiplications (GEMMs)** (X @ W1, A1 @ W2, A2 @ W3). Conversely, backpropagation requires **5 GEMMs** (A2.T @ dZ3, dZ3 @ W3.T, A1.T @ dZ2, dZ2 @ W2.T, X.T @ dZ1) to simultaneously propagate errors backward through activations and compute parameter gradients.

---

## 4. Three Conclusions in Your Own Words
1. **Numerical Gradient Checking is Indispensable for Derivative Validation:** Validating analytic chain rule derivations against centered finite differences (+/- 1e-5) provides mathematical certainty of backpropagation correctness, preventing silent implementation bugs that corrupt gradient descent.
2. **Weight Scale Interacts Heavily with Learning Rate Dynamics:** Because parameters were initialized from a small Gaussian standard deviation (sigma = 0.01), small step sizes (lr <= 0.01) stalled in symmetric plateaus near the initial loss (ln(10) = 2.30), whereas lr = 0.1 successfully broke symmetry to reach 94.44% test accuracy on the W = 64 architecture.
3. **Backpropagation Inherently Demands Double the Matrix Multiplications of Inference:** In feedforward architectures, backward passes require roughly double the floating-point operations of inference because gradients must be computed both with respect to layer weights (for updates) and with respect to layer activations (to continue error backpropagation).
