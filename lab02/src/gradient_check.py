import numpy as np
from src.network import TwoHiddenLayerMLP

def perform_numerical_gradient_check(model: TwoHiddenLayerMLP, X_batch: np.ndarray, Y_batch: np.ndarray, eps: float = 1e-5, num_checks: int = 10, seed: int = 1):
    """
    Task 5: Numerical Gradient Verification
    Nudges 10 random weights by +/- eps, compares analytic and numerical slopes.
    Computes relative error: |analytic - numeric| / max(|analytic|, |numeric|, 1e-12).
    Requires relative error < 1e-6 for at least 9 of 10 weights.
    """
    print("==================================================================")
    print("   Task 5: Analytical vs. Numerical Gradient Checking Suite       ")
    print("==================================================================")
    print(f"Epsilon perturbation : {eps}")
    print(f"Evaluated Batch Size : {X_batch.shape[0]} samples\n")

    # 1. Forward and backward to obtain analytical gradients
    model.forward(X_batch, Y_batch)
    model.backward()

    # Collect weight matrices and their gradients
    param_groups = [
        ("Layer 1 (W1)", model.fc1.W, model.fc1.dW),
        ("Layer 2 (W2)", model.fc2.W, model.fc2.dW),
        ("Layer 3 (W3)", model.fc3.W, model.fc3.dW)
    ]

    checks = []
    passing_count = 0
    rng = np.random.RandomState(seed)

    for i in range(num_checks):
        # Select layer in round-robin fashion
        layer_idx = i % len(param_groups)
        layer_name, W, dW = param_groups[layer_idx]

        r = rng.randint(0, W.shape[0])
        c = rng.randint(0, W.shape[1])

        analytic_grad = dW[r, c]

        # Numerical perturbation: +eps
        W[r, c] += eps
        _, _, loss_plus = model.forward(X_batch, Y_batch)

        # Numerical perturbation: -eps
        W[r, c] -= 2 * eps
        _, _, loss_minus = model.forward(X_batch, Y_batch)

        # Reset weight back to original
        W[r, c] += eps

        numeric_grad = (loss_plus - loss_minus) / (2.0 * eps)

        # Relative error formula
        denominator = max(abs(analytic_grad), abs(numeric_grad), 1e-12)
        rel_error = abs(analytic_grad - numeric_grad) / denominator

        passed = rel_error < 1e-6
        if passed:
            passing_count += 1

        check_record = {
            "check": i + 1,
            "layer": layer_name,
            "index": f"[{r}, {c}]",
            "analytic": float(analytic_grad),
            "numeric": float(numeric_grad),
            "rel_error": float(rel_error),
            "passed": bool(passed)
        }
        checks.append(check_record)

        status_str = "PASS" if passed else "MARGINAL"
        print(f"Check {i+1:2d} | {layer_name:14s} @ [{r:2d}, {c:2d}] | Analytic: {analytic_grad:+.8e} | Numeric: {numeric_grad:+.8e} | RelErr: {rel_error:.2e} -> {status_str}")

    print(f"\nGradient Check Result: {passing_count} / {num_checks} weights passed (Requirement: >= 9/10).")
    assert passing_count >= 9, f"Gradient checking failed! Only {passing_count}/10 passed."
    print("[SUCCESS] Analytical gradient derivations verified numerically with high precision!\n")

    return {
        "passing_count": passing_count,
        "total_checks": num_checks,
        "details": checks
    }
