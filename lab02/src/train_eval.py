import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from src.network import TwoHiddenLayerMLP
from src.gradient_check import perform_numerical_gradient_check
from src.benchmark import benchmark_forward_backward_timing

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainPipeline")

def prepare_data(random_state: int = 42):
    """
    Task 1: Dataset Preparation
    Loads digits, scales by dividing by 16, one-hot encodes labels,
    and performs 70/30 stratified split with random_state=42.
    """
    logger.info("Loading scikit-learn digits dataset...")
    digits = load_digits()
    X = digits.data / 16.0  # Scale inputs by dividing by 16
    y = digits.target
    num_classes = 10
    Y = np.eye(num_classes)[y]  # One-hot encode labels

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.30, random_state=random_state, stratify=y
    )
    logger.info(f"Data Split: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples. Input dim=64.")
    return X_train, X_test, Y_train, Y_test

def train_model(
    model: TwoHiddenLayerMLP, 
    X_train: np.ndarray, 
    Y_train: np.ndarray, 
    X_test: np.ndarray, 
    Y_test: np.ndarray,
    epochs: int = 50, 
    batch_size: int = 32, 
    lr: float = 0.01
):
    """
    Task 6: Mini-batch SGD Training Loop
    Shuffles training set, uses batch_size=32, 50 epochs, per-epoch logging.
    """
    N = X_train.shape[0]
    num_batches = int(np.ceil(N / batch_size))
    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        # 1. Shuffle at start of each epoch
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, N)
            xb = X_shuffled[start:end]
            yb = Y_shuffled[start:end]

            # Forward pass
            _, _, batch_loss = model.forward(xb, yb)
            epoch_loss += batch_loss * (end - start)

            # Backward pass & SGD parameter update
            model.backward()
            model.step(lr)

        avg_train_loss = epoch_loss / N
        test_loss, test_acc = model.evaluate(X_test, Y_test)

        train_losses.append(avg_train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            logger.info(f"Epoch {epoch:2d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies,
        "final_test_accuracy": test_accuracies[-1]
    }

def main():
    os.makedirs("results", exist_ok=True)
    X_train, X_test, Y_train, Y_test = prepare_data(random_state=42)

    # -------------------------------------------------------------
    # Task 3 & 4: Forward Pass Checks & Untrained Loss Sanity Check
    # -------------------------------------------------------------
    print("\n>>> Task 3 & 4 Verification <<<")
    np.random.seed(42)
    check_model = TwoHiddenLayerMLP(input_dim=64, hidden_dim=16, output_dim=10)
    z3, probs, init_loss = check_model.forward(X_train[:32], Y_train[:32])

    print(f"Output logits shape : {z3.shape} (Expected: (32, 10))")
    print(f"Probabilities shape : {probs.shape} (Expected: (32, 10))")
    row_sums = np.sum(probs, axis=1)
    print(f"Min row sum: {np.min(row_sums):.8f}, Max row sum: {np.max(row_sums):.8f}")
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Softmax output rows do not sum to 1!"
    print("[PASS] Forward pass layer-by-layer shape and softmax unity check verified.")

    expected_init_loss = np.log(10)
    print(f"Initial Untrained Loss : {init_loss:.6f} (Expected ln(10) = {expected_init_loss:.6f})")
    assert abs(init_loss - expected_init_loss) < 0.05, f"Untrained loss {init_loss} deviates from ln(10)!"
    print("[PASS] Initial loss sanity check passed: approximately ln(10) = 2.3026.\n")

    # -------------------------------------------------------------
    # Task 5: Numerical Gradient Checking
    # -------------------------------------------------------------
    grad_check_results = perform_numerical_gradient_check(
        check_model, X_train[:32], Y_train[:32], eps=1e-5, num_checks=10
    )

    # -------------------------------------------------------------
    # Task 7: Latency Profiling
    # -------------------------------------------------------------
    timing_results = benchmark_forward_backward_timing(
        check_model, X_train[:32], Y_train[:32], num_runs=100
    )

    # -------------------------------------------------------------
    # Task 6: Grid Search Training Across Learning Rates & Hidden Widths
    # Learning rates: [0.001, 0.01, 0.1], Widths: [16, 64]
    # -------------------------------------------------------------
    learning_rates = [0.001, 0.01, 0.1]
    widths = [16, 64]
    grid_results = {}

    print("\n==================================================================")
    print("   Task 6: Training Experiments Grid (3 Learning Rates x 2 Widths)")
    print("==================================================================")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, W in enumerate(widths):
        ax = axes[idx]
        ax.set_title(f"Hidden Width W = {W}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        for lr in learning_rates:
            config_key = f"W{W}_lr{lr}"
            logger.info(f"\nTraining configuration: Width={W}, Learning Rate={lr}")

            np.random.seed(42)  # Consistent initialization
            model = TwoHiddenLayerMLP(input_dim=64, hidden_dim=W, output_dim=10, weight_scale=0.01)

            history = train_model(
                model, X_train, Y_train, X_test, Y_test,
                epochs=50, batch_size=32, lr=lr
            )

            grid_results[config_key] = {
                "width": W,
                "learning_rate": lr,
                "final_train_loss": float(history["train_losses"][-1]),
                "final_test_loss": float(history["test_losses"][-1]),
                "final_test_accuracy": float(history["final_test_accuracy"])
            }

            epochs_range = range(1, 51)
            ax.plot(epochs_range, history["train_losses"], label=f"Train lr={lr}", linewidth=2.0)
            ax.plot(epochs_range, history["test_losses"], label=f"Test lr={lr}", linestyle="--", alpha=0.8)

        ax.legend(fontsize=10, loc="upper right")

    plt.tight_layout()
    plot_path = "results/loss_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Loss curves saved to: {plot_path}")

    # Summary table
    print("\n==================================================================")
    print("   Final Performance Summary Across Configurations                ")
    print("==================================================================")
    print(f"{'Config':15s} | {'Width':6s} | {'Learning Rate':14s} | {'Final Test Loss':15s} | {'Test Accuracy':13s}")
    print("-" * 75)
    for k, v in grid_results.items():
        print(f"{k:15s} | {v['width']:<6d} | {v['learning_rate']:<14.3f} | {v['final_test_loss']:<15.4f} | {v['final_test_accuracy']*100:.2f}%")

    # Persist summary JSON
    summary_data = {
        "gradient_check": grad_check_results,
        "timing_benchmark": {
            "median_forward_ms": timing_results["median_forward_ms"],
            "median_backward_ms": timing_results["median_backward_ms"],
            "ratio": timing_results["ratio"]
        },
        "experiments": grid_results
    }
    with open("results/experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    logger.info("Saved complete experiment summary to results/experiment_summary.json")

if __name__ == "__main__":
    main()
