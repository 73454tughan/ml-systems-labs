"""
End-to-End Training & Systems Benchmark on make_moons: Custom Scalar Engine vs. PyTorch
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 05)
Architecture: 2 -> 8 -> 1 (ReLU hidden, Sigmoid output, BCE loss)
"""

import time
import random
import statistics
import json
import os
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
from engine import ScalarNode


class Neuron:
    def __init__(self, nin: int, nonlin: bool = True):
        # Xavier-style random initialization
        scale = (2.0 / nin) ** 0.5
        self.w = [ScalarNode(random.gauss(0.0, scale)) for _ in range(nin)]
        self.b = ScalarNode(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # act = sum(w * x) + b
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + (wi * xi)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int, nonlin: bool = True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params


class TinyScalarMLP:
    def __init__(self):
        # 2 -> 8 (ReLU) -> 1 (Linear followed by Sigmoid)
        self.l1 = Layer(2, 8, nonlin=True)
        self.l2 = Layer(8, 1, nonlin=False)

    def forward(self, x):
        h = self.l1(x)
        raw_out = self.l2(h)
        return raw_out.sigmoid()

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


# PyTorch Baseline Model with identical capacity
class PyTorchTinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def binary_cross_entropy(y_true, y_pred):
    # BCE = - (1/N) * sum( y * log(p) + (1-y) * log(1-p) )
    n = len(y_true)
    losses = []
    for yt, yp in zip(y_true, y_pred):
        eps = 1e-12
        p_safe = yp
        one_minus_p = (ScalarNode(1.0) - yp)
        if yt == 1:
            loss_i = -p_safe.log()
        else:
            loss_i = -one_minus_p.log()
        losses.append(loss_i)
    
    total = losses[0]
    for li in losses[1:]:
        total = total + li
    return total * (1.0 / n)


def train_custom_model(X, y, epochs=50, lr=0.2):
    random.seed(42)
    model = TinyScalarMLP()
    
    # Pre-wrap inputs into ScalarNodes
    X_nodes = [[ScalarNode(val) for val in row] for row in X]
    
    t0 = time.perf_counter()
    history = []
    
    for epoch in range(epochs):
        # Forward pass
        preds = [model.forward(x) for x in X_nodes]
        loss = binary_cross_entropy(y, preds)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # SGD parameter update
        for p in model.parameters():
            p.val -= lr * p.grad
            
        history.append(loss.val)
        
    duration = time.perf_counter() - t0
    
    # Accuracy check
    correct = sum((1 if p.val >= 0.5 else 0) == yt for p, yt in zip(preds, y))
    accuracy = correct / len(y)
    
    return duration, history[-1], accuracy, history


def train_pytorch_model(X, y, epochs=50, lr=0.2):
    torch.manual_seed(42)
    model = PyTorchTinyMLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    t0 = time.perf_counter()
    history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        
    duration = time.perf_counter() - t0
    
    with torch.no_grad():
        final_preds = (model(X_t) >= 0.5).float()
        acc = (final_preds == y_t).float().mean().item()
        
    return duration, history[-1], acc, history


def run_benchmark_suite():
    print("[BENCHMARK] Generating make_moons dataset (N=100, noise=0.1)...")
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    epochs = 40
    
    custom_times = []
    custom_losses = []
    custom_accs = []
    
    pytorch_times = []
    pytorch_losses = []
    pytorch_accs = []
    
    print("[BENCHMARK] Executing 5 independent trials...")
    for trial in range(5):
        c_time, c_loss, c_acc, _ = train_custom_model(X, y, epochs=epochs, lr=0.25)
        p_time, p_loss, p_acc, _ = train_pytorch_model(X, y, epochs=epochs, lr=0.25)
        
        custom_times.append(c_time)
        custom_losses.append(c_loss)
        custom_accs.append(c_acc)
        
        pytorch_times.append(p_time)
        pytorch_losses.append(p_loss)
        pytorch_accs.append(p_acc)
        
        print(f"  Trial {trial+1}/5 | Custom: {c_time*1000:.2f} ms (Loss: {c_loss:.4f}) | PyTorch: {p_time*1000:.2f} ms (Loss: {p_loss:.4f})")
        
    custom_median_ms = statistics.median(custom_times) * 1000.0
    pytorch_median_ms = statistics.median(pytorch_times) * 1000.0
    slowdown_ratio = custom_median_ms / pytorch_median_ms
    
    results = {
        'student': "Tughan Bulut Kurtulus (73454)",
        'dataset': "make_moons (n=100, noise=0.1)",
        'architecture': "2 -> 8 -> 1 (ReLU, Sigmoid)",
        'loss_function': "Binary Cross-Entropy (BCE)",
        'epochs': epochs,
        'trials': 5,
        'custom_engine': {
            'times_ms': [t * 1000 for t in custom_times],
            'median_time_ms': custom_median_ms,
            'mean_time_ms': statistics.mean(custom_times) * 1000,
            'final_loss_median': statistics.median(custom_losses),
            'final_accuracy_median': statistics.median(custom_accs)
        },
        'pytorch_engine': {
            'times_ms': [t * 1000 for t in pytorch_times],
            'median_time_ms': pytorch_median_ms,
            'mean_time_ms': statistics.mean(pytorch_times) * 1000,
            'final_loss_median': statistics.median(pytorch_losses),
            'final_accuracy_median': statistics.median(pytorch_accs)
        },
        'slowdown_custom_vs_pytorch': slowdown_ratio
    }
    
    os.makedirs('results', exist_ok=True)
    with open('results/moons_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Custom Engine Median:  {custom_median_ms:.2f} ms")
    print(f"PyTorch Engine Median: {pytorch_median_ms:.2f} ms")
    print(f"Overhead Factor:       {slowdown_ratio:.1f}x (due to Python object allocation & dynamic dispatch)")
    return results


if __name__ == '__main__':
    run_benchmark_suite()
