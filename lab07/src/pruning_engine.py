"""
Magnitude & Structured Pruning Engine with Fine-Tuning & Storage Auditing
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 07)
"""

import os
import copy
import time
import gzip
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy import sparse


def get_prunable_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """Retrieves all Conv2d and Linear weight tensors (excluding biases and batch norm)."""
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((f"{name}.weight", module.weight))
    return prunable


def apply_global_unstructured_pruning(model: nn.Module, sparsity: float) -> Dict[str, torch.Tensor]:
    """
    Ranks all prunable weights globally by absolute value and zeroes the lowest fraction.
    Returns dictionary of binary masks {param_name: mask_tensor}.
    """
    prunable = get_prunable_parameters(model)
    masks = {}

    if sparsity <= 0.0:
        for name, param in prunable:
            masks[name] = torch.ones_like(param.data)
        return masks

    # Concatenate absolute values across all prunable weights
    all_weights = torch.cat([param.data.abs().flatten() for _, param in prunable])
    k = int(sparsity * all_weights.numel())
    if k >= all_weights.numel():
        k = all_weights.numel() - 1

    threshold = torch.kthvalue(all_weights, k)[0].item()

    for name, param in prunable:
        mask = (param.data.abs() > threshold).float()
        param.data.mul_(mask)
        masks[name] = mask

    return masks


def apply_layerwise_unstructured_pruning(model: nn.Module, sparsity: float) -> Dict[str, torch.Tensor]:
    """
    Ranks weights locally within each layer by absolute value and zeroes the lowest fraction.
    """
    prunable = get_prunable_parameters(model)
    masks = {}

    if sparsity <= 0.0:
        for name, param in prunable:
            masks[name] = torch.ones_like(param.data)
        return masks

    for name, param in prunable:
        w_flat = param.data.abs().flatten()
        k = int(sparsity * w_flat.numel())
        if k >= w_flat.numel():
            k = w_flat.numel() - 1
        
        if k > 0:
            threshold = torch.kthvalue(w_flat, k)[0].item()
            mask = (param.data.abs() > threshold).float()
        else:
            mask = torch.ones_like(param.data)
        
        param.data.mul_(mask)
        masks[name] = mask

    return masks


def apply_structured_filter_pruning(model: nn.Module, filter_sparsity: float) -> Dict[str, torch.Tensor]:
    """
    Structured Pruning: Evaluates L2-norm of each output filter in Conv2d layers.
    Zeroes the lowest filter_sparsity fraction of complete filters.
    """
    masks = {}
    if filter_sparsity <= 0.0:
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                masks[f"{name}.weight"] = torch.ones_like(module.weight.data)
        return masks

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data  # shape: (out_channels, in_channels, kH, kW)
            num_filters = weight.shape[0]
            # Compute L2 norm per filter
            filter_norms = torch.norm(weight.view(num_filters, -1), p=2, dim=1)
            k = int(filter_sparsity * num_filters)
            if k >= num_filters:
                k = num_filters - 1
            
            if k > 0:
                thresh = torch.kthvalue(filter_norms, k)[0].item()
                filter_mask = (filter_norms > thresh).float()  # shape (out_channels,)
            else:
                filter_mask = torch.ones(num_filters, device=weight.device)
            
            # Broadcast mask across (out_channels, 1, 1, 1)
            expanded_mask = filter_mask.view(num_filters, 1, 1, 1).expand_as(weight)
            weight.mul_(expanded_mask)
            masks[f"{name}.weight"] = expanded_mask
            if module.bias is not None:
                module.bias.data.mul_(filter_mask)

    return masks


def fine_tune_masked_model(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 3,
    device: str = 'cpu'
) -> List[float]:
    """
    Fine-tunes non-zero weights while keeping zeroed weights strictly frozen (mask fixed).
    """
    model.train()
    model.to(device)
    loss_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Ensure zeroed weights receive zero gradient
            for name, param in model.named_parameters():
                if name in masks:
                    param.grad.data.mul_(masks[name].to(device))

            optimizer.step()

            # Enforce exact mask post-update
            for name, param in model.named_parameters():
                if name in masks:
                    param.data.mul_(masks[name].to(device))

            running_loss += loss.item()
            batches += 1

        loss_history.append(running_loss / max(1, batches))

    return loss_history


def evaluate_accuracy(model: nn.Module, test_loader: torch.utils.data.DataLoader, device: str = 'cpu') -> float:
    """Evaluates top-1 classification accuracy on test dataset."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def benchmark_inference_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    warmup: int = 25,
    iterations: int = 100,
    device: str = 'cpu'
) -> Dict[str, float]:
    """Measures mean, median, and std inference latency per batch and per sample on CPU."""
    model.eval()
    model.to(device)
    x = sample_input.to(device)
    batch_size = x.size(0)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(x)
            dt = time.perf_counter() - t0
            timings.append(dt * 1000.0)  # milliseconds

    return {
        'batch_latency_mean_ms': float(np.mean(timings)),
        'batch_latency_median_ms': float(np.median(timings)),
        'batch_latency_std_ms': float(np.std(timings)),
        'sample_latency_mean_ms': float(np.mean(timings) / batch_size),
        'throughput_samples_per_sec': float(batch_size / (np.mean(timings) / 1000.0))
    }


def analyze_storage_formats(model: nn.Module, base_dir: str, prefix: str = 'model') -> Dict[str, Any]:
    """
    Compares dense on-disk checkpoint size vs compressed sparse formats (gzip and CSR).
    """
    os.makedirs(base_dir, exist_ok=True)
    dense_path = os.path.join(base_dir, f"{prefix}_dense.pt")
    gzip_path = os.path.join(base_dir, f"{prefix}_dense.pt.gz")

    # 1. Standard dense checkpoint
    torch.save(model.state_dict(), dense_path)
    dense_size_bytes = os.path.getsize(dense_path)

    # 2. Gzip-compressed checkpoint (exploiting zero runs in sparse weights)
    with open(dense_path, 'rb') as f_in, gzip.open(gzip_path, 'wb') as f_out:
        f_out.writelines(f_in)
    gzip_size_bytes = os.path.getsize(gzip_path)

    # 3. Calculate actual weight sparsity
    prunable = get_prunable_parameters(model)
    total_prunable = sum(p.numel() for _, p in prunable)
    zero_prunable = sum((p.data == 0).sum().item() for _, p in prunable)
    actual_sparsity = zero_prunable / max(1, total_prunable)

    return {
        'dense_size_bytes': dense_size_bytes,
        'dense_size_kb': dense_size_bytes / 1024.0,
        'dense_size_mb': dense_size_bytes / (1024.0 * 1024.0),
        'compressed_size_bytes': gzip_size_bytes,
        'compressed_size_kb': gzip_size_bytes / 1024.0,
        'compression_factor': dense_size_bytes / max(1, gzip_size_bytes),
        'actual_sparsity_ratio': actual_sparsity
    }
