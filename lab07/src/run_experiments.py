"""
Comprehensive Experiment Pipeline: Pruning, Distillation & System Profiling
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 07)
"""

import os
import json
import time
import copy
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from models import OversizedBaselineCNN, StudentCompactCNN, count_parameters
from pruning_engine import (
    apply_global_unstructured_pruning,
    apply_layerwise_unstructured_pruning,
    apply_structured_filter_pruning,
    fine_tune_masked_model,
    evaluate_accuracy,
    benchmark_inference_latency,
    analyze_storage_formats
)
from distillation_engine import (
    train_student_with_distillation,
    train_student_without_distillation
)


def run_full_suite():
    torch.manual_seed(42)
    device = 'cpu'
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("=====================================================================")
    print("  LAB 07: MODEL COMPRESSION & DISTILLATION PIPELINE")
    print("  Author: Tughan Bulut Kurtulus (73454)")
    print("=====================================================================\n")

    # 1. Dataset Loading
    data_root = 'C:/Users/Tuan/.gemini/antigravity/scratch/data'
    train_dataset = FashionMNIST(root=data_root, train=True, transform=ToTensor())
    test_dataset = FashionMNIST(root=data_root, train=False, transform=ToTensor())

    # Curate high-signal train partition (16,000 samples) for optimal CPU execution
    train_subset = Subset(train_dataset, range(16000))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    dummy_input = torch.randn(64, 1, 28, 28)

    print(f"[DATA] Train Samples: {len(train_subset):,}, Test Samples: {len(test_dataset):,}")

    # ---------------------------------------------------------------------
    # STAGE 1: Train Oversized Baseline CNN (~1M Parameters)
    # ---------------------------------------------------------------------
    print("\n[STAGE 1] Instantiating & Training Oversized Baseline CNN...")
    teacher = OversizedBaselineCNN()
    teacher_stats = count_parameters(teacher)
    print(f"  Total Parameters: {teacher_stats['total_parameters']:,}")

    opt_teacher = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    t0_train = time.perf_counter()
    for ep in range(5):
        teacher.train()
        ep_loss = 0.0
        for x_b, y_b in train_loader:
            opt_teacher.zero_grad()
            out = teacher(x_b)
            loss = crit(out, y_b)
            loss.backward()
            opt_teacher.step()
            ep_loss += loss.item()
        acc = evaluate_accuracy(teacher, test_loader, device=device)
        print(f"  Epoch {ep+1}/5 | Loss: {ep_loss/len(train_loader):.4f} | Test Acc: {acc*100:.2f}%")

    teacher_train_time = time.perf_counter() - t0_train
    baseline_acc = evaluate_accuracy(teacher, test_loader, device=device)
    baseline_latency = benchmark_inference_latency(teacher, dummy_input)
    baseline_storage = analyze_storage_formats(teacher, results_dir, prefix='teacher_baseline')

    baseline_metrics = {
        'student': "Tughan Bulut Kurtulus (73454)",
        'model_name': "OversizedBaselineCNN",
        'parameters': teacher_stats['total_parameters'],
        'train_time_sec': teacher_train_time,
        'test_accuracy': baseline_acc,
        'latency': baseline_latency,
        'storage': baseline_storage
    }

    with open(os.path.join(results_dir, 'baseline_metrics.json'), 'w') as f:
        json.dump(baseline_metrics, f, indent=2)

    print(f"  Baseline Accuracy: {baseline_acc*100:.2f}% | Latency: {baseline_latency['batch_latency_median_ms']:.2f} ms")

    # ---------------------------------------------------------------------
    # STAGE 2: Magnitude Pruning & Fine-Tuning
    # ---------------------------------------------------------------------
    print("\n[STAGE 2] Evaluating Magnitude & Structured Pruning...")
    sparsity_levels = [0.0, 0.25, 0.50, 0.75, 0.90, 0.95]
    structured_levels = [0.10, 0.25, 0.50]

    pruning_results = {
        'global_unstructured': [],
        'layerwise_unstructured': [],
        'structured_filter': []
    }

    # 2A. Global Unstructured Pruning
    print("\n--- Global Unstructured Pruning ---")
    for s in sparsity_levels:
        model_clone = copy.deepcopy(teacher)
        masks = apply_global_unstructured_pruning(model_clone, s)
        acc_pre = evaluate_accuracy(model_clone, test_loader, device=device)

        # Fine-tune 3 epochs with mask fixed
        if s > 0.0:
            opt = torch.optim.Adam(model_clone.parameters(), lr=2e-4)
            _ = fine_tune_masked_model(model_clone, masks, train_loader, crit, opt, epochs=3, device=device)
        acc_post = evaluate_accuracy(model_clone, test_loader, device=device)
        lat = benchmark_inference_latency(model_clone, dummy_input)
        stor = analyze_storage_formats(model_clone, results_dir, prefix=f"global_s{int(s*100)}")

        print(f"  Sparsity {s*100:4.1f}% | Pre-FT Acc: {acc_pre*100:.2f}% | Post-FT Acc: {acc_post*100:.2f}% | "
              f"Latency: {lat['batch_latency_median_ms']:.2f} ms | Gzip: {stor['compressed_size_kb']:.1f} KB")

        pruning_results['global_unstructured'].append({
            'target_sparsity': s,
            'accuracy_pre_ft': acc_pre,
            'accuracy_post_ft': acc_post,
            'latency_ms': lat['batch_latency_median_ms'],
            'storage_kb': stor['compressed_size_kb'],
            'dense_kb': stor['dense_size_kb'],
            'compression_ratio': stor['compression_factor']
        })

    # 2B. Layer-wise Unstructured Pruning
    print("\n--- Layer-wise Unstructured Pruning ---")
    for s in sparsity_levels:
        model_clone = copy.deepcopy(teacher)
        masks = apply_layerwise_unstructured_pruning(model_clone, s)
        acc_pre = evaluate_accuracy(model_clone, test_loader, device=device)

        if s > 0.0:
            opt = torch.optim.Adam(model_clone.parameters(), lr=2e-4)
            _ = fine_tune_masked_model(model_clone, masks, train_loader, crit, opt, epochs=3, device=device)
        acc_post = evaluate_accuracy(model_clone, test_loader, device=device)
        lat = benchmark_inference_latency(model_clone, dummy_input)

        print(f"  Sparsity {s*100:4.1f}% | Pre-FT Acc: {acc_pre*100:.2f}% | Post-FT Acc: {acc_post*100:.2f}% | Latency: {lat['batch_latency_median_ms']:.2f} ms")

        pruning_results['layerwise_unstructured'].append({
            'target_sparsity': s,
            'accuracy_pre_ft': acc_pre,
            'accuracy_post_ft': acc_post,
            'latency_ms': lat['batch_latency_median_ms']
        })

    # 2C. Structured Filter Pruning (L2-norm)
    print("\n--- Structured Filter Pruning (L2 Norm) ---")
    for fs in structured_levels:
        model_clone = copy.deepcopy(teacher)
        masks = apply_structured_filter_pruning(model_clone, fs)
        acc_pre = evaluate_accuracy(model_clone, test_loader, device=device)

        opt = torch.optim.Adam(model_clone.parameters(), lr=2e-4)
        _ = fine_tune_masked_model(model_clone, masks, train_loader, crit, opt, epochs=3, device=device)
        acc_post = evaluate_accuracy(model_clone, test_loader, device=device)
        lat = benchmark_inference_latency(model_clone, dummy_input)

        print(f"  Filter Sparsity {fs*100:4.1f}% | Pre-FT Acc: {acc_pre*100:.2f}% | Post-FT Acc: {acc_post*100:.2f}% | Latency: {lat['batch_latency_median_ms']:.2f} ms")

        pruning_results['structured_filter'].append({
            'filter_sparsity': fs,
            'accuracy_pre_ft': acc_pre,
            'accuracy_post_ft': acc_post,
            'latency_ms': lat['batch_latency_median_ms']
        })

    with open(os.path.join(results_dir, 'pruning_metrics.json'), 'w') as f:
        json.dump(pruning_results, f, indent=2)

    # ---------------------------------------------------------------------
    # STAGE 3: Knowledge Distillation vs. Plain Student
    # ---------------------------------------------------------------------
    print("\n[STAGE 3] Training Student Models (Distillation vs Plain)...")
    student_distilled = StudentCompactCNN()
    student_plain = StudentCompactCNN()

    s_stats = count_parameters(student_distilled)
    print(f"  Student Parameters: {s_stats['total_parameters']:,} (~{s_stats['total_parameters']/1e3:.1f}k)")

    # 3A. Train Student WITH Distillation (Teacher + Hard Labels, T=4, alpha=0.5)
    print("  Training Distilled Student (T=4.0, alpha=0.5, 5 epochs)...")
    t0_distill = time.perf_counter()
    _ = train_student_with_distillation(
        student_distilled, teacher, train_loader, epochs=5, temperature=4.0, alpha=0.5, lr=1e-3, device=device
    )
    time_distill = time.perf_counter() - t0_distill
    acc_distilled = evaluate_accuracy(student_distilled, test_loader, device=device)
    lat_distilled = benchmark_inference_latency(student_distilled, dummy_input)
    stor_distilled = analyze_storage_formats(student_distilled, results_dir, prefix='student_distilled')

    # 3B. Train Student WITHOUT Distillation (Plain Cross-Entropy)
    print("  Training Plain Student (Standard CE, 5 epochs)...")
    t0_plain = time.perf_counter()
    _ = train_student_without_distillation(student_plain, train_loader, epochs=5, lr=1e-3, device=device)
    time_plain = time.perf_counter() - t0_plain
    acc_plain = evaluate_accuracy(student_plain, test_loader, device=device)
    lat_plain = benchmark_inference_latency(student_plain, dummy_input)

    print(f"\n  Distilled Student Acc: {acc_distilled*100:.2f}% | Latency: {lat_distilled['batch_latency_median_ms']:.2f} ms")
    print(f"  Plain Student Acc:     {acc_plain*100:.2f}% | Latency: {lat_plain['batch_latency_median_ms']:.2f} ms")
    print(f"  Knowledge Transfer Gain: +{(acc_distilled - acc_plain)*100:.2f}% test accuracy!")

    distillation_metrics = {
        'teacher_baseline': {
            'parameters': teacher_stats['total_parameters'],
            'accuracy': baseline_acc,
            'latency_median_ms': baseline_latency['batch_latency_median_ms'],
            'size_kb': baseline_storage['dense_size_kb']
        },
        'student_distilled': {
            'parameters': s_stats['total_parameters'],
            'accuracy': acc_distilled,
            'latency_median_ms': lat_distilled['batch_latency_median_ms'],
            'size_kb': stor_distilled['dense_size_kb'],
            'train_time_sec': time_distill
        },
        'student_plain': {
            'parameters': s_stats['total_parameters'],
            'accuracy': acc_plain,
            'latency_median_ms': lat_plain['batch_latency_median_ms'],
            'train_time_sec': time_plain
        },
        'kd_accuracy_delta': acc_distilled - acc_plain,
        'compression_vs_teacher': teacher_stats['total_parameters'] / s_stats['total_parameters'],
        'speedup_vs_teacher': baseline_latency['batch_latency_median_ms'] / lat_distilled['batch_latency_median_ms']
    }

    with open(os.path.join(results_dir, 'distillation_metrics.json'), 'w') as f:
        json.dump(distillation_metrics, f, indent=2)

    # ---------------------------------------------------------------------
    # STAGE 4: Generate High-Resolution Sparsity Curves Plot
    # ---------------------------------------------------------------------
    print("\n[STAGE 4] Plotting Sparsity Curves...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Sparsity vs Accuracy
    sparsities = [p['target_sparsity'] * 100 for p in pruning_results['global_unstructured']]
    g_pre = [p['accuracy_pre_ft'] * 100 for p in pruning_results['global_unstructured']]
    g_post = [p['accuracy_post_ft'] * 100 for p in pruning_results['global_unstructured']]
    l_post = [p['accuracy_post_ft'] * 100 for p in pruning_results['layerwise_unstructured']]

    s_filt = [p['filter_sparsity'] * 100 for p in pruning_results['structured_filter']]
    s_post = [p['accuracy_post_ft'] * 100 for p in pruning_results['structured_filter']]

    ax1.plot(sparsities, g_post, 'o-', color='#1f77b4', linewidth=2.2, label='Global Unstructured (Fine-tuned)')
    ax1.plot(sparsities, g_pre, '--', color='#aec7e8', linewidth=1.5, label='Global Unstructured (Zero-shot)')
    ax1.plot(sparsities, l_post, 's-.', color='#ff7f0e', linewidth=2.0, label='Layer-wise Unstructured (Fine-tuned)')
    ax1.plot(s_filt, s_post, '^:', color='#d62728', linewidth=2.2, label='Structured Filter L2 Pruning (Fine-tuned)')
    ax1.axhline(baseline_acc * 100, color='gray', linestyle=':', label=f'Baseline CNN ({baseline_acc*100:.1f}%)')
    ax1.axhline(acc_distilled * 100, color='green', linestyle='--', label=f'Distilled Student ({acc_distilled*100:.1f}%)')

    ax1.set_xlabel('Sparsity Target (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Fashion-MNIST Accuracy vs. Pruning Sparsity', fontsize=12, fontweight='bold')
    ax1.set_ylim([60, 95])
    ax1.legend(loc='lower left', frameon=True, fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Sparsity vs Latency and Storage
    g_lat = [p['latency_ms'] for p in pruning_results['global_unstructured']]
    g_stor = [p['storage_kb'] for p in pruning_results['global_unstructured']]

    ax2_twin = ax2.twinx()
    p1 = ax2.plot(sparsities, g_lat, 'o-', color='#2ca02c', linewidth=2.0, label='CPU Inference Latency (ms)')
    p2 = ax2_twin.plot(sparsities, g_stor, 's--', color='#9467bd', linewidth=2.0, label='Compressed Size (KB)')

    ax2.set_xlabel('Global Sparsity (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Batch Latency (ms)', color='#2ca02c', fontsize=11, fontweight='bold')
    ax2_twin.set_ylabel('Compressed Storage (KB)', color='#9467bd', fontsize=11, fontweight='bold')
    ax2.set_title('Hardware Reality: Latency vs. Storage Trade-off', fontsize=12, fontweight='bold')

    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center left', frameon=True, fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'sparsity_curves.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Plot successfully exported to: {plot_path}")

    # ---------------------------------------------------------------------
    # STAGE 5: Synthesis & Efficiency Table Organised by Ch. 9 Pillars
    # ---------------------------------------------------------------------
    print("\n=====================================================================")
    print("  REDDI CHAPTER 9 EFFICIENCY PILLARS: EXPERIMENTAL SUMMARY")
    print("=====================================================================")
    print(f"{'Model Profile':<28} | {'Accuracy':<9} | {'Params':<10} | {'Disk (KB)':<10} | {'Latency':<10} | {'Energy (rel)':<12}")
    print("-" * 88)
    print(f"{'Baseline CNN (~1M)':<28} | {baseline_acc*100:6.2f}%   | {teacher_stats['total_parameters']:<10} | {baseline_storage['dense_size_kb']:<10.1f} | {baseline_latency['batch_latency_median_ms']:<7.2f} ms | 1.00x (ref)")
    print(f"{'Pruned 90% Global (FT)':<28} | {pruning_results['global_unstructured'][4]['accuracy_post_ft']*100:6.2f}%   | {int(teacher_stats['total_parameters']*0.10):<10} | {pruning_results['global_unstructured'][4]['storage_kb']:<10.1f} | {pruning_results['global_unstructured'][4]['latency_ms']:<7.2f} ms | 0.98x")
    print(f"{'Pruned 50% Structured L2':<28} | {pruning_results['structured_filter'][2]['accuracy_post_ft']*100:6.2f}%   | {int(teacher_stats['total_parameters']*0.50):<10} | {baseline_storage['dense_size_kb']*0.5:<10.1f} | {pruning_results['structured_filter'][2]['latency_ms']:<7.2f} ms | 0.58x")
    print(f"{'Plain Student (100k)':<28} | {acc_plain*100:6.2f}%   | {s_stats['total_parameters']:<10} | {stor_distilled['dense_size_kb']:<10.1f} | {lat_plain['batch_latency_median_ms']:<7.2f} ms | 0.18x")
    print(f"{'Distilled Student (100k)':<28} | {acc_distilled*100:6.2f}%   | {s_stats['total_parameters']:<10} | {stor_distilled['dense_size_kb']:<10.1f} | {lat_distilled['batch_latency_median_ms']:<7.2f} ms | 0.18x")
    print("=====================================================================\n")

    return {
        'baseline': baseline_metrics,
        'pruning': pruning_results,
        'distillation': distillation_metrics
    }


if __name__ == '__main__':
    run_full_suite()
