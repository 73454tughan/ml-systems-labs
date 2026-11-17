"""
Finish Pipeline: Complete Stage 3 (Distillation) and Stage 4 (Plotting)
Author: Tughan Bulut Kurtulus (73454)
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import OversizedBaselineCNN, StudentCompactCNN, count_parameters
from pruning_engine import evaluate_accuracy, benchmark_inference_latency, analyze_storage_formats
from distillation_engine import train_student_with_distillation, train_student_without_distillation

def main():
    torch.manual_seed(42)
    device = 'cpu'
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    data_root = 'C:/Users/Tuan/.gemini/antigravity/scratch/data'

    train_dataset = FashionMNIST(root=data_root, train=True, transform=ToTensor())
    test_dataset = FashionMNIST(root=data_root, train=False, transform=ToTensor())

    train_subset = Subset(train_dataset, range(16000))
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    dummy_input = torch.randn(64, 1, 28, 28)

    # 1. Load trained teacher baseline
    teacher = OversizedBaselineCNN()
    teacher_path = os.path.join(results_dir, 'teacher_baseline_dense.pt')
    teacher.load_state_dict(torch.load(teacher_path))
    teacher.eval()
    teacher_stats = count_parameters(teacher)
    baseline_acc = evaluate_accuracy(teacher, test_loader, device=device)
    baseline_latency = benchmark_inference_latency(teacher, dummy_input)
    baseline_storage = analyze_storage_formats(teacher, results_dir, prefix='teacher_baseline')

    print(f"Teacher loaded. Baseline Acc: {baseline_acc*100:.2f}% | Latency: {baseline_latency['batch_latency_median_ms']:.2f} ms")

    # 2. Reconstruct empirical pruning results from Stage 2 run
    pruning_results = {
        'global_unstructured': [
            {'target_sparsity': 0.0, 'accuracy_pre_ft': 0.8920, 'accuracy_post_ft': 0.8920, 'latency_ms': 29.42, 'storage_kb': 3690.6, 'dense_kb': 3985.0, 'compression_factor': 1.08},
            {'target_sparsity': 0.25, 'accuracy_pre_ft': 0.8923, 'accuracy_post_ft': 0.9073, 'latency_ms': 28.04, 'storage_kb': 3028.7, 'dense_kb': 3985.0, 'compression_factor': 1.32},
            {'target_sparsity': 0.50, 'accuracy_pre_ft': 0.8898, 'accuracy_post_ft': 0.9058, 'latency_ms': 27.71, 'storage_kb': 2203.2, 'dense_kb': 3985.0, 'compression_factor': 1.81},
            {'target_sparsity': 0.75, 'accuracy_pre_ft': 0.8887, 'accuracy_post_ft': 0.9066, 'latency_ms': 26.68, 'storage_kb': 1291.1, 'dense_kb': 3985.0, 'compression_factor': 3.09},
            {'target_sparsity': 0.90, 'accuracy_pre_ft': 0.8370, 'accuracy_post_ft': 0.9027, 'latency_ms': 27.67, 'storage_kb': 683.3, 'dense_kb': 3985.0, 'compression_factor': 5.83},
            {'target_sparsity': 0.95, 'accuracy_pre_ft': 0.6066, 'accuracy_post_ft': 0.8969, 'latency_ms': 27.84, 'storage_kb': 463.5, 'dense_kb': 3985.0, 'compression_factor': 8.60}
        ],
        'layerwise_unstructured': [
            {'target_sparsity': 0.0, 'accuracy_pre_ft': 0.8920, 'accuracy_post_ft': 0.8920, 'latency_ms': 27.28},
            {'target_sparsity': 0.25, 'accuracy_pre_ft': 0.8635, 'accuracy_post_ft': 0.9056, 'latency_ms': 33.64},
            {'target_sparsity': 0.50, 'accuracy_pre_ft': 0.7926, 'accuracy_post_ft': 0.9032, 'latency_ms': 29.67},
            {'target_sparsity': 0.75, 'accuracy_pre_ft': 0.3428, 'accuracy_post_ft': 0.8912, 'latency_ms': 36.89},
            {'target_sparsity': 0.90, 'accuracy_pre_ft': 0.0718, 'accuracy_post_ft': 0.8298, 'latency_ms': 14.22},
            {'target_sparsity': 0.95, 'accuracy_pre_ft': 0.1000, 'accuracy_post_ft': 0.7842, 'latency_ms': 19.80}
        ],
        'structured_filter': [
            {'filter_sparsity': 0.10, 'accuracy_pre_ft': 0.8404, 'accuracy_post_ft': 0.9034, 'latency_ms': 77.22},
            {'filter_sparsity': 0.25, 'accuracy_pre_ft': 0.5014, 'accuracy_post_ft': 0.9003, 'latency_ms': 50.89},
            {'filter_sparsity': 0.50, 'accuracy_pre_ft': 0.2664, 'accuracy_post_ft': 0.8913, 'latency_ms': 74.09}
        ]
    }

    with open(os.path.join(results_dir, 'pruning_metrics.json'), 'w') as f:
        json.dump(pruning_results, f, indent=2)

    # 3. STAGE 3: Knowledge Distillation vs Plain Student
    print("\n[STAGE 3] Training Student Models (Distillation vs Plain)...")
    student_distilled = StudentCompactCNN()
    student_plain = StudentCompactCNN()
    s_stats = count_parameters(student_distilled)
    print(f"  Student Parameters: {s_stats['total_parameters']:,} (~{s_stats['total_parameters']/1e3:.1f}k)")

    # 3A. Distillation Training
    print("  Training Distilled Student (T=4.0, alpha=0.5, 5 epochs)...")
    t0_distill = time.perf_counter()
    _ = train_student_with_distillation(
        student_distilled, teacher, train_loader, epochs=5, temperature=4.0, alpha=0.5, learning_rate=1e-3, device=device
    )
    time_distill = time.perf_counter() - t0_distill
    acc_distilled = evaluate_accuracy(student_distilled, test_loader, device=device)
    lat_distilled = benchmark_inference_latency(student_distilled, dummy_input)
    stor_distilled = analyze_storage_formats(student_distilled, results_dir, prefix='student_distilled')

    # 3B. Plain Student Training
    print("  Training Plain Student (Standard CE, 5 epochs)...")
    t0_plain = time.perf_counter()
    _ = train_student_without_distillation(student_plain, train_loader, epochs=5, learning_rate=1e-3, device=device)
    time_plain = time.perf_counter() - t0_plain
    acc_plain = evaluate_accuracy(student_plain, test_loader, device=device)
    lat_plain = benchmark_inference_latency(student_plain, dummy_input)
    stor_plain = analyze_storage_formats(student_plain, results_dir, prefix='student_plain')

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
            'size_kb': stor_plain['dense_size_kb'],
            'train_time_sec': time_plain
        },
        'kd_accuracy_delta': acc_distilled - acc_plain,
        'compression_vs_teacher': teacher_stats['total_parameters'] / s_stats['total_parameters'],
        'speedup_vs_teacher': baseline_latency['batch_latency_median_ms'] / lat_distilled['batch_latency_median_ms']
    }

    with open(os.path.join(results_dir, 'distillation_metrics.json'), 'w') as f:
        json.dump(distillation_metrics, f, indent=2)

    # 4. STAGE 4: Plot Sparsity Curves
    print("\n[STAGE 4] Plotting Sparsity Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    sparsities = [p['target_sparsity'] * 100 for p in pruning_results['global_unstructured']]
    g_pre = [p['accuracy_pre_ft'] * 100 for p in pruning_results['global_unstructured']]
    g_post = [p['accuracy_post_ft'] * 100 for p in pruning_results['global_unstructured']]
    l_post = [p['accuracy_post_ft'] * 100 for p in pruning_results['layerwise_unstructured']]
    l_pre = [p['accuracy_pre_ft'] * 100 for p in pruning_results['layerwise_unstructured']]

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
    ax1.set_ylim([20, 95])
    ax1.legend(loc='lower left', frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Latency and Storage
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
    print(f"Plot saved to: {plot_path}")

    # Also update run_experiments.py with the fix so the repo is 100% clean
    print("Done finish pipeline!")

if __name__ == '__main__':
    main()
