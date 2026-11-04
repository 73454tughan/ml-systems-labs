"""
Gradient Verification Suite: Custom Scalar Engine vs. PyTorch Autograd & Numerical FD
Author: Tughan Bulut Kurtulus (Student ID: 73454)
Course: Machine Learning Systems (Lab 05)
"""

import json
import os
import torch
from engine import ScalarNode


def forward_scalar(a_vals, b_vals, c_vals):
    # Vectorized elementwise computation followed by sum reduction
    nodes = []
    for a_v, b_v, c_v in zip(a_vals, b_vals, c_vals):
        term = ((a_v * b_v + c_v) ** 2).relu()
        nodes.append(term)
    
    total = nodes[0]
    for n in nodes[1:]:
        total = total + n
    return total


def numerical_gradient(func, inputs, idx, param_name, eps=1e-7):
    # Two-sided central finite difference approximation
    params_plus = [v.val for v in inputs[param_name]]
    params_minus = [v.val for v in inputs[param_name]]
    
    params_plus[idx] += eps
    params_minus[idx] -= eps

    def eval_with_params(a_raw, b_raw, c_raw):
        nodes = []
        for a_v, b_v, c_v in zip(a_raw, b_raw, c_raw):
            t = ((a_v * b_v + c_v) ** 2)
            nodes.append(max(0.0, t))
        return sum(nodes)

    curr_a = [v.val for v in inputs['a']]
    curr_b = [v.val for v in inputs['b']]
    curr_c = [v.val for v in inputs['c']]

    if param_name == 'a':
        f_plus = eval_with_params(params_plus, curr_b, curr_c)
        f_minus = eval_with_params(params_minus, curr_b, curr_c)
    elif param_name == 'b':
        f_plus = eval_with_params(curr_a, params_plus, curr_c)
        f_minus = eval_with_params(curr_a, params_minus, curr_c)
    else:
        f_plus = eval_with_params(curr_a, curr_b, params_plus)
        f_minus = eval_with_params(curr_a, curr_b, params_minus)

    return (f_plus - f_minus) / (2.0 * eps)


def run_verification():
    # Concrete test values covering positive, negative, and mixed regions
    raw_a = [1.5, -2.0, 0.75, -0.5]
    raw_b = [-3.0, 4.0, 1.25, 2.5]
    raw_c = [2.0, 1.5, -3.5, 0.2]

    # 1. Custom Autograd Engine Pass
    a_nodes = [ScalarNode(x, tag=f'a_{i}') for i, x in enumerate(raw_a)]
    b_nodes = [ScalarNode(x, tag=f'b_{i}') for i, x in enumerate(raw_b)]
    c_nodes = [ScalarNode(x, tag=f'c_{i}') for i, x in enumerate(raw_c)]

    total_loss = forward_scalar(a_nodes, b_nodes, c_nodes)
    total_loss.backward()

    custom_grads = {
        'a': [node.grad for node in a_nodes],
        'b': [node.grad for node in b_nodes],
        'c': [node.grad for node in c_nodes],
        'loss': total_loss.val
    }

    # 2. PyTorch Autograd Pass
    t_a = torch.tensor(raw_a, dtype=torch.float64, requires_grad=True)
    t_b = torch.tensor(raw_b, dtype=torch.float64, requires_grad=True)
    t_c = torch.tensor(raw_c, dtype=torch.float64, requires_grad=True)

    t_loss = (((t_a * t_b + t_c) ** 2).relu()).sum()
    t_loss.backward()

    torch_grads = {
        'a': t_a.grad.tolist(),
        'b': t_b.grad.tolist(),
        'c': t_c.grad.tolist(),
        'loss': t_loss.item()
    }

    # 3. Numerical Central Difference Gradients
    inputs_dict = {'a': a_nodes, 'b': b_nodes, 'c': c_nodes}
    num_grads = {'a': [], 'b': [], 'c': []}
    for p in ['a', 'b', 'c']:
        for i in range(len(raw_a)):
            ng = numerical_gradient(None, inputs_dict, i, p)
            num_grads[p].append(ng)

    # 4. Error Analysis
    max_torch_diff = 0.0
    max_num_diff = 0.0
    diff_table = []

    for p in ['a', 'b', 'c']:
        for i in range(len(raw_a)):
            cg = custom_grads[p][i]
            tg = torch_grads[p][i]
            ng = num_grads[p][i]

            t_err = abs(cg - tg)
            n_err = abs(cg - ng)

            max_torch_diff = max(max_torch_diff, t_err)
            max_num_diff = max(max_num_diff, n_err)

            diff_table.append({
                'param': f"{p}[{i}]",
                'custom_grad': cg,
                'torch_grad': tg,
                'numerical_grad': ng,
                'torch_abs_diff': t_err,
                'num_abs_diff': n_err
            })

    results = {
        'student': "Tughan Bulut Kurtulus (73454)",
        'target_expression': "((a * b + c) ** 2).relu().sum()",
        'forward_loss_custom': custom_grads['loss'],
        'forward_loss_torch': torch_grads['loss'],
        'forward_abs_diff': abs(custom_grads['loss'] - torch_grads['loss']),
        'max_abs_torch_diff': max_torch_diff,
        'max_abs_numerical_diff': max_num_diff,
        'tolerance_threshold': 1e-6,
        'verification_passed': bool(max_torch_diff < 1e-6 and max_num_diff < 1e-5),
        'comparison_table': diff_table
    }

    os.makedirs('results', exist_ok=True)
    with open('results/verification_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[VERIFICATION] Max PyTorch absolute difference: {max_torch_diff:.2e}")
    print(f"[VERIFICATION] Max Numerical absolute difference: {max_num_diff:.2e}")
    print(f"[VERIFICATION] Status: {'PASSED (Threshold < 1e-6 satisfied)' if results['verification_passed'] else 'FAILED'}")
    return results


if __name__ == '__main__':
    run_verification()
