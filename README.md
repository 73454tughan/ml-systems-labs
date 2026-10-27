# Machine Learning Systems Architecture & Hardware-Aware Optimization

**Author:** Tughan Bulut Kurtulus  
**Student ID:** 73454  
**Track:** Systems & Software Engineering, Vistula University  
**Academic Year:** 2025/2026  
**Curriculum Reference:** *Machine Learning Systems* (Vijay Janapa Reddi, Harvard University)  

---

## 1. Executive Overview

This portfolio documents four hands-on systems investigations addressing the computational boundaries of deep learning and data engineering:

```
├── lab01/  # Microsecond latency benchmarking & TinyML deployment tier classification
├── lab02/  # First-principles tensor backpropagation with numerical finite-difference verification
├── lab03/  # FLOPs derivation, profiler analysis (thop), and Pareto efficiency frontiers
└── lab04/  # Production-grade tabular pipeline with strict zero-leakage enforcement
```

## 2. Laboratory Modules & Key Systems Telemetry

| Module | Core Investigation | Systems Method | Key Findings |
| :--- | :--- | :--- | :--- |
| **Lab 01** | Baseline Profiling | High-resolution wall-clock timer & memory profiler | Logistic Regression fits in 1.05 KB Flash; 0.027 ms latency enables sub-milliwatt TinyML. |
| **Lab 02** | NumPy Neural Engine | Analytical calculus vs two-sided finite difference | Relative error < 1e-7; 2.1x backward/forward workload ratio confirms GEMM doubling. |
| **Lab 03** | Architectural Cost | Theoretical FLOP derivations + `thop` MAC profiling | CNN delivers supreme parameter efficiency (89.88% acc, 20.4k weights vs MLP 235k). |
| **Lab 04** | Data Quality Pipeline | Zero-silent-drop schema validation + training-isolated scalers | Feature synthesis (`capital_net`, `hours_per_age`) expands linear decision surface. |

## 3. Environment Replication

```bash
# Set up isolated virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install strictly pinned dependencies
pip install -r requirements.txt
```
