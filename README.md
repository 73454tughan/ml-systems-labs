# Machine Learning Systems Engineering Portfolio

**Student:** Tughan Bulut Kurtulus  
**Student ID:** 73454  
**Textbook:** *Vijay Janapa Reddi, Introduction to Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems.*  
**Environment:** Python 3.11.8 (Pinned Stack)  

---

## Repository Structure

This repository contains the complete laboratory coursework for the Machine Learning Systems curriculum, structured strictly according to the course specifications:

```
ml-systems-labs/
├── README.md                 # Portfolio overview and navigation
├── requirements.txt          # Strictly pinned dependency versions
├── .gitignore                # Environment and dataset ignore rules
├── lab01/                    # Environment and First System Measurements (Ch. 1, 2, 12)
│   ├── report.md
│   ├── src/
│   └── results/
├── lab02/                    # A Neural Network from Scratch (Ch. 3)
│   ├── report.md
│   ├── src/
│   └── results/
├── lab03/                    # Architectures and Their Costs (Ch. 4)
│   ├── report.md
│   ├── src/
│   └── results/
└── lab04/                    # Data Pipeline and Quality Engineering (Ch. 5-6)
    ├── report.md
    ├── src/
    └── results/
```

---

## Laboratory Index

| Lab | Title | Textbook Reference | Graded Evidence & Artifacts | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Lab 01** | **Environment and First System Measurements** | Ch. 1, 2, 12 | Pinned versions, training time, single-sample latency, memory, deployment fit table | ✅ Complete |
| **Lab 02** | **A Neural Network from Scratch** | Ch. 3 | Matrix calculus chain rule, 10/10 numerical grad check, 6-config grid loss curves | ✅ Complete |
| **Lab 03** | **Architectures and Their Costs** | Ch. 4 | PyTorch MLP, CNN, Attention; hand vs thop FLOPs, latency profiling, Pareto trade-off plots | ✅ Complete |
| **Lab 04** | **Data Pipeline and Quality Engineering** | Ch. 5, 6 | OpenML Adult pipeline, zero-leakage transforms, feature synthesis, duplicate audit, SHA-256 hash | ✅ Complete |

---

## Reproduction Instructions

```bash
# Clone the repository
git clone https://github.com/73454tughan/ml-systems-labs.git
cd ml-systems-labs

# Create and activate isolated environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run individual lab pipelines
python lab01/src/run_benchmarks.py
python lab02/src/train_eval.py
python lab03/src/train_profile.py
python lab04/src/data_pipeline.py
```
