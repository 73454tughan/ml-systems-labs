# Lab 01: Environment and First System Measurements

**Student:** Tughan Bulut Kurtulus (ID: 73454)  
**Readings:** Reddi, Chapters 1, 2, and 12  

---

## 1. Goal
The primary objective of this laboratory is to establish an isolated, reproducible Machine Learning Systems engineering environment with strictly pinned dependencies, train baseline classification models on the Breast Cancer Wisconsin diagnostic dataset using stratified evaluation, rigorously measure systems costs (training duration, single-sample inference latency, serialized model footprint, and resident memory usage), and quantitatively evaluate deployment suitability across Cloud, Edge, Mobile, and TinyML hardware resource tiers.

---

## 2. Method
1. **Isolated Environment & Dependency Verification:** Configured an isolated Python 3.11 environment verifying pinned library versions (`numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`, `torchvision`, `torchinfo`, `thop`, `psutil`, `joblib`), persisting diagnostic environment metadata to `results/versions.txt`.
2. **Reproducible Modeling:** Ingested the Breast Cancer Wisconsin dataset (569 samples, 30 real-valued features) via `load_breast_cancer(return_X_y=True)`. Applied a 70/30 stratified train/test split pinned with `random_state=42` (398 training samples, 171 test samples).
3. **Model Selection:**
   - Linear Baseline: `LogisticRegression(max_iter=1000, random_state=42)`
   - Tree Ensemble: `RandomForestClassifier(n_estimators=100, random_state=42)`
4. **Repeatable Systems Benchmarking:**
   - Training latency: Executed 1 warm-up iteration followed by 5 benchmark runs; recorded the median duration.
   - Inference latency: Executed 100 single-sample inference passes on an isolated test vector; recorded the median execution time.
   - Storage footprint: Serialized models using `joblib.dump` to `results/model.joblib` and measured exact byte and kilobyte size.
   - Memory consumption: Profiled operating system Resident Set Size (RSS) peak memory using `psutil.Process().memory_info().rss`.
5. **Deployment Suitability Assessment:** Compared empirical model footprint and latency metrics against standard operational budgets (Cloud, Edge, Mobile, TinyML).

---

## 3. Results Table or Plot

### Empirical Performance and Systems Cost Summary

| Metric / Dimension | Linear Baseline (`LogisticRegression`) | Tree Ensemble (`RandomForestClassifier`) | Measurement Protocol |
| :--- | :---: | :---: | :--- |
| **Test Accuracy** | **0.9415** (94.15%) | **0.9357** (93.57%) | 70/30 Stratified (`random_state=42`) |
| **Median Training Time** | **136.37 ms** | **176.82 ms** | Median of 5 runs after 1 warm-up |
| **Single-Sample Latency** | **0.1736 ms** | **7.5053 ms** | Median of 100 single-sample inferences |
| **Serialized Model Size** | **1103 bytes** (1.08 KB) | **290889 bytes** (284.07 KB) | `joblib.dump` on disk |
| **Peak Resident Memory (RSS)** | **~430.4 MB** | **~430.4 MB** | `psutil.Process().memory_info().rss` |

### Deployment Suitability Matrix Against Resource Budgets

| Hardware Tier | Memory Budget | Latency Budget | Storage Budget | `LogisticRegression` Fit | `RandomForestClassifier` Fit | Rationale & Trade-offs |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cloud** | >= 1 GB | <= 100 ms | <= 500 MB | **Eligible** | **Eligible** | Both models easily operate within cloud limits; negligible infrastructure overhead. |
| **Edge Gateway** | 256-1024 MB | <= 50 ms | <= 50 MB | **Eligible** | **Eligible** | Low latency (0.174 ms vs 7.505 ms) permits real-time edge processing. |
| **Mobile / Embedded** | 64-256 MB | <= 20 ms | <= 10 MB | **Eligible** | **Eligible** | Both models consume < 284.1 KB disk and run well below the 20 ms budget. |
| **TinyML (Microcontroller)** | <= 256 KB | <= 10 ms | <= 100 KB | **Eligible** (1.1 KB) | **Ineligible** (284.1 KB) | `LogisticRegression` strictly satisfies the 100 KB limit; `RandomForest` exceeds storage. |

---

## 4. Three Conclusions in Your Own Words
1. **Linear Simplicity Enables Extreme Edge & TinyML Deployment:** `LogisticRegression` achieves higher test accuracy (94.15%) on this tabular diagnostic dataset than the ensemble model while maintaining an ultra-compact binary footprint of only 1.08 KB. This makes it uniquely qualified for zero-dependency deployment on memory-constrained TinyML microcontrollers (< 256 KB RAM).
2. **Ensemble Overhead Scales Memory Without Accuracy Gain:** The 100-tree `RandomForestClassifier` introduces a substantial storage penalty (284.1 KB vs 1.08 KB) and requires higher per-sample inference latency due to traversing hundreds of decision tree branches, demonstrating that architectural complexity must be justified by empirical performance gains.
3. **Median Measurement Filters Operating System Jitter:** Using isolated warm-up runs combined with median filtering across 100 single-sample inferences effectively neutralized operating system context-switching noise, providing deterministic latency measurements essential for real-world service-level objective (SLO) capacity planning.
