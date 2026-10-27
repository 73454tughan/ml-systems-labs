# Lab 04: Data Pipeline and Quality Engineering

**Student:** Tughan Bulut Kurtulus (ID: 73454)  
**Readings:** Reddi, Chapters 5 and 6  

---

## 1. Goal
The primary objective of this laboratory is to construct a modular, robust, and mathematically **leakage-free** tabular machine learning data pipeline targeting the UCI Adult Census Income dataset (openml.org, ID: dult, version 2). The system incorporates lossless ingestion logging, automated schema and domain bound validation under a zero-silent-drop policy, strict training-partition-only preprocessing parameterization, feature engineering with baseline model verification, a comprehensive duplicate-label noise audit, and cryptographic SHA-256 data tracking for end-to-end reproducibility.

---

## 2. Method
1. **Lossless Ingestion:** Programmatically fetched the Adult dataset (48,842 rows and 15 attributes), recording raw shapes and cryptographic hashes without preliminary row deletion.
2. **Schema & Contract Enforcement:** Enforced strict column data types, validated realistic numeric domains (ge in [17, 90], hours-per-week in [1, 99]), and mapped string missingness tokens ('?') to formal null representations while maintaining a strict zero-silent-drop policy prior to controlled partitioning.
3. **Leakage-Free Preprocessing:**
   - Partitioned data into a 70/30 stratified split (
andom_state=42) based on the binary target (income >50K).
   - Imputation parameters (median for numeric, mode for categorical), outlier clipping percentiles (1st and 99th), standard scaling, and one-hot encoding dictionaries were **learned strictly on the training partition (70%)** and subsequently applied to the test partition (30%).
   - Serialized fitted pipeline to 
esults/preprocessor.joblib.
4. **Feature Engineering:**
   - Derived capital_net = capital-gain - capital-loss to capture aggregate financial wealth accumulation.
   - Derived hours_per_age = hours-per-week / (age - 16) to capture labor intensity relative to working life span.
5. **Baseline Modeling & Evaluation:** Trained a LogisticRegression classifier comparing performance with and without the engineered features, evaluating Accuracy, F1-score (positive class >50K), Precision, and Recall.
6. **Label Quality & Consistency Audit:** Identified exact duplicate feature rows carrying conflicting target labels (label noise) and audited edge-case records.

---

## 3. Results Table or Plot

### Empirical Impact of Feature Engineering on Baseline Model

| Model Evaluation Variant | Test Accuracy | F1-Score (Income >50K) | Precision | Recall | Delta / Gain |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Without Engineered Features)** | **0.8540** | **0.6601** | 0.7448 | 0.5927 | Baseline reference |
| **Enhanced (With capital_net & hours_per_age)** | **0.8578** | **0.6696** | 0.7537 | 0.6024 | **+0.0038 Acc / +0.0095 F1** |

### Data Quality & Pipeline Integrity Audit Summary

| Validation Check | Empirical Finding | Pipeline Handling Action |
| :--- | :--- | :--- |
| **Total Ingested Volume** | 48,842 rows x 15 columns | Losslessly recorded with 0 preliminary drops |
| **Missingness Distribution** | workclass (2,799), occupation (2,809), 
ative-country (857) | Handled via training-set mode imputation |
| **Duplicate Conflicts** | Conflicting duplicate records detected | Logged in audit report without silent deletion |
| **Target Imbalance** | ~75.9% <=50K, ~24.1% >50K | Stratified 70/30 split preserves class ratio |

---

## 4. Three Conclusions in Your Own Words
1. **F1-Score is Essential for Skewed Target Distributions:** With a 76/24 class distribution, accuracy is misleading because a trivial majority-class classifier achieves ~76% accuracy while providing zero predictive utility. The F1-score (0.6696) correctly measures balanced precision and recall on the minority high-income class.
2. **Strict Zero-Leakage Protocol Prevents Optimistic Evaluation:** Deriving missing value statistics and outlier percentiles exclusively from the training split ensures that the model evaluation reflects true generalization performance rather than data leakage contamination.
3. **Domain-Specific Feature Synthesis Delivers Empirical Value:** Introducing compound variables (capital_net and hours_per_age) directly yielded measurable gains in both accuracy (+0.38%) and F1-score (+0.95%) on the standard linear baseline, demonstrating that feature engineering remains highly impactful in tabular data engineering.
