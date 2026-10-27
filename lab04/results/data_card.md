# Dataset Card: UCI Adult Census Income

**Maintainer / Author:** Tughan Bulut Kurtulus (Student ID: 73454)  
**Task Type:** Tabular Binary Classification (`income > 50K`)  
**Standard Identifier:** OpenML Dataset `adult` (Version 2)  

---

## 1. Dataset Description & Provenance
* **Source:** Extracted from the 1994 United States Census Bureau database by Ronny Kohavi and Barry Becker (Data Mining and Visualization, Silicon Graphics).
* **OpenML Endpoint:** `fetch_openml(name='adult', version=2, as_frame=True)`.
* **Primary Task:** Predict whether individual gross income exceeds $50,000/year based on demographic, employment, and educational indicators.

---

## 2. Dataset Size & Structural Dimensions
* **Total Instances:** 48,842 rows (combining original 32,561 training and 16,281 test splits into the standard full cohort).
* **Total Variables:** 15 columns (14 predictive features + 1 target variable `income`).
* **Splits:**
  * **Train Set (70%):** 34,189 rows initially, reduced to 34,165 rows after strict deduplication on training data only.
  * **Test Set (30%):** 14,653 rows preserved untouched (zero rows dropped to avoid evaluation bias).
* **Feature Schema:**
  * 6 Continuous Numerical Features (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`).
  * 8 Categorical Features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
  * 2 Engineered Numerical Features (`capital_net`, `hours_per_age`).

---

## 3. License & Intellectual Property
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Citation Policy:** Public domain census-derived research dataset originally published in AAAI-96 Workshop on Knowledge Discovery in Databases.

---

## 4. Known Biases & Demographic Disparities
1. **Historical Gender Disparity:** Survey conducted in 1994 reflects historical male-dominated participation in high-income employment brackets (~30% of men earned >$50K vs only ~11% of women in this sample).
2. **Geographical Skew:** Over 89.7% of surveyed individuals list `United-States` as their country of origin, leaving international immigrants sparsely represented across minority subgroups.
3. **Severe Capital Distribution Skew:** The vast majority of survey respondents (>91%) report exactly $0 for both `capital-gain` and `capital-loss`. A distinct artificial cap is present at $99,999 for capital gains.
4. **Class Imbalance:** Significant skew towards the negative class (~75.9% `<=50K` vs ~24.1% `>50K`).

---

## 5. Privacy & Ethical Considerations
* **Personally Identifiable Information (PII):** Direct identifiers (e.g., social security numbers, full names, precise residential addresses) were stripped by the Census Bureau prior to public dissemination.
* **Sampling Weight (`fnlwgt`):** The `fnlwgt` variable represents the estimated number of individuals in the US population having matching demographic characteristics, preventing 1-to-1 re-identification of survey participants.
* **Fairness & Sensitive Attributes:** Protected attributes such as `race` and `sex` are directly present in the feature space. Any production deployment must implement algorithmic fairness constraints (e.g., demographic parity or equalized odds) to avoid perpetuating historical socioeconomic inequities.

---

## 6. Cryptographic Verification & Pipeline Reproducibility
* **Preprocessing Pipeline:** `results/preprocessor.joblib`
* **Configuration:** `config/config.yaml`
* **Train Set Hash (SHA-256):** Embedded in `config/config.yaml`
* **Execution Command:** `python run_pipeline.py`
