import logging
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

logger = logging.getLogger(__name__)

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Feature Engineering
    - capital_net = capital-gain - capital-loss
    - hours_per_age = hours-per-week / (age - 16)
    """
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    # Guard against division by zero (though min age is 17 in schema)
    effective_age_denominator = (out["age"] - 16).replace(0, 1.0)
    out["hours_per_age"] = out["hours-per-week"] / effective_age_denominator
    return out

def evaluate_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    num_cols: list,
    cat_cols: list,
    random_state: int = 42,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Trains and compares LogisticRegression baseline WITH and WITHOUT derived features.
    Records accuracy and F1 score.
    """
    logger.info("Evaluating LogisticRegression baseline without derived features...")

    # 1. Pipeline WITHOUT derived features
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    X_tr_cat = ohe.fit_transform(X_train[cat_cols])
    X_te_cat = ohe.transform(X_test[cat_cols])

    X_tr_num = scaler.fit_transform(X_train[num_cols])
    X_te_num = scaler.transform(X_test[num_cols])

    X_tr_base = np.hstack([X_tr_num, X_tr_cat])
    X_te_base = np.hstack([X_te_num, X_te_cat])

    clf_base = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_base.fit(X_tr_base, y_train)
    y_pred_base = clf_base.predict(X_te_base)

    metrics_without = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_base)), 4),
        "f1": round(float(f1_score(y_test, y_pred_base)), 4),
        "precision": round(float(precision_score(y_test, y_pred_base)), 4),
        "recall": round(float(recall_score(y_test, y_pred_base)), 4)
    }
    logger.info(f"Baseline WITHOUT features: Accuracy={metrics_without['accuracy']}, F1={metrics_without['f1']}")

    # 2. Pipeline WITH derived features
    logger.info("Generating derived features (capital_net, hours_per_age)...")
    X_train_eng = add_derived_features(X_train)
    X_test_eng = add_derived_features(X_test)

    num_cols_eng = num_cols + ["capital_net", "hours_per_age"]
    scaler_eng = StandardScaler()

    X_tr_num_eng = scaler_eng.fit_transform(X_train_eng[num_cols_eng])
    X_te_num_eng = scaler_eng.transform(X_test_eng[num_cols_eng])

    X_tr_with = np.hstack([X_tr_num_eng, X_tr_cat])
    X_te_with = np.hstack([X_te_num_eng, X_te_cat])

    clf_eng = LogisticRegression(max_iter=1000, random_state=random_state)
    clf_eng.fit(X_tr_with, y_train)
    y_pred_with = clf_eng.predict(X_te_with)

    metrics_with = {
        "accuracy": round(float(accuracy_score(y_test, y_pred_with)), 4),
        "f1": round(float(f1_score(y_test, y_pred_with)), 4),
        "precision": round(float(precision_score(y_test, y_pred_with)), 4),
        "recall": round(float(recall_score(y_test, y_pred_with)), 4)
    }
    logger.info(f"Baseline WITH features:    Accuracy={metrics_with['accuracy']}, F1={metrics_with['f1']}")

    diff_acc = round(metrics_with["accuracy"] - metrics_without["accuracy"], 4)
    diff_f1 = round(metrics_with["f1"] - metrics_without["f1"], 4)

    results = {
        "without_derived_features": metrics_without,
        "with_derived_features": metrics_with,
        "delta": {
            "accuracy_change": diff_acc,
            "f1_change": diff_f1
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved evaluation metrics to: {metrics_path}")

    return results
