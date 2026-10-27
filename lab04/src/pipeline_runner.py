import logging
import hashlib
import os
import yaml
import pandas as pd
from src.ingestion import load_raw_dataset
from src.validation import validate_dataset, generate_validation_markdown_report
from src.transformation import prepare_and_transform_data, NUMERICAL_COLS, CATEGORICAL_COLS
from src.feature_engineering import evaluate_baseline_models
from src.label_audit import perform_label_quality_audit

logger = logging.getLogger(__name__)

def compute_dataset_sha256(df: pd.DataFrame) -> str:
    """Computes deterministic SHA-256 hash of processed DataFrame."""
    # Convert dataframe to deterministic CSV byte stream
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()

def execute_pipeline(config_path: str = "config/config.yaml"):
    """
    Orchestrates the complete end-to-end data pipeline:
    1. Ingestion
    2. Validation
    3. Cleaning & Transformation (Leakage-Free 70/30)
    4. Feature Engineering & Evaluation
    5. Label-Quality Audit
    6. SHA-256 Hashing & Config persistence
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("================================================================")
    logger.info("  Starting Data Pipeline & Quality Engineering - Tughan (73454) ")
    logger.info("================================================================")

    # Load initial config
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config = {
        "dataset_name": "adult",
        "openml_version": 2,
        "random_seed": 42,
        "train_split_ratio": 0.70,
        "test_split_ratio": 0.30,
        "target_column": "income",
        "features": {
            "numerical": NUMERICAL_COLS,
            "categorical": CATEGORICAL_COLS
        },
        "engineered_features": ["capital_net", "hours_per_age"],
        "pipeline_version": "1.0.0"
    }

    # Step 1 & 2: Ingestion
    raw_df = load_raw_dataset()

    # Step 3: Validation
    validation_summary = validate_dataset(raw_df)
    generate_validation_markdown_report(validation_summary, "reports/validation_report.md")

    # Step 4: Cleaning & Transformation (Leakage-Free)
    X_train_clean, X_test_clean, y_train, y_test, transformer = prepare_and_transform_data(
        raw_df, random_state=config["random_seed"], output_dir="results"
    )

    # Step 5: Feature Engineering & Baseline Comparison
    evaluation_metrics = evaluate_baseline_models(
        X_train_clean, X_test_clean, y_train, y_test,
        num_cols=NUMERICAL_COLS,
        cat_cols=CATEGORICAL_COLS,
        random_state=config["random_seed"],
        output_dir="results"
    )

    # Step 6: Label Quality & Duplicate Audit
    audit_summary = perform_label_quality_audit(
        raw_df, random_state=config["random_seed"], output_path="reports/label_quality_report.md"
    )

    # Step 7: SHA-256 Hashing of processed dataset
    processed_train = pd.concat([X_train_clean, y_train], axis=1)
    train_hash = compute_dataset_sha256(processed_train)
    test_processed = pd.concat([X_test_clean, y_test], axis=1)
    test_hash = compute_dataset_sha256(test_processed)

    logger.info(f"Processed Train SHA-256: {train_hash}")
    logger.info(f"Processed Test SHA-256:  {test_hash}")

    config["data_hashes"] = {
        "processed_train_sha256": train_hash,
        "processed_test_sha256": test_hash
    }
    config["metrics"] = evaluation_metrics

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Updated configuration saved with SHA-256 hashes: {config_path}")

    logger.info("================================================================")
    logger.info("  Pipeline Executed Successfully! All Graded Artifacts Ready.   ")
    logger.info("================================================================")
    return config
