import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

NUMERICAL_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL_COLS = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

class LeakageFreeTransformer:
    """
    Leakage-Free Preprocessor:
    Fits all imputation values (median for numeric, mode for categorical)
    and outlier boundaries (1st and 99th percentiles) STRICTLY on training data.
    """
    def __init__(self, num_cols: List[str], cat_cols: List[str]):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.medians: Dict[str, float] = {}
        self.modes: Dict[str, str] = {}
        self.clipping_bounds: Dict[str, Tuple[float, float]] = {}
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame):
        X = X_train.copy()
        # Replace '?' with np.nan for clean calculation
        for col in self.cat_cols:
            cleaned = X[col].astype(str).str.strip().replace("?", np.nan)
            mode_series = cleaned.dropna().mode()
            self.modes[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

        for col in self.num_cols:
            cleaned = pd.to_numeric(X[col], errors="coerce")
            self.medians[col] = float(cleaned.median())
            p01 = float(cleaned.quantile(0.01))
            p99 = float(cleaned.quantile(0.99))
            self.clipping_bounds[col] = (p01, p99)

        self.is_fitted = True
        logger.info(f"Fitted LeakageFreeTransformer on {len(X)} training samples.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Transformer must be fitted before calling transform!")
        
        out = df.copy()

        # 1. Impute categorical with training mode
        for col in self.cat_cols:
            out[col] = out[col].astype(str).str.strip().replace("?", np.nan)
            out[col] = out[col].fillna(self.modes[col])

        # 2. Impute numeric with training median & clip to training 1st/99th percentiles
        for col in self.num_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(self.medians[col])
            p_low, p_high = self.clipping_bounds[col]
            out[col] = out[col].clip(lower=p_low, upper=p_high)

        return out

def prepare_and_transform_data(
    df: pd.DataFrame, 
    random_state: int = 42,
    output_dir: str = "results"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LeakageFreeTransformer]:
    """
    Executes Step 4 requirements:
    1. Split train/test 70/30 stratified before cleaning.
    2. Drop exact duplicates from train only (keep test untouched).
    3. Fit transformer on train only (mode for cat, median for num, 1st/99th clipping).
    4. Apply transformation to train and test.
    5. Save fitted transformer with joblib.dump to results/preprocessor.joblib.
    """
    logger.info("Initiating Step 4: Stratified Train/Test Split & Leakage-Free Preprocessing...")

    # Standardize target to binary 0/1
    target_series = df["income"].astype(str).str.strip()
    y = (target_series.isin([">50K", ">50K."])).astype(int)
    X = df.drop(columns=["income"])

    # 1. Stratified 70/30 train/test split BEFORE cleaning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    logger.info(f"Split completed: Train={len(X_train)} (70%), Test={len(X_test)} (30%).")

    # 2. Drop exact duplicates from TRAIN ONLY
    train_combined = pd.concat([X_train, y_train], axis=1)
    duplicates_count = train_combined.duplicated().sum()
    train_dedup = train_combined.drop_duplicates()
    
    y_train = train_dedup["income"]
    X_train = train_dedup.drop(columns=["income"])
    logger.info(f"Dropped {duplicates_count} exact duplicates from TRAIN. Test set remained completely untouched.")

    # 3. Fit transformer on train only
    transformer = LeakageFreeTransformer(NUMERICAL_COLS, CATEGORICAL_COLS)
    transformer.fit(X_train)

    # 4. Transform train and test
    X_train_clean = transformer.transform(X_train)
    X_test_clean = transformer.transform(X_test)

    # 5. Save fitted transformer
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(transformer, save_path)
    logger.info(f"Fitted transformer successfully serialized to: {save_path}")

    return X_train_clean, X_test_clean, y_train, y_test, transformer
