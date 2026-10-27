import logging
import pandas as pd
from sklearn.datasets import fetch_openml

logger = logging.getLogger(__name__)

def load_raw_dataset() -> pd.DataFrame:
    """
    Ingestion Step:
    Fetches raw UCI Adult dataset from OpenML without modification.
    Logs row count (48842) and column count (15).
    """
    logger.info("Starting ingestion step: fetching UCI Adult dataset (version=2)...")
    dataset = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    raw_df = dataset.frame.copy()

    # Standardize target column name if needed
    if "class" in raw_df.columns:
        raw_df = raw_df.rename(columns={"class": "income"})

    row_count, col_count = raw_df.shape
    logger.info(f"Ingestion completed successfully. Row count: {row_count}, Column count: {col_count}")

    # Explicit validation check against expected benchmark
    if row_count != 48842 or col_count != 15:
        logger.warning(f"Unexpected shape! Expected (48842, 15), got ({row_count}, {col_count})")
    else:
        logger.info("Shape check verified: Exactly 48842 rows and 15 columns loaded.")

    return raw_df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = load_raw_dataset()
    print("Ingested dataset sample:")
    print(df.head(3))
