"""
data_preprocessing.py

This module handles:
1. Loading raw data
2. Cleaning data
3. Handling missing values
4. Encoding categorical features
5. Saving processed data
"""

from pathlib import Path
from utils import get_logger
import pandas as pd

logger = get_logger(__name__)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_cleaned.csv"
)

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    df = pd.read_csv(file_path)

    logger.info("Dataset loaded successfully.")
    logger.info(f"Shape: {df.shape}")

    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns.
    """

    df = df.drop(columns=["customerID"])

    logger.info("Dropped customerID column.")

    return df

def convert_total_charges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert TotalCharges from string to numeric.

    Invalid values are converted to NaN.
    """

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    logger.info(
        "Converted TotalCharges to numeric."
    )

    return df

def missing_value_report(df: pd.DataFrame) -> None:
    """
    Display missing value summary.
    """

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if missing.empty:
        logger.info("No missing values found.")
    else:
        logger.info(
            f"\nMissing Values:\n{missing}"
        )

def inspect_missing_values(df: pd.DataFrame) -> None:
    """
    Display rows containing missing values.
    """

    missing_rows = df[df["TotalCharges"].isnull()]

    logger.info(
        "Displaying rows with missing TotalCharges."
    )

    print(missing_rows)

def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing missing values.
    """

    rows_before = df.shape[0]

    df = df.dropna()

    rows_after = df.shape[0]

    logger.info(
        f"Missing values removed. "
        f"Rows removed: {rows_before - rows_after}"
    )

    logger.info(
        f"Remaining rows: {rows_after}"
    )

    return df    

def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary categorical features.
    """

    binary_mappings = {
        "gender": {
            "Female": 0,
            "Male": 1,
        },
        "Partner": {
            "No": 0,
            "Yes": 1,
        },
        "Dependents": {
            "No": 0,
            "Yes": 1,
        },
        "PhoneService": {
            "No": 0,
            "Yes": 1,
        },
        "PaperlessBilling": {
            "No": 0,
            "Yes": 1,
        },
        "Churn": {
            "No": 0,
            "Yes": 1,
        },
    }

    for column, mapping in binary_mappings.items():
        df[column] = df[column].map(mapping)

    logger.info("Binary features encoded.")
    return df

def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encode multi-class categorical features.
    """

    multi_class_columns = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]

    df = pd.get_dummies(
        df,
        columns=multi_class_columns,
        drop_first=True,
        dtype=int,
    )

    logger.info("One-Hot Encoding completed.")

    return df

def save_processed_data(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save processed dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        f"Processed dataset saved: {output_path}"
    )

def main():

    df = load_data(RAW_DATA_PATH)

    df = drop_columns(df)

    df = convert_total_charges(df)

    print("\nMissing Value Report")
    print("-" * 40)

    missing_value_report(df)

    inspect_missing_values(df)

    df = remove_missing_values(df)

    df = encode_binary_features(df)

    df = one_hot_encode(df)

    print("\nFinal Dataset Shape")
    print(df.shape)

    print("\nFirst Five Rows")
    print(df.head())

    print("\nData Types")
    print(df.info())

    save_processed_data(
        df,
        PROCESSED_DATA_PATH,
    )
    logger.info(
        "Data preprocessing completed successfully."
    )


if __name__ == "__main__":
    main()        