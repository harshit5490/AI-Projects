"""
data_preprocessing.py

This module handles:
1. Loading raw data
2. Data validation
3. Missing value checking
4. Duplicate checking
5. Removing unnecessary columns
6. Encoding categorical features
7. Saving cleaned dataset
"""

from pathlib import Path

import pandas as pd

from utils import (
    get_logger,
    load_dataframe,
    save_dataframe,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Mall_Customers.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mall_customers_cleaned.csv"
)

def load_data():
    """
    Load raw dataset.
    """

    df = load_dataframe(
        RAW_DATA_PATH
    )

    logger.info(
        "Dataset loaded successfully."
    )

    logger.info(
        f"Shape : {df.shape}"
    )

    return df

def check_missing_values(df):
    """
    Check missing values.
    """

    missing = df.isnull().sum()

    logger.info(
        "\nMissing Values"
    )

    logger.info(f"\n{missing}")

def check_duplicates(df):
    """
    Check duplicate rows.
    """

    duplicates = df.duplicated().sum()

    logger.info(
        f"\nDuplicate Rows : {duplicates}"
    )

def remove_customer_id(df):
    """
    Remove CustomerID column.
    """

    df = df.drop(
        columns=["CustomerID"]
    )

    logger.info(
        "CustomerID removed."
    )

    return df

def encode_gender(df):
    """
    Encode Gender column.
    """

    mapping = {

        "Male": 1,

        "Female": 0,

    }

    df["Gender"] = df[
        "Gender"
    ].map(mapping)

    logger.info(
        "Gender encoded."
    )

    return df

def save_cleaned_dataset(df):
    """
    Save cleaned dataset.
    """

    save_dataframe(
        df,
        PROCESSED_DATA_PATH,
    )

    logger.info(
        "Cleaned dataset saved."
    )

def main():

    df = load_data()

    check_missing_values(df)

    check_duplicates(df)

    df = remove_customer_id(df)

    df = encode_gender(df)

    logger.info(
        f"\nFinal Shape : {df.shape}"
    )

    logger.info(
        f"\nColumns\n{list(df.columns)}"
    )

    save_cleaned_dataset(df)

    logger.info(
        "Data preprocessing completed successfully."
    )


if __name__ == "__main__":
    main()            