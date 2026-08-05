"""
feature_engineering.py

This module handles:
1. Loading cleaned dataset
2. Feature selection
3. Feature scaling
4. Saving scaler
5. Saving processed dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import (
    get_logger,
    load_dataframe,
    save_dataframe,
    save_model
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

CLEANED_DATA_PATH = (
    PROCESSED_DIR
    / "mall_customers_cleaned.csv"
)

SCALED_DATA_PATH = (
    PROCESSED_DIR
    / "mall_customers_scaled.csv"
)

def load_cleaned_data():
    """
    Load cleaned dataset.
    """

    df = load_dataframe(
        CLEANED_DATA_PATH
    )

    logger.info(
        "Cleaned dataset loaded."
    )

    logger.info(
        f"Shape : {df.shape}"
    )

    return df

def select_features(df):
    """
    Select features for clustering.
    """

    feature_columns = [

        "Age",

        "Annual Income (k$)",

        "Spending Score (1-100)",

    ]

    X = df[feature_columns]

    logger.info(
        "Features selected."
    )

    logger.info(
        f"Features : {feature_columns}"
    )

    return X

def scale_features(X):
    """
    Scale selected features.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    logger.info(
        "Feature scaling completed."
    )

    return X_scaled, scaler

def save_scaler(scaler):
    """
    Save fitted scaler.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model(
        scaler,
        MODELS_DIR / "scaler.joblib"
    )

    logger.info(
        "Scaler saved."
    )

def save_scaled_dataset(
    X_scaled,
    columns,
):
    """
    Save scaled dataset.
    """

    scaled_df = pd.DataFrame(
        X_scaled,
        columns=columns,
    )

    save_dataframe(
        scaled_df,
        SCALED_DATA_PATH,
    )

    logger.info(
        "Scaled dataset saved."
    )    


def main():

    df = load_cleaned_data()

    X = select_features(df)

    X_scaled, scaler = scale_features(X)

    save_scaler(scaler)

    save_scaled_dataset(
        X_scaled,
        X.columns,
    )

    logger.info(
        "Feature engineering completed successfully."
    )


if __name__ == "__main__":
    main()    