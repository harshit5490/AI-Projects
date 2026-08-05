"""
feature_engineering.py

This module handles:
1. Loading processed data
2. Splitting features and target
3. Train-test split
4. Feature scaling
5. Saving prepared datasets
"""

from pathlib import Path
from utils import get_logger

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_cleaned.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def load_processed_data(file_path: Path) -> pd.DataFrame:
    """
    Load cleaned dataset.
    """

    df = pd.read_csv(file_path)

    logger.info("Processed dataset loaded.")
    print(f"Shape : {df.shape}")

    return df

def split_features_target(
    df: pd.DataFrame,
):
    """
    Separate features and target.
    """

    X = df.drop(columns=["Churn"])

    y = df["Churn"]

    print("Features Shape :", X.shape)
    print("Target Shape :", y.shape)

    return X, y

def split_train_test(X, y):
    """
    Train-test split.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    logger.info("Train-Test Split Completed")

    print("X_train :", X_train.shape)
    print("X_test  :", X_test.shape)

    print("y_train :", y_train.shape)
    print("y_test  :", y_test.shape)

    return X_train, X_test, y_train, y_test

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Scale numerical features.
    """

    scaler = StandardScaler()

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    logger.info("Feature Scaling Completed")

    return X_train, X_test, scaler

def save_scaler(scaler):
    """
    Save trained scaler.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.joblib",
    )

    logger.info("Scaler saved.")

def save_datasets(
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Save train and test datasets.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)

    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    logger.info("Prepared datasets saved.")

def main():

    df = load_processed_data(PROCESSED_DATA_PATH)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test,
    )

    save_scaler(scaler)

    save_datasets(
        X_train,
        X_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()        