"""
utils.py

Utility functions used across the project.
"""


import logging
import joblib
import time
import pandas as pd

from pathlib import Path
from functools import wraps

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOG_DIR = PROJECT_ROOT / "logs"

def get_logger(name: str) -> logging.Logger:
    """
    Create and return a project logger.
    """

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_DIR / "project.log"
    )

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger

def save_model(
    model,
    output_path: Path,
) -> None:
    """
    Save trained model.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        output_path,
    )

def load_model(
    model_path: Path,
):
    """
    Load trained model.
    """

    return joblib.load(
        model_path
    )

def save_dataframe(
    df,
    output_path: Path,
):
    """
    Save DataFrame.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

def load_dataframe(
    file_path: Path,
):
    """
    Load CSV file.
    """

    return pd.read_csv(
        file_path
    )

def timer(func):
    """
    Measure execution time.
    """

    @wraps(func)

    def wrapper(*args, **kwargs):

        start = time.perf_counter()

        result = func(
            *args,
            **kwargs,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        return result, elapsed

    return wrapper        

def save_plot(
    plt,
    output_path: Path,
):
    """
    Save matplotlib figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close()