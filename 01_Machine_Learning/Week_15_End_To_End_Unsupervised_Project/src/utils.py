"""
utils.py

Common utility functions used across the project.
"""

from pathlib import Path
import json
import logging

import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs" 

def get_logger(name: str):
    """
    Create project logger.
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


def save_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path,
):
    """
    Save dataframe as CSV.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )


def load_dataframe(
    file_path: Path,
):
    """
    Load CSV file.
    """

    return pd.read_csv(file_path)


def save_model(
    model,
    output_path: Path,
):
    """
    Save ML model.
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
    Load ML model.
    """

    return joblib.load(model_path)


def save_plot(
    plt_object,
    output_path: Path,
):
    """
    Save matplotlib figure.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt_object.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt_object.close()


def save_json(
    data,
    output_path: Path,
):
    """
    Save dictionary as JSON.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def load_json(
    file_path: Path,
):
    """
    Load JSON file.
    """

    with open(
        file_path,
        "r",
    ) as file:

        return json.load(file)