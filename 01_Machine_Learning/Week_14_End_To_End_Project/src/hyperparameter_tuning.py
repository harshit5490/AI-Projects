"""
hyperparameter_tuning.py

This module handles:
1. Loading datasets
2. Hyperparameter tuning
3. Comparing baseline and tuned models
4. Saving tuned model
"""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from utils import (
    load_dataframe,
    load_model,
    save_dataframe,
    save_model,
    get_logger,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

logger = get_logger(__name__)

def load_training_data():
    """
    Load training and testing datasets.
    """

    X_train = load_dataframe(
        PROCESSED_DIR / "X_train.csv"
    )

    X_test = load_dataframe(
        PROCESSED_DIR / "X_test.csv"
    )

    y_train = (
        load_dataframe(
            PROCESSED_DIR / "y_train.csv"
        )
        .squeeze()
    )

    y_test = (
        load_dataframe(
            PROCESSED_DIR / "y_test.csv"
        )
        .squeeze()
    )

    logger.info("Training datasets loaded.")

    return X_train, X_test, y_train, y_test

def load_baseline_model():
    """
    Load the baseline Gradient Boosting model.
    """

    model = load_model(
        MODELS_DIR / "gradient_boosting.joblib"
    )

    logger.info(
        "Baseline Gradient Boosting model loaded."
    )

    return model

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate a trained model.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred,
        ),
    }

    return metrics

PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

def get_cross_validator():
    """
    Return Stratified K-Fold cross validator.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    logger.info("Stratified K-Fold initialized.")

    return cv

def tune_model(
    X_train,
    y_train,
):
    """
    Tune Gradient Boosting model.
    """

    cv = get_cross_validator()

    grid_search = GridSearchCV(

        estimator=GradientBoostingClassifier(
            random_state=42,
        ),

        param_grid=PARAM_GRID,

        scoring="f1",

        cv=cv,

        n_jobs=-1,

        verbose=2,

    )

    logger.info(
        "Hyperparameter tuning started."
    )

    grid_search.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Hyperparameter tuning completed."
    )

    return grid_search

def compare_models(
    baseline_metrics,
    tuned_metrics,
):
    """
    Compare baseline and tuned model.
    """

    comparison = pd.DataFrame(
        [
            {
                "Model": "Baseline",
                **baseline_metrics,
            },
            {
                "Model": "Tuned",
                **tuned_metrics,
            },
        ]
    )

    comparison = comparison.round(4)

    print("\nModel Comparison")
    print(comparison)

    save_dataframe(
        comparison,
        REPORTS_DIR / "tuned_model_metrics.csv",
    )

    logger.info(
        "Model comparison saved."
    )

    return comparison

def save_best_parameters(
    grid_search,
):
    """
    Save best hyperparameters.
    """

    parameters = pd.DataFrame(
        [grid_search.best_params_]
    )

    parameters["Best CV Score"] = (
        grid_search.best_score_
    )

    save_dataframe(
        parameters,
        REPORTS_DIR / "hyperparameter_results.csv",
    )

    logger.info(
        "Best parameters saved."
    )

def save_best_model(
    baseline_metrics,
    tuned_metrics,
    tuned_model,
):
    save_model(
        tuned_model,
        MODELS_DIR / "tuned_model.joblib",
    )

    if tuned_metrics["F1 Score"] > baseline_metrics["F1 Score"]:

        save_model(
            tuned_model,
            MODELS_DIR / "best_model.joblib",
        )

        logger.info("Best model updated.")

    else:

        logger.info("Baseline model retained.")

        
def main():

    X_train, X_test, y_train, y_test = load_training_data()

    baseline_model = load_baseline_model()

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    grid_search = tune_model(
        X_train,
        y_train,
    )

    tuned_model = grid_search.best_estimator_

    tuned_metrics = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )

    compare_models(
        baseline_metrics,
        tuned_metrics,
    )

    save_best_parameters(
        grid_search,
    )

    save_best_model(
        baseline_metrics,
        tuned_metrics,
        tuned_model,
    )

    logger.info(
        "Hyperparameter tuning completed successfully."
    )


if __name__ == "__main__":
    main()            