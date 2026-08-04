"""
evaluate.py

Evaluate the best trained model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score
)

from utils import (
    load_model,
    load_dataframe,
    save_dataframe,
    save_plot,
    get_logger,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGES_DIR = PROJECT_ROOT / "images"

logger = get_logger(__name__)

def load_test_data():

    X_test = load_dataframe(
        PROCESSED_DIR / "X_test.csv"
    )

    y_test = (
        load_dataframe(
            PROCESSED_DIR / "y_test.csv"
        )
        .squeeze()
    )

    logger.info("Test dataset loaded.")

    return X_test, y_test

def load_best_model():

    model = load_model(
        MODELS_DIR / "best_model.joblib"
    )

    logger.info("Best model loaded.")

    return model

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the best model.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
    )

    recall = recall_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
    )
    probability = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probability,
    )

    average_precision = average_precision_score(
        y_test,
        probability,
    )

    report = classification_report(
        y_test,
        y_pred,
    )

    print(report)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print(f"Average Precision : {average_precision:.4f}")


    logger.info("Evaluation completed.")

    return (
    y_pred,
    report,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    average_precision,)

def save_classification_report(
    report,
):
    """
    Save classification report.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORTS_DIR /
        "classification_report.txt",
        "w",
    ) as f:

        f.write(report)

    logger.info(
        "Classification report saved."
    )

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
):

    plt.figure(figsize=(8, 6))
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("Confusion Matrix")

    save_plot(
        plt,
        IMAGES_DIR /
        "confusion_matrix.png",
    )
    logger.info(
        "Confusion matrix saved."
    )

def plot_roc_curve(
    model,
    X_test,
    y_test,
):
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title("ROC Curve")

    save_plot(
        plt,
        IMAGES_DIR /
        "roc_curve.png",
    )
    logger.info(
    "ROC curve saved."
    )

def plot_pr_curve(
    model,
    X_test,
    y_test,
):
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    plt.title(
        "Precision Recall Curve"
    )

    save_plot(
        plt,
        IMAGES_DIR /
        "precision_recall_curve.png",
    )

    logger.info(
    "Precision-Recall curve saved."
    )

def save_feature_importance(
    model,
    X_test,
):

    if not hasattr(
        model,
        "feature_importances_",
    ):

        logger.warning(
            "Feature importance unavailable."
        )

        return

    importance = pd.DataFrame({

        "Feature": X_test.columns,

        "Importance":
        model.feature_importances_,

    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False,
    )

    save_dataframe(

        importance,

        REPORTS_DIR /
        "feature_importance.csv",

    )

    plt.figure(figsize=(10,6))

    plt.barh(

        importance["Feature"][:15],

        importance["Importance"][:15],

    )

    plt.gca().invert_yaxis()

    plt.title("Top 15 Feature Importances")

    save_plot(

        plt,

        IMAGES_DIR /
        "feature_importance.png",

    )

    logger.info(
        "Feature importance saved."
    )

def save_evaluation_metrics(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
    average_precision: float,
    ) -> None:
    """
    Save evaluation metrics.
    """

    metrics = pd.DataFrame(
        [
            {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "ROC-AUC": roc_auc,
                "Average Precision": average_precision,
            }
        ]
    )

    save_dataframe(
        metrics,
        REPORTS_DIR / "evaluation_metrics.csv",
    )

    logger.info(
        "Evaluation metrics saved."
    )

def main():

    X_test, y_test = load_test_data()

    model = load_best_model()

    (
    y_pred,
    report,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    average_precision,
    ) = evaluate_model(
    model,
    X_test,
    y_test,
    )

    save_classification_report(
        report,
    )

    save_evaluation_metrics(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    average_precision,
    )

    plot_confusion_matrix(
        model,
        X_test,
        y_test,
    )

    plot_roc_curve(
        model,
        X_test,
        y_test,
    )

    plot_pr_curve(
        model,
        X_test,
        y_test,
    )

    save_feature_importance(
        model,
        X_test,
    )

    logger.info(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    main()                    