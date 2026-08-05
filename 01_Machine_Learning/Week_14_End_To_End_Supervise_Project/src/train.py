"""
train.py

This module handles:
1. Loading training data
2. Training multiple ML models
3. Comparing model performance
4. Selecting the best model
"""

from pathlib import Path
from utils import get_logger
import time

import joblib
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

logger = get_logger(__name__)

def load_training_data():
    """
    Load train and test datasets.
    """

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze()

    logger.info("Training data loaded.")

    return X_train, X_test, y_train, y_test

def get_models():
    """
    Return all machine learning models.
    """

    models = {

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            random_state=42
        ),

        "Extra Trees": ExtraTreesClassifier(
            random_state=42
        ),

        "AdaBoost": AdaBoostClassifier(
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),

        "LightGBM": LGBMClassifier(
            random_state=42,
            verbose=-1,
        ),

        "CatBoost": CatBoostClassifier(
            random_state=42,
            verbose=0,
        ),
    }

    return models

def add_stacking_model(models):
    """
    Add stacking classifier.
    """

    estimators = [

        (
            "rf",
            RandomForestClassifier(
                random_state=42
            ),
        ),

        (
            "xgb",
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                verbosity=0,
            ),
        ),

        (
            "lgbm",
            LGBMClassifier(
                random_state=42,
                verbose=-1,
            ),
        ),

    ]

    stacking = StackingClassifier(

        estimators=estimators,

        final_estimator=LogisticRegression(),

        cv=5,

        n_jobs=-1,

    )

    models["Stacking"] = stacking

    return models

def train_model(
    model,
    X_train,
    y_train,
):
    """
    Train model.
    """

    start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_time = (
        time.perf_counter()
        - start
    )

    return model, training_time

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate model.
    """

    start = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - start
    )

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

    report = classification_report(
        y_test,
        y_pred,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        report,
        prediction_time,
    )

def save_model(
    model,
    model_name: str,
    ) -> None:
    """
    Save trained model.
    """

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(
        model,
        MODELS_DIR / filename,
    )

    logger.info(
        f"{filename} saved successfully."
    )

def main():

    X_train, X_test, y_train, y_test = load_training_data()

    models = get_models()

    models = add_stacking_model(models)

    results = []

    best_model = None
    best_model_name = ""
    best_f1 = -1

    for name, model in models.items():

        print("=" * 60)
        print(name)
        print("=" * 60)

        model, training_time = train_model(
            model,
            X_train,
            y_train,
        )

        (
            accuracy,
            precision,
            recall,
            f1,
            report,
            prediction_time,
        ) = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print(report)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print(
            f"Training Time : {training_time:.4f} sec"
        )

        print(
            f"Prediction Time : {prediction_time:.4f} sec"
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Training Time": training_time,
                "Prediction Time": prediction_time,
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

        save_model(model, name)    

    save_model(
        best_model,
        "best_model",
    )

    logger.info(f"Best Model : {best_model_name}")

    logger.info(
        f"Best F1 Score: {best_f1:.4f}"
    )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1 Score",
        ascending=False,
    )

    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)

    print(results_df)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        REPORTS_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)

    print(f"Models Trained : {len(models)}")
    print(f"Best Model     : {best_model_name}")
    print(f"Best F1 Score  : {best_f1:.4f}")

    print("\nTraining completed successfully.")

    return best_model, results_df

if __name__ == "__main__":
    main()        