"""
clustering.py

This module handles:

1. Training clustering algorithms
2. Evaluating clustering performance
3. Comparing clustering models
4. Saving trained models
5. Saving cluster labels
"""

from pathlib import Path

import pandas as pd

from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from utils import (
    get_logger,
    load_dataframe,
    save_dataframe,
    save_model,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

def load_scaled_data():
    """
    Load scaled feature matrix.
    """

    X = load_dataframe(
        PROCESSED_DIR /
        "mall_customers_scaled.csv"
    )

    logger.info(
        "Scaled dataset loaded."
    )

    logger.info(
        f"Shape : {X.shape}"
    )

    return X

def get_models():
    """
    Create clustering models.
    """

    models = {

        "KMeans": KMeans(
            n_clusters=6,
            random_state=42,
            n_init=10,
        ),

        "Agglomerative": AgglomerativeClustering(
            n_clusters=6,
        ),

        "DBSCAN": DBSCAN(
            eps=0.8,
            min_samples=5,
        ),

    }

    return models

def train_model(
    model,
    X,
):
    """
    Train clustering model.
    """

    labels = model.fit_predict(X)

    return labels

def evaluate_model(
    X,
    labels,
):
    """
    Evaluate clustering model.
    """

    unique_clusters = len(
        set(labels)
    )

    valid_clusters = set(labels)

    if -1 in valid_clusters:
        valid_clusters.remove(-1)

    if len(valid_clusters) < 2:

        logger.warning(
            "Less than two valid clusters."
        )

        return None

    silhouette = silhouette_score(
        X,
        labels,
    )

    davies = davies_bouldin_score(
        X,
        labels,
    )

    calinski = calinski_harabasz_score(
        X,
        labels,
    )

    metrics = {
        "Number of Clusters": len(valid_clusters),

        "Silhouette Score": silhouette,

        "Davies-Bouldin Index": davies,

        "Calinski-Harabasz Index": calinski,

    }

    return metrics

def save_cluster_labels(
    X,
    labels,
    model_name,
):
    """
    Save cluster labels.
    """

    clustered_df = X.copy()

    clustered_df["Cluster"] = labels

    save_dataframe(

        clustered_df,

        REPORTS_DIR
        / f"{model_name.lower()}_clusters.csv",

    )

    logger.info(
        f"{model_name} cluster labels saved."
    )

def save_trained_model(
    model,
    model_name,
):
    """
    Save clustering model.
    """

    filename = (
        model_name.lower()
        + ".joblib"
    )

    save_model(

        model,

        MODELS_DIR / filename,

    )

    logger.info(
        f"{model_name} saved."
    )

def compare_models(results):
    """
    Compare clustering models.
    """

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="Silhouette Score",
        ascending=False,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dataframe(
        results_df,
        REPORTS_DIR / "model_comparison.csv",
    )

    logger.info("Model comparison saved.")

    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(results_df)

    return results_df

def save_best_model(
    best_model,
):
    """
    Save best clustering model.
    """

    save_model(
        best_model,
        MODELS_DIR / "best_model.joblib",
    )

    logger.info(
        "Best model saved."
    )


def main():

    X = load_scaled_data()

    models = get_models()

    results = []

    best_model = None
    best_score = -1

    for model_name, model in models.items():

        print("=" * 70)
        print(model_name)
        print("=" * 70)

        labels = train_model(
            model,
            X,
        )

        metrics = evaluate_model(
            X,
            labels,
        )

        if metrics is None:

            logger.warning(
                f"{model_name} skipped."
            )

            continue

        print(metrics)

        save_trained_model(
            model,
            model_name,
        )

        save_cluster_labels(
            X,
            labels,
            model_name,
        )

        results.append({

            "Model": model_name,

            "Silhouette Score":
            metrics["Silhouette Score"],

            "Davies-Bouldin Index":
            metrics["Davies-Bouldin Index"],

            "Calinski-Harabasz Index":
            metrics["Calinski-Harabasz Index"],

        })

        if (
            metrics["Silhouette Score"]
            > best_score
        ):

            best_score = metrics[
                "Silhouette Score"
            ]

            best_model = model

    compare_models(results)

    save_best_model(best_model)

    logger.info(
        "Clustering completed successfully."
    )


if __name__ == "__main__":
    main()            