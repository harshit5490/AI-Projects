from pathlib import Path
from itertools import product

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
    save_json,
)

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

def load_scaled_data():
    """
    Load scaled dataset.
    """

    X = load_dataframe(
        PROCESSED_DIR /
        "mall_customers_scaled.csv"
    )

    logger.info(
        "Scaled dataset loaded."
    )

    return X

def get_parameter_grids():

    """
    Hyperparameter grids.
    """

    grids = {

        "KMeans": {

            "n_clusters": [4, 5, 6, 7, 8],

            "init": [
                "k-means++",
                "random",
            ],

            "n_init": [10, 20],

            "max_iter": [300, 500],

        },

        "Agglomerative": {

            "n_clusters": [4, 5, 6, 7, 8],

            "linkage": [

                "ward",

                "complete",

                "average",

            ],

        },

        "DBSCAN": {

            "eps": [

                0.3,
                0.5,
                0.7,
                0.9,
                1.1,

            ],

            "min_samples": [

                3,
                5,
                7,
                10,

            ],

        },

    }

    return grids

def evaluate_clustering(
    X,
    labels,
):
    """
    Evaluate clustering performance.
    """

    valid_clusters = set(labels)

    if -1 in valid_clusters:
        valid_clusters.remove(-1)

    if len(valid_clusters) < 2:
        return None

    metrics = {

        "Silhouette Score": silhouette_score(
            X,
            labels,
        ),

        "Davies-Bouldin Index":
        davies_bouldin_score(
            X,
            labels,
        ),

        "Calinski-Harabasz Index":
        calinski_harabasz_score(
            X,
            labels,
        ),

    }

    return metrics

def tune_model(
    model_class,
    parameter_grid,
    X,
):
    """
    Tune a clustering model.
    """

    best_model = None

    best_params = None

    best_metrics = None

    best_score = -1

    keys = list(
        parameter_grid.keys()
    )

    values = list(
        parameter_grid.values()
    )

    combinations = product(*values)

    for combination in combinations:

        params = dict(

            zip(
                keys,
                combination,
            )

        )

        model = model_class(
            **params
        )

        labels = model.fit_predict(
            X
        )

        metrics = evaluate_clustering(
            X,
            labels,
        )

        if metrics is None:
            continue

        if (
            metrics["Silhouette Score"]
            > best_score
        ):

            best_score = metrics[
                "Silhouette Score"
            ]

            best_model = model

            best_params = params

            best_metrics = metrics

    return (

        best_model,

        best_params,

        best_metrics,

    )    

def tune_all_models(
    X,
    grids,
):
    """
    Tune all clustering algorithms.
    """

    tuned_models = {}

    algorithms = {

        "KMeans": KMeans,

        "Agglomerative": AgglomerativeClustering,

        "DBSCAN": DBSCAN,

    }

    for name, model_class in algorithms.items():

        logger.info(
            f"Tuning {name}..."
        )

        (
            model,
            params,
            metrics,
        ) = tune_model(

            model_class,

            grids[name],

            X,

        )

        tuned_models[name] = {

            "Model": model,

            "Parameters": params,

            "Metrics": metrics,

        }

    return tuned_models

def compare_tuned_models(
    tuned_models,
):
    """
    Compare tuned clustering models.
    """

    results = []

    for name, values in tuned_models.items():

        metrics = values["Metrics"]

        if metrics is None:
            continue

        results.append({

            "Model": name,

            "Silhouette Score":
            metrics["Silhouette Score"],

            "Davies-Bouldin Index":
            metrics["Davies-Bouldin Index"],

            "Calinski-Harabasz Index":
            metrics["Calinski-Harabasz Index"],

        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by="Silhouette Score",

        ascending=False,

    )

    save_dataframe(

        results_df,

        REPORTS_DIR /
        "tuned_model_comparison.csv",

    )

    logger.info(
        "Tuned model comparison saved."
    )

    print("\n")
    print("=" * 80)
    print("Tuned Model Comparison")
    print("=" * 80)
    print(results_df)

    return results_df

def save_best_parameters(
    tuned_models,
    results_df,
):
    """
    Save best parameters.
    """

    best_algorithm = results_df.iloc[0]["Model"]

    best_parameters = tuned_models[
        best_algorithm
    ]["Parameters"]

    save_json(

        best_parameters,

        REPORTS_DIR /
        "best_parameters.json",

    )

    logger.info(
        "Best parameters saved."
    )

    return best_algorithm

def save_best_tuned_model(
    tuned_models,
    best_algorithm,
):
    """
    Save best tuned model.
    """

    model = tuned_models[
        best_algorithm
    ]["Model"]

    save_model(

        model,

        MODELS_DIR /
        "best_model.joblib",

    )

    logger.info(
        "Best tuned model saved."
    )

def main():

    X = load_scaled_data()

    grids = get_parameter_grids()

    tuned_models = tune_all_models(

        X,

        grids,

    )

    results_df = compare_tuned_models(

        tuned_models,

    )

    best_algorithm = save_best_parameters(

        tuned_models,

        results_df,

    )

    save_best_tuned_model(

        tuned_models,

        best_algorithm,

    )

    logger.info(
        "Hyperparameter tuning completed successfully."
    )


if __name__ == "__main__":
    main()    