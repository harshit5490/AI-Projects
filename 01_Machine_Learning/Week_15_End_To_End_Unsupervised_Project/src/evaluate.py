from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from utils import (
    get_logger,
    load_dataframe,
    load_model,
    save_dataframe,
    save_plot,
    save_json
)
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGES_DIR = PROJECT_ROOT / "images"

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

def load_best_model():
    """
    Load best clustering model.
    """

    model = load_model(
        MODELS_DIR /
        "best_model.joblib"
    )

    logger.info(
        "Best model loaded."
    )

    return model

def predict_clusters(
    model,
    X,
):
    """
    Predict cluster labels.
    """

    if hasattr(model, "predict"):

        labels = model.predict(
            X
        )

    else:

        labels = model.fit_predict(
            X
        )

    logger.info(
        "Cluster prediction completed."
    )

    return labels

def evaluate_clusters(
    X,
    labels,
):
    """
    Evaluate clustering performance.
    """

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

    print("=" * 60)

    print("Clustering Evaluation")

    print("=" * 60)

    print(
        f"Silhouette Score : {silhouette:.4f}"
    )

    print(
        f"Davies-Bouldin : {davies:.4f}"
    )

    print(
        f"Calinski-Harabasz : {calinski:.4f}"
    )

    metrics = {

        "Silhouette Score": silhouette,

        "Davies-Bouldin Index": davies,

        "Calinski-Harabasz Index": calinski,

    }

    return metrics

def plot_cluster_distribution(
    labels,
):
    """
    Plot cluster distribution.
    """

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    save_dataframe(
        cluster_counts.reset_index().rename(
            columns={
                "index": "Cluster",
                0: "Count",
            }
        ),
        REPORTS_DIR / "cluster_distribution.csv",
    )

    plt.figure(figsize=(8, 5))

    cluster_counts.plot(
        kind="bar"
    )

    plt.title("Cluster Distribution")

    plt.xlabel("Cluster")

    plt.ylabel("Number of Customers")

    save_plot(
        plt,
        IMAGES_DIR / "cluster_distribution.png",
    )

    logger.info(
        "Cluster distribution saved."
    )

def plot_pca_clusters(
    X,
    labels,
):
    """
    PCA visualization.
    """

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8,6))

    plt.scatter(
        X_pca[:,0],
        X_pca[:,1],
        c=labels,
        cmap="tab10",
    )

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title("Customer Segments")

    plt.colorbar()

    save_plot(
        plt,
        IMAGES_DIR / "pca_clusters.png",
    )

    logger.info(
        "PCA visualization saved."
    )

# def save_cluster_centers(
#     model,
#     X,
# ):
#     """
#     Save KMeans cluster centers.
#     """

#     if not hasattr(
#         model,
#         "cluster_centers_",
#     ):

#         logger.info(
#             "Cluster centers unavailable."
#         )

#         return

#     scaler = load_model(
#         MODELS_DIR / "scaler.joblib"
#     )

#     centers = scaler.inverse_transform(
#         model.cluster_centers_
#     )

#     centers_df = pd.DataFrame(

#         centers,

#         columns=X.columns,

#     )

#     save_dataframe(

#         centers_df,

#         REPORTS_DIR /
#         "cluster_centers.csv",

#     )

#     logger.info(
#         "Cluster centers saved."
#     )

def save_cluster_centers(
    model,
    scaler,
    X,
):
    """
    Save cluster centers in original scale.
    """

    if not hasattr(
        model,
        "cluster_centers_",
    ):

        logger.info(
            "Cluster centers unavailable."
        )

        return

    centers = scaler.inverse_transform(
        model.cluster_centers_
    )

    centers_df = pd.DataFrame(

        centers,

        columns=X.columns,

    )

    save_dataframe(

        centers_df,

        REPORTS_DIR /
        "cluster_centers.csv",

    )

    logger.info(
        "Cluster centers saved."
    )

    return centers_df

def save_report(
    metrics,
):
    """
    Save evaluation report.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(

        REPORTS_DIR /
        "evaluation_report.txt",

        "w",

    ) as file:

        file.write(
            "Clustering Evaluation\n"
        )

        file.write("=" * 40)

        file.write("\n\n")

        for key, value in metrics.items():

            file.write(
                f"{key}: {value:.4f}\n"
            )

    logger.info(
        "Evaluation report saved."
    )

def generate_segment_names(
    centers_df,
):
    """
    Generate meaningful business segment names
    from cluster centers.

    Parameters
    ----------
    centers_df : pd.DataFrame
        DataFrame containing cluster centers in
        the original (inverse transformed) scale.

    Returns
    -------
    dict
        Dictionary mapping cluster ID to segment name.
    """

    segments = {}

    for cluster_id, row in centers_df.iterrows():

        age = row["Age"]

        income = row["Annual Income (k$)"]

        spending = row["Spending Score (1-100)"]

        # ----------------------------
        # Age Group
        # ----------------------------
        if age < 30:
            age_group = "Young"

        elif age < 50:
            age_group = "Adult"

        else:
            age_group = "Senior"

        # ----------------------------
        # Income Level
        # ----------------------------
        if income >= 70:
            income_level = "High"

        elif income >= 40:
            income_level = "Medium"

        else:
            income_level = "Low"

        # ----------------------------
        # Spending Level
        # ----------------------------
        if spending >= 70:
            spending_level = "High"

        elif spending >= 40:
            spending_level = "Medium"

        else:
            spending_level = "Low"

        # ----------------------------
        # Business Segment Name
        # ----------------------------
        segment_name = (
            f"{age_group} | "
            f"{income_level} Income | "
            f"{spending_level} Spending"
        )

        segments[str(cluster_id)] = segment_name

    save_json(
        segments,
        REPORTS_DIR / "cluster_segments.json",
    )

    logger.info(
        "Cluster segment names generated."
    )

    return segments


def main():

    X = load_scaled_data()

    model = load_best_model()

    labels = predict_clusters(
        model,
        X,
    )

    metrics = evaluate_clusters(
        X,
        labels,
    )

    plot_cluster_distribution(
        labels,
    )

    plot_pca_clusters(
        X,
        labels,
    )

    scaler = load_model(
        MODELS_DIR / "scaler.joblib"
    )

    centers_df = save_cluster_centers(
        model,
        scaler,
        X,
    )

    generate_segment_names(
        centers_df,
    )

    save_report(
        metrics,
    )

    logger.info(
        "Evaluation completed successfully."
    )


if __name__ == "__main__":
    main()