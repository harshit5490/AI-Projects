"""
predict_cluster.py

Predict customer cluster using the trained clustering model.
"""

from pathlib import Path

import pandas as pd

from utils import (
    get_logger,
    load_model,
    load_json
)
logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

def load_scaler():
    """
    Load saved scaler.
    """

    scaler = load_model(
        MODELS_DIR / "scaler.joblib"
    )

    logger.info("Scaler loaded.")

    return scaler

def load_best_model():
    """
    Load best clustering model.
    """

    model = load_model(
        MODELS_DIR / "best_model.joblib"
    )

    logger.info("Best model loaded.")

    return model

def get_customer_data():
    """
    Sample customer.
    """

    customer = pd.DataFrame({

        "Age": [28],

        "Annual Income (k$)": [75],

        "Spending Score (1-100)": [82],

    })

    return customer

def preprocess_customer(
    customer,
    scaler,
):
    """
    Scale customer features.
    """

    scaled = scaler.transform(
        customer
    )

    return scaled

def predict_cluster(
    model,
    customer,
):
    """
    Predict customer cluster.
    """

    cluster = model.predict(
        customer
    )[0]

    logger.info(
        f"Predicted Cluster : {cluster}"
    )

    return cluster

# def get_segment_name(cluster):
#     """
#     Convert cluster ID into business name.
#     """

#     segments = {

#         0: "Budget Customers",

#         1: "Premium Customers",

#         2: "High Income - Low Spending",

#         3: "High Income - High Spending",

#         4: "Average Customers",

#         5: "Low Income - High Spending",

#     }

#     return segments.get(
#         cluster,
#         "Unknown Segment",
#     )

def get_segment_name(cluster):
    """
    Load generated business segment names.
    """

    segments = load_json(
        REPORTS_DIR /
        "cluster_segments.json"
    )

    return segments.get(
        str(cluster),
        "Unknown Segment",
    )

def get_recommendation(segment):
    """
    Generate recommendation based on segment.
    """

    recommendation = []

    if "High Income" in segment:
        recommendation.append(
            "Offer premium products."
        )

    if "Low Income" in segment:
        recommendation.append(
            "Provide discounts and budget-friendly offers."
        )

    if "High Spending" in segment:
        recommendation.append(
            "Recommend loyalty rewards and VIP membership."
        )

    if "Low Spending" in segment:
        recommendation.append(
            "Send personalized promotions to increase engagement."
        )

    if "Young" in segment:
        recommendation.append(
            "Promote trendy and lifestyle products."
        )

    if "Senior" in segment:
        recommendation.append(
            "Recommend essential and value-based products."
        )

    if not recommendation:

        recommendation.append(
            "General marketing campaign."
        )

    return "\n".join(recommendation)


def display_result(
    cluster,
    segment,
    recommendation,
):
    """
    Display prediction result.
    """

    print("=" * 60)
    print("Customer Segmentation Result")
    print("=" * 60)

    print(f"Predicted Cluster : {cluster}")

    print(f"Segment : {segment}")

    print()

    print("Recommendation")

    print(recommendation)


def main():

    scaler = load_scaler()

    model = load_best_model()

    customer = get_customer_data()

    customer = preprocess_customer(
        customer,
        scaler,
    )

    cluster = predict_cluster(
        model,
        customer,
    )

    segment = get_segment_name(
        cluster,
    )

    recommendation = get_recommendation(
        segment,
    )

    display_result(

        cluster,

        segment,

        recommendation,

    )

    logger.info(
        "Prediction completed successfully."
    )


if __name__ == "__main__":
    main()