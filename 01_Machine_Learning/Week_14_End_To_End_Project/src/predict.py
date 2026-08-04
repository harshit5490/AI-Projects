"""
predict.py

Predict customer churn for new customer data.
"""

from pathlib import Path

import pandas as pd

from utils import (
    load_model,
    get_logger,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

logger = get_logger(__name__)

def load_prediction_objects():
    """
    Load trained model and scaler.
    """

    model = load_model(
        MODELS_DIR / "best_model.joblib"
    )

    scaler = load_model(
        MODELS_DIR / "scaler.joblib"
    )

    logger.info(
        "Model and scaler loaded."
    )

    return model, scaler

def create_sample_customer():
    """
    Create sample customer.
    """

    customer = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 12,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "MonthlyCharges": 75.50,
        "TotalCharges": 900.25,

        "MultipleLines_No phone service": 0,
        "MultipleLines_Yes": 1,

        "InternetService_Fiber optic": 1,
        "InternetService_No": 0,

        "OnlineSecurity_No internet service": 0,
        "OnlineSecurity_Yes": 0,

        "OnlineBackup_No internet service": 0,
        "OnlineBackup_Yes": 1,

        "DeviceProtection_No internet service": 0,
        "DeviceProtection_Yes": 1,

        "TechSupport_No internet service": 0,
        "TechSupport_Yes": 0,

        "StreamingTV_No internet service": 0,
        "StreamingTV_Yes": 1,

        "StreamingMovies_No internet service": 0,
        "StreamingMovies_Yes": 1,

        "Contract_One year": 0,
        "Contract_Two year": 0,

        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": 1,
        "PaymentMethod_Mailed check": 0,
    }

    return pd.DataFrame([customer])

def preprocess_customer(
    customer,
    scaler,
):
    """
    Scale numerical features.
    """

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    customer = customer.copy()

    customer[numerical_columns] = scaler.transform(
        customer[numerical_columns]
    )

    return customer

def predict_customer(
    model,
    customer,
):
    """
    Predict customer churn.
    """

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    return prediction, probability

def display_prediction(
    prediction,
    probability,
):
    """
    Display prediction.
    """

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)

    print(
        f"Prediction : {'Churn' if prediction == 1 else 'No Churn'}"
    )

    print(
        f"Probability: {probability:.2%}"
    )

def main():

    model, scaler = load_prediction_objects()

    customer = create_sample_customer()

    customer = preprocess_customer(
        customer,
        scaler,
    )

    prediction, probability = predict_customer(
        model,
        customer,
    )

    display_prediction(
        prediction,
        probability,
    )

    logger.info(
        "Prediction completed successfully."
    )


if __name__ == "__main__":
    main()    