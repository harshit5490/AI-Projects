import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV,RandomizedSearchCV,learning_curve,validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

# ==========================
# Load Dataset
# ==========================
BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR/"datasets"/"WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ==========================
# Data Cleaning
# ==========================

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df.dropna(inplace=True)

df.drop("customerID", axis=1, inplace=True)

# ==========================
# Label Encoding
# ==========================

le = LabelEncoder()

label_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]

for col in label_columns:
    df[col] = le.fit_transform(df[col])

# ==========================
# One-Hot Encoding
# ==========================

one_hot_columns = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]

df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

# ==========================
# Features & Target
# ==========================

X = df.drop("Churn", axis=1)

y = df["Churn"]

# ==========================
# Gradient Boosting
# ==========================

gb = GradientBoostingClassifier(random_state=42)

scores = cross_val_score(
    estimator=gb,
    X=X,
    y=y,
    cv=5,
    scoring="accuracy"
)

# print("Fold Scores :", scores)
# print("Mean Accuracy :", scores.mean())
# print("Standard Deviation :", scores.std())


# ------------------StratifiedKFold----------------------
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    estimator=gb,
    X=X,
    y=y,
    cv=skf,
    scoring="accuracy"
)

# print("Fold Scores:", scores)
# print("Mean Accuracy:", scores.mean())
# print("Standard Deviation:", scores.std())

# ------------GridSearchCV-------------------------

param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4]
}

# grid_search = GridSearchCV(
#     estimator=gb,
#     param_grid=param_grid,
#     cv=skf,
#     scoring="accuracy",
#     n_jobs=-1
# )
# grid_search.fit(X, y)
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)
# print("Best Model:", grid_search.best_estimator_)

# -------------------RandomizedSearchCV-----------------

param_distributions = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4]
}
# random_search = RandomizedSearchCV(
#     estimator=gb,
#     param_distributions=param_distributions,
#     n_iter=10,
#     cv=skf,
#     scoring="accuracy",
#     random_state=42,
#     n_jobs=-1
# )
# random_search.fit(X,y)
# print("Best Parameters:", random_search.best_params_)
# print("Best Score:", random_search.best_score_)
# print("Best Model:", random_search.best_estimator_)

# --------------Learning Curve--------------------------

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=gb,
    X=X,
    y=y,
    cv=skf,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)
train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)
# print(train_sizes)
# print(train_mean)
# print(validation_mean)

# plt.figure(figsize=(8,5))

# plt.plot(train_sizes, train_mean, marker="o", label="Training Score")
# plt.plot(train_sizes, validation_mean, marker="o", label="Validation Score")

# plt.xlabel("Training Examples")
# plt.ylabel("Accuracy")
# plt.title("Learning Curve")

# plt.legend()

# plt.grid(True)

# plt.show()

# -----------------Validation Curve-------------------

tree = DecisionTreeClassifier(random_state=42)

train_scores, validation_scores = validation_curve(
    estimator=tree,
    X=X,
    y=y,
    param_name="max_depth",
    param_range=np.arange(1,11),
    cv=skf,
    scoring="accuracy",
    n_jobs=-1
)

# train_mean = train_scores.mean(axis=1)
# validation_mean = validation_scores.mean(axis=1)
# print(train_mean)
# print(validation_mean)

# plt.plot(
#     np.arange(1,11),
#     train_mean,
#     marker="o",
#     label="Training Score"
# )

# plt.plot(
#     np.arange(1,11),
#     validation_mean,
#     marker="o",
#     label="Validation Score"
# )

# plt.xlabel("max_depth")
# plt.ylabel("Accuracy")
# plt.title("Validation Curve")

# plt.legend()
# plt.grid(True)

# plt.show()

# ------------------Feature Importance -------------------

gb.fit(X, y)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gb.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# print(feature_importance)

# plt.figure(figsize=(10,6))

# plt.barh(
#     feature_importance["Feature"],
#     feature_importance["Importance"]
# )

# plt.xlabel("Importance")
# plt.ylabel("Features")

# plt.title("Gradient Boosting Feature Importance")

# plt.tight_layout()

# plt.show()

# ----------------------SHAP--------------------------
# Code Not Work Due To Version Conflict Do It Later

#  ----------------------Model Saving----------------------

# joblib.dump(gb, "01_Machine_Learning/Week_13_Model_Optimization/models/gradient_boosting_model.joblib")

# loaded_model = joblib.load("gradient_boosting_model.joblib")

# # Predict
# predictions = loaded_model.predict(X.iloc[:5])

# print(predictions)

# -------------------Entire Pipline ---------------------------
# pipeline = Pipeline([
#     ("model", gb)
# ])
# joblib.dump(pipeline, "pipeline.joblib")