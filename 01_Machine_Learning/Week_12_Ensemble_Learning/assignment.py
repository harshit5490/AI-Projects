import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from pathlib import Path

from sklearn.preprocessing import (
    LabelEncoder,StandardScaler
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR/"datasets"/"WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Analysis Of Data
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.describe(include="string"))
# print(df.isnull().sum())
# print(df.duplicated().sum())

# for col in df.columns:
#     print("="*40)
#     print(col)
#     print(df[col].nunique())

# print(df["Churn"].value_counts())
# print(df["Churn"].value_counts(normalize=True)*100)    

# print((df["TotalCharges"] == " ").sum())

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
# print(df.isnull().sum())
# print(df[df["TotalCharges"].isnull()])
# print(df.info())
# numerical_columns = [
#     "tenure",
#     "MonthlyCharges",
#     "TotalCharges"
# ]

# df[numerical_columns].hist(
#     figsize=(12,5),
#     bins=30
# )

# plt.tight_layout()

# plt.show()
df.dropna(inplace=True)
# print(df.isnull().sum())

#==================== univariate analysis==========================

# print(df["gender"].value_counts())

# plt.figure(figsize=(5,4))

# sns.countplot(
#     data=df,
#     x="gender"
# )

# plt.title("Gender Distribution")

# plt.show()

# print(df["SeniorCitizen"].value_counts())

# plt.figure(figsize=(5,4))

# sns.countplot(
#     data=df,
#     x="SeniorCitizen"
# )

# plt.title("Senior Citizen")

# plt.show()

# print(df["Partner"].value_counts())

# plt.figure(figsize=(5,4))

# sns.countplot(
#     data=df,
#     x="Partner"
# )

# plt.show()

# print(df["Dependents"].value_counts())

# plt.figure(figsize=(5,4))

# sns.countplot(
#     data=df,
#     x="Dependents"
# )

# plt.show()

# print(df["Contract"].value_counts())

# plt.figure(figsize=(7,4))

# sns.countplot(
#     data=df,
#     x="Contract"
# )

# plt.show()

# print(df["InternetService"].value_counts())

# plt.figure(figsize=(6,4))

# sns.countplot(
#     data=df,
#     x="InternetService"
# )

# plt.show()

# print(df["PaymentMethod"].value_counts())

# plt.figure(figsize=(10,5))

# sns.countplot(
#     data=df,
#     x="PaymentMethod"
# )

# plt.xticks(rotation=20)

# plt.show()

# =========================Bivariate Analysis=========================

# plt.figure(figsize=(6,4))

# sns.countplot(
#     data=df,
#     x="gender",
#     hue="Churn"
# )

# plt.title("Gender vs Churn")

# plt.show()

# plt.figure(figsize=(6,4))

# sns.countplot(
#     data=df,
#     x="SeniorCitizen",
#     hue="Churn"
# )

# plt.title("Senior Citizen vs Churn")

# plt.show()

# plt.figure(figsize=(6,4))

# sns.countplot(
#     data=df,
#     x="Partner",
#     hue="Churn"
# )

# plt.show()

# plt.figure(figsize=(6,4))

# sns.countplot(
#     data=df,
#     x="Dependents",
#     hue="Churn"
# )

# plt.show()

# plt.figure(figsize=(8,4))

# sns.countplot(
#     data=df,
#     x="Contract",
#     hue="Churn"
# )

# plt.show()

# plt.figure(figsize=(10,4))

# sns.countplot(
#     data=df,
#     x="PaymentMethod",
#     hue="Churn"
# )

# plt.xticks(rotation=20)

# plt.show()

# plt.figure(figsize=(10,5))

# sns.boxplot(
#     data=df,
#     x="Churn",
#     y="tenure"
# )

# plt.show()

# plt.figure(figsize=(8,5))

# sns.boxplot(
#     data=df,
#     x="Churn",
#     y="MonthlyCharges"
# )

# plt.show()

# plt.figure(figsize=(8,5))

# sns.boxplot(
#     data=df,
#     x="Churn",
#     y="TotalCharges"
# )

# plt.show()

# ======================= Data preprocessing ===================================

df.drop("customerID",axis=1,inplace=True)
# print(df.info())
# for col in df.columns:
#     print("=" * 50)
#     print(col)
#     print(df[col].unique())

# ---------- Encoding ---------------

binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "Churn"
]
le = LabelEncoder()

for col in binary_columns:
    df[col] = le.fit_transform(df[col])

multi_columns = [
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

df = pd.get_dummies(
    df,
    columns=multi_columns,
    drop_first=True,
    dtype=int
)

# print(df.head())
# print(df.info())
# print(df.shape)

# --------------Train Test Split-----------------
X = df.drop("Churn", axis=1)

y = df["Churn"]  
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -------------Scaling------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print("X_train :", X_train.shape)
# print("X_test  :", X_test.shape)

# print("y_train :", y_train.shape)
# print("y_test  :", y_test.shape)

# print("X_train_scaled :", X_train_scaled.shape)
# print("X_test_scaled  :", X_test_scaled.shape)

# ======================Resuable Metrics Evaluation Function===============================
results = []

def evaluate_model(model,model_name,X_train,X_test,y_train,y_test):

    # Training  Time

    start_train = time.time()

    model.fit(X_train, y_train)

    end_train = time.time()

    train_time = end_train - start_train

    # Prediction Time

    start_pred = time.time()

    y_pred = model.predict(X_test)

    end_pred = time.time()

    predict_time = end_pred - start_pred

    # Probability

    if hasattr(model,"predit_prob"):
        y_prob = model.predict_prob(X_test)[:,1]
    else:
        y_prob = None 

    # Metrics 

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    roc_auc = (
        roc_auc_score(y_test, y_prob)
        if y_prob is not None
        else None
    )

    # Result

    results.append({

        "Model": model_name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "ROC AUC": roc_auc,

        "Train Time": train_time,

        "Prediction Time": predict_time

    })

    # print Summary

    print("=" * 60)

    print(model_name)

    print("=" * 60)

    print(classification_report(y_test, y_pred))

    print("Accuracy :", round(accuracy, 4))

    print("Precision:", round(precision, 4))

    print("Recall   :", round(recall, 4))

    print("F1 Score :", round(f1, 4))

    if roc_auc is not None:
        print("ROC AUC  :", round(roc_auc, 4))

    print("Training Time :", round(train_time, 4), "seconds")

    print("Prediction Time:", round(predict_time, 4), "seconds")

    print() 


#===========================Baseline Model==========================================================
 
dt = DecisionTreeClassifier(
    random_state=42
)     
# evaluate_model(model=dt,model_name="Decision Tree",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# ------- Random Forest-----------

rf = RandomForestClassifier(
    random_state=42
)
# evaluate_model(model=rf,model_name="Random Forest",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# -------Extra Tree----------

et = ExtraTreesClassifier(
    random_state=42
)
# evaluate_model(model=et,model_name="Extra Tree",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)


# ---------------- AdaBoost-------

ada = AdaBoostClassifier(
    random_state=42
)
# evaluate_model(model=ada,model_name="AdaBoost",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# -----------Gradient Boosting------------

gb = GradientBoostingClassifier(
    random_state=42
)
# evaluate_model(model=gb,model_name="Gradient Boosting",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# ---------------XGBoost-------------
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)
# evaluate_model(model=xgb,model_name="XGBoost",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)

# -----------------LightGBM-----------
lgbm = LGBMClassifier(
    random_state=42,
    verbose = 1
)
# evaluate_model(model=lgbm,model_name="LightGBM",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)
# ---------------CatBoost------------
cat = CatBoostClassifier(
    random_state=42,
    verbose=0
)
evaluate_model(model=cat,model_name="CatBoost",X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test)