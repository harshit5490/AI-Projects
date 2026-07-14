"""
====================================================
Week 7 Capstone Project
Employee Feature Engineering Pipeline

Dataset:
IBM HR Analytics Employee Attrition & Performance

Author: Harshit Gupta
====================================================
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import(
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures
)

from sklearn.feature_selection import(
    VarianceThreshold,
    SelectKBest,
    f_classif,
    RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent

# Load dataset
df = pd.read_csv(
    BASE_DIR/"datasets"/"WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# print("=" * 70)
# print("Dataset Loaded Successfully")
# print("=" * 70)

# print(df.head())

# print("\nShape of Dataset:")
# print(df.shape)

# print("\nColumn Names:")
# print(df.columns.tolist())

# print("\nDataset Information:")
# df.info()

# print("\nStatistical Summary:")
# print(df.describe())

# print("\nMissing Values")
# print(df.isnull().sum())

# print("\nDuplicate Rows")
# print(df.duplicated().sum())

# print("\nUnique Values")

# for column in df.columns:
#     print(f"{column}: {df[column].nunique()}")

# categorical_columns = df.select_dtypes(include="object").columns

# print("\nCategorical Columns")

# print(categorical_columns)

# numerical_columns = df.select_dtypes(include="number").columns

# print("\nNumerical Columns")

# print(numerical_columns)

# print(df["Attrition"].value_counts())
# print(df["Attrition"].value_counts(normalize=True) * 100)

# columns = [
#     "BusinessTravel",
#     "Department",
#     "Gender",
#     "JobRole",
#     "MaritalStatus",
#     "Over18",
#     "OverTime",
#     "Attrition"
# ]

# for col in columns:
#     print("\n", "=" * 40)
#     print(col)
#     print("=" * 40)
#     print(df[col].value_counts())

#Create artificial missing values
df.loc[[5, 20, 100], "Age"] = np.nan
df.loc[[10, 50, 200], "MonthlyIncome"] = np.nan

# print("Missing Values After Creating:")
# print(df[["Age", "MonthlyIncome"]].isnull().sum())

# Fill missing Age with mean
df["Age"] = df["Age"].fillna(df["Age"].mean())

# Fill missing MonthlyIncome with median
df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())

# print("\nMissing Values After Filling:")
# print(df[["Age", "MonthlyIncome"]].isnull().sum())

# print("=" * 70)
# print("Removing Unnecessary Features")
# print("=" * 70)

# print("Shape Before Removing Columns:")
# print(df.shape)

columns_to_drop = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop)

# print("\nColumns Removed Successfully!")
# print("\nShape After Removing Columns:")
# print(df.shape)
# print("\nRemaining Columns:")

# for column in df.columns:
#     print(column)

# print("=" * 70)
# print("Skewness Analysis")
# print("=" * 70)

# skewness = df.select_dtypes(include="number").skew()

# print(skewness.sort_values(ascending=False))

# plt.figure(figsize=(8, 5))

# plt.hist(df["MonthlyIncome"], bins=30, edgecolor="black")

# plt.title("Monthly Income Distribution")

# plt.xlabel("Monthly Income")

# plt.ylabel("Frequency")

# plt.show()

# Check skewness before transformation
# print("Before:")
# print(df["MonthlyIncome"].skew())

# Apply log transformation
df["MonthlyIncome"] = np.log1p(df["MonthlyIncome"])

# Check skewness after transformation
# print("\nAfter:")
# print(df["MonthlyIncome"].skew())

# plt.figure(figsize=(8, 5))

# plt.hist(df["MonthlyIncome"], bins=30, edgecolor="black")

# plt.title("Monthly Income After Log Transformation")
# plt.xlabel("Monthly Income")
# plt.ylabel("Frequency")

# plt.show()

# print("=" * 70)
# print("Age Analysis")
# print("=" * 70)

# print(df["Age"].describe())

# print("\nSkewness:")
# print(df["Age"].skew())

# plt.figure(figsize=(8,5))

# plt.hist(df["Age"], bins=15, edgecolor="black")

# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Frequency")

# plt.show()

# plt.figure(figsize=(8,5))

# plt.boxplot([
#     df[df["Attrition"] == 0]["Age"],
#     df[df["Attrition"] == 1]["Age"]
# ])

# plt.xticks([1, 2], ["No Attrition", "Attrition"])

# plt.title("Age vs Attrition")

# plt.ylabel("Age")

# plt.show()
# print(
#     df.groupby("Attrition")["Age"].mean()
# )


# Create encoder
label_encoder = LabelEncoder()

# Binary columns
binary_columns = ["Attrition", "Gender", "OverTime"]

for col in binary_columns:
    label_encoder.fit(df[col])

    # print(f"\n{col}")
    # print(dict(zip(label_encoder.classes_,
                #    label_encoder.transform(label_encoder.classes_))))

    df[col] = label_encoder.transform(df[col])

# print("=" * 70)
# print("Label Encoding Completed")
# print("=" * 70)

one_hot_columns = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus"
]

df = pd.get_dummies(
    df,
    columns=one_hot_columns,
    drop_first=True,
    dtype=int
)

# print("=" * 70)
# print("One-Hot Encoding Completed")
# print("=" * 70)

# print("Shape After Encoding:")
# print(df.shape)
# print(df.head())

# continuous_features = [
#     "Age",
#     "MonthlyIncome",
#     "TotalWorkingYears",
#     "YearsAtCompany"
# ]

# print("=" * 70)
# print("Correlation Matrix")
# print("=" * 70)

# print(df[continuous_features].corr())

# corr = df[continuous_features].corr()

# plt.figure(figsize=(6,5))

# plt.imshow(corr, cmap="coolwarm", interpolation="nearest")

# plt.colorbar()

# plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)

# plt.yticks(range(len(corr.columns)), corr.columns)

# plt.title("Correlation Matrix")

# plt.tight_layout()

# plt.show()

# print("=" * 70)
# print("Before Scaling")
# print("=" * 70)

# X = df.drop("Attrition", axis=1)

# y = df["Attrition"]

# print("=" * 70)
# print("Feature Variance")
# print("=" * 70)

# print(X.var().sort_values())

# selector = VarianceThreshold(
#     threshold=0.01
# )

# X_selected = selector.fit_transform(X)

# selected_columns = X.columns[selector.get_support()]

# print("=" * 70)
# print("Selected Features")
# print("=" * 70)

# print(selected_columns)

# print()

# print("Original Features :", X.shape[1])
# print("Remaining Features:", len(selected_columns))

# print("=" * 70)
# print("Correlation with Target")
# print("=" * 70)

# correlation = df.corr(numeric_only=True)["Attrition"].sort_values(ascending=False)

# print(correlation)
# correlation = correlation.drop("Attrition")

# print(correlation)

# print("=" * 70)
# print("Top 10 Positive Correlations")
# print("=" * 70)

# print(correlation.head(10))
# print("=" * 70)
# print("Top 10 Negative Correlations")
# print("=" * 70)

# print(correlation.tail(10))

# plt.figure(figsize=(10, 12))

# correlation.sort_values().plot(kind="barh")

# plt.title("Feature Correlation with Attrition")

# plt.xlabel("Correlation")

# plt.show()

X = df.drop("Attrition", axis=1)
y = df["Attrition"]
# selector = SelectKBest(
#     score_func=f_classif,
#     k=10
# )

# X_new = selector.fit_transform(X, y)
# scores = selector.scores_

# feature_scores = (
#     pd.DataFrame({
#         "Feature": X.columns,
#         "Score": scores
#     })
#     .sort_values("Score", ascending=False)
# )

# print("=" * 70)
# print("Feature Scores")
# print("=" * 70)

# print(feature_scores)
# selected_features = X.columns[
#     selector.get_support()
# ]

# print("=" * 70)
# print("Top 10 Selected Features")
# print("=" * 70)

# print(selected_features)
# plt.figure(figsize=(10,6))

# top10 = feature_scores.head(10)

# plt.barh(
#     top10["Feature"],
#     top10["Score"]
# )

# plt.xlabel("ANOVA F Score")
# plt.title("Top 10 Features (SelectKBest)")

# plt.gca().invert_yaxis()

# plt.show()

# rf = RandomForestClassifier(
#     n_estimators=100,
#     random_state=42
# )

# rf.fit(X, y)
# importance = rf.feature_importances_

# feature_importance = pd.DataFrame({
#     "Feature": X.columns,
#     "Importance": importance
# })

# feature_importance = feature_importance.sort_values(
#     by="Importance",
#     ascending=False
# )

# print("=" * 70)
# print("Random Forest Feature Importance")
# print("=" * 70)

# print(feature_importance)

# top10 = feature_importance.head(10)

# plt.figure(figsize=(10,6))

# plt.barh(
#     top10["Feature"],
#     top10["Importance"]
# )

# plt.title("Top 10 Features (Random Forest)")
# plt.xlabel("Importance Score")

# plt.gca().invert_yaxis()

# plt.show()

# print("=" * 70)
# print("Recursive Feature Elimination (RFE)")
# print("=" * 70)


# # Model
# model = LogisticRegression(max_iter=5000,random_state=42,solver="liblinear")

# # Select Top 10 Features
# rfe = RFE(
#     estimator=model,
#     n_features_to_select=10
# )

# rfe.fit(X, y)

# selected_features = X.columns[rfe.support_]

# ranking = pd.DataFrame({
#     "Feature": X.columns,
#     "Selected": rfe.support_,
#     "Ranking": rfe.ranking_
# }).sort_values("Ranking")

# print(ranking)

# print("=" * 70)
# print("Top Features (RFE)")
# print("=" * 70)

# print(selected_features)

# pca_df = pd.DataFrame(
#     X_pca,
#     columns=["PC1", "PC2"]
# )

# print(pca_df.head())
# plt.figure(figsize=(8,6))

# plt.scatter(
#     pca_df["PC1"],
#     pca_df["PC2"],
#     alpha=0.6
# )

# plt.title("PCA Visualization")

# plt.xlabel("Principal Component 1")

# plt.ylabel("Principal Component 2")

# plt.grid(True)

# plt.show()
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95)

X_pca = pca.fit_transform(X_scaled)
print("=" * 70)
print("Original Shape")
print(X_scaled.shape)

print()

print("After PCA")
print(X_pca.shape)
print("=" * 70)
print("Explained Variance Ratio")
print("=" * 70)

print(pca.explained_variance_ratio_)
print()

print("Total Variance Retained:")

print(sum(pca.explained_variance_ratio_))

scale_columns = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]
# print(df[scale_columns].head())

scaler = StandardScaler()

df[scale_columns] = scaler.fit_transform(df[scale_columns])

# print("\nScaling Completed Successfully!")

# print("=" * 70)
# print("After Scaling")
# print("=" * 70)

# print(df[scale_columns].head())


