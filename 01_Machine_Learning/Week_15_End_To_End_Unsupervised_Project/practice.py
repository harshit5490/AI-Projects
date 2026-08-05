from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

plt.style.use("default")

PROJECT_ROOT = Path(__file__).resolve().parent
print(PROJECT_ROOT)
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Mall_Customers.csv"
)

df = pd.read_csv(DATA_PATH)

# print("=" * 70)
# print("Dataset Loaded")
# print("=" * 70)

# print(df.head())

# print("\nDataset Shape")
# print(df.shape)

# print("\nFirst Five Rows")
# print(df.head())

# print("\nLast Five Rows")
# print(df.tail())

# print("\nRandom Sample")

# print(
#     df.sample(
#         5,
#         random_state=42,
#     )
# )

# print("\nDataset Information")

# print(df.info())

# print("\nNumerical Summary")

# print(df.describe())

# print("\nCategorical Summary")

# print(
#     df.describe(
#         include="object"
#     )
# )

# print("\nMissing Values")

# print(df.isnull().sum())

# print("\nDuplicate Rows")

# print(df.duplicated().sum())

# print("\nUnique Values")

# for column in df.columns:

#     print("=" * 60)

#     print(column)

#     print(df[column].nunique())

# numerical_columns = [
#     "Age",
#     "Annual Income (k$)",
#     "Spending Score (1-100)",
# ]

# categorical_columns = [
#     "Gender",
# ]

# for column in numerical_columns:

#     plt.figure(figsize=(6,4))

#     sns.histplot(
#         data=df,
#         x=column,
#         bins=20,
#         kde=True,
#     )

#     plt.title(f"Distribution of {column}")

#     plt.show()

# for column in numerical_columns:

#     plt.figure(figsize=(6,2))

#     sns.boxplot(
#         x=df[column],
#     )

#     plt.title(column)

#     plt.show()

# plt.figure(figsize=(5,4))

# sns.countplot(
#     data=df,
#     x="Gender",
# )

# plt.title("Gender Distribution")

# plt.show()

# sns.pairplot(
#     df,
#     hue="Gender",
# )

# plt.show()

# correlation = df[
#     numerical_columns
# ].corr()

# print(correlation)

# plt.figure(figsize=(6,5))

# sns.heatmap(
#     correlation,
#     annot=True,
#     cmap="coolwarm",
# )

# plt.title("Correlation Matrix")

# plt.show()

# plt.figure(figsize=(8,6))

# sns.scatterplot(
#     data=df,
#     x="Annual Income (k$)",
#     y="Spending Score (1-100)",
#     hue="Gender",
# )

# plt.title("Income vs Spending")

# plt.show()

# plt.figure(figsize=(8,6))

# sns.scatterplot(
#     data=df,
#     x="Age",
#     y="Spending Score (1-100)",
#     hue="Gender",
# )

# plt.show()

# plt.figure(figsize=(8,6))

# sns.scatterplot(
#     data=df,
#     x="Age",
#     y="Annual Income (k$)",
#     hue="Gender",
# )

# plt.show()

numerical_columns = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)",
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    df[numerical_columns]
)

scaled_df = pd.DataFrame(
    scaled_features,
    columns=numerical_columns,
)

# print(scaled_df.head())

# print(df[numerical_columns].describe())

# print(scaled_df.describe())

# for column in numerical_columns:

#     plt.figure(figsize=(6,4))

#     sns.histplot(
#         scaled_df[column],
#         bins=20,
#         kde=True,
#     )

#     plt.title(f"Scaled {column}")

#     plt.show()

# wcss = []

# for k in range(1, 11):

#     model = KMeans(
#         n_clusters=k,
#         random_state=42,
#         n_init=10,
#     )

#     model.fit(scaled_df)

#     wcss.append(model.inertia_)

# plt.figure(figsize=(8,5))

# plt.plot(
#     range(1,11),
#     wcss,
#     marker="o",
# )

# plt.xlabel("Number of Clusters")

# plt.ylabel("WCSS (Inertia)")

# plt.title("Elbow Method")

# plt.grid(True)

# plt.show()     

# scores = []

# for k in range(2,11):

#     model = KMeans(
#         n_clusters=k,
#         random_state=42,
#         n_init=10,
#     )

#     labels = model.fit_predict(
#         scaled_df
#     )

#     score = silhouette_score(
#         scaled_df,
#         labels,
#     )

#     scores.append(score)

#     print(
#         f"k = {k}  ->  {score:.4f}"
#     )

# plt.figure(figsize=(8,5))

# plt.plot(
#     range(2,11),
#     scores,
#     marker="o",
# )

# plt.xlabel("Number of Clusters")

# plt.ylabel("Silhouette Score")

# plt.title("Silhouette Analysis")

# plt.grid(True)

# plt.show()    

pca = PCA()

pca.fit(scaled_df)

explained_variance = pca.explained_variance_ratio_

print(explained_variance)

cumulative_variance = np.cumsum(
    explained_variance
)

print(cumulative_variance)

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(explained_variance)+1
    ),
    explained_variance,
    marker="o",
)

plt.xlabel("Principal Component")

plt.ylabel("Explained Variance Ratio")

plt.title("Scree Plot")

plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(cumulative_variance)+1
    ),
    cumulative_variance,
    marker="o",
)

plt.xlabel("Number of Components")

plt.ylabel("Cumulative Variance")

plt.title("Cumulative Explained Variance")

plt.grid(True)

plt.show()

pca = PCA(
    n_components=2,
    random_state=42,
)

X_pca = pca.fit_transform(
    scaled_df
)

print(X_pca.shape)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("PCA Projection")

plt.show()

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10,
)

labels = kmeans.fit_predict(
    X_pca
)

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=labels,
    cmap="tab10",
)

plt.xlabel("PC1")

plt.ylabel("PC2")

plt.title("Customer Clusters (PCA)")

plt.colorbar()

plt.show()

original_score = silhouette_score(
    scaled_df,
    KMeans(
        n_clusters=6,
        random_state=42,
        n_init=10,
    ).fit_predict(scaled_df),
)

print(original_score)

pca_score = silhouette_score(
    X_pca,
    KMeans(
        n_clusters=6,
        random_state=42,
        n_init=10,
    ).fit_predict(X_pca),
)

print(pca_score)