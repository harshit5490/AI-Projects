import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import joblib

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

BASE_DIR = Path(__file__).parent

df = pd.read_csv(BASE_DIR/"datasets"/"Mall_Customers.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.nunique())

# sns.histplot(
#     df["Age"],
#     bins=20,
#     kde=True,
#     color="steelblue"
# )

# plt.title("Age Distribution")
# plt.xlabel("Age")
# plt.ylabel("Number of Customers")

# # plt.show()

# sns.histplot(
#     df["Annual Income (k$)"],
#     bins=20,
#     kde=True,
#     color="green"
# )

# plt.title("Annual Income Distribution")
# plt.xlabel("Annual Income (k$)")
# plt.ylabel("Number of Customers")

# # plt.show()

# sns.histplot(
#     df["Spending Score (1-100)"],
#     bins=20,
#     kde=True,
#     color="orange"
# )

# plt.title("Spending Score Distribution")
# plt.xlabel("Spending Score")
# plt.ylabel("Number of Customers")

# # plt.show()

# sns.countplot(
#     x="Gender",
#     data=df,
#     palette="Set2"
# )

# plt.title("Gender Distribution")

# # plt.show()

# sns.boxplot(
#     x=df["Age"],
#     color="skyblue"
# )

# plt.title("Age Boxplot")

# # plt.show()

# sns.boxplot(
#     x=df["Annual Income (k$)"],
#     color="lightgreen"
# )

# plt.title("Annual Income Boxplot")

# # plt.show()

# sns.boxplot(
#     x=df["Spending Score (1-100)"],
#     color="orange"
# )

# plt.title("Spending Score Boxplot")

# plt.show()

df_processed = df.copy()
df_processed.drop(columns="CustomerID", inplace=True)

df_processed["Gender"] = df_processed["Gender"].map({
    "Male" : 0,
    "Female" : 1
})
# print(df_processed.head())

X = df_processed
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)
# print(X_scaled.head())

# print(X_scaled.describe().round(2))

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

# plt.plot(
#     range(1,11),
#     wcss,
#     marker="o",
#     linewidth=2
# )

# plt.title("Elbow Method")
# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("WCSS (Inertia)")

# plt.grid(True)

# plt.show()
# print(wcss)

scores = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    scores.append(score)

    # print(f"K = {k}  Silhouette Score = {score:.4f}")


kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)  
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
# print(clusters[:20])
feature_columns = X.columns
df_processed["Cluster"] = clusters
# print(df_processed.head())

# print(df_processed["Cluster"].value_counts().sort_index())
centroids = kmeans.cluster_centers_

# print(centroids)

centroids_original = scaler.inverse_transform(centroids)

centroids_df = pd.DataFrame(
    centroids_original,
    columns=feature_columns
)

# print(centroids_df)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Cluster"] = clusters
# print(pca_df.head())

# sns.scatterplot(
#     data=pca_df,
#     x="PC1",
#     y="PC2",
#     hue="Cluster",
#     palette="Set2",
#     s=80
# )

# plt.title("Customer Segments (PCA Visualization)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")

# plt.legend(title="Cluster")

# plt.show()

# centroids_pca = pca.transform(kmeans.cluster_centers_)
# sns.scatterplot(
#     data=pca_df,
#     x="PC1",
#     y="PC2",
#     hue="Cluster",
#     palette="Set2",
#     s=70,
#     alpha=0.7
# )

# plt.scatter(
#     centroids_pca[:, 0],
#     centroids_pca[:, 1],
#     s=300,
#     c="black",
#     marker="X",
#     label="Centroids"
# )

# plt.title("Customer Segments with Centroids")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")

# plt.legend()

# plt.show()

df_final = df.copy()

df_final["Cluster"] = clusters

# print(df_final.head())

cluster_summary = df_final.groupby("Cluster").agg({
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean"
}).round(2)

# print(cluster_summary)

cluster_summary["Customers"] = df_final.groupby("Cluster").size()

# print(cluster_summary)

gender_summary = pd.crosstab(
    df_final["Cluster"],
    df_final["Gender"]
)

# print(gender_summary)

# sns.barplot(
#     data=cluster_summary.reset_index(),
#     x="Cluster",
#     y="Annual Income (k$)"
# )

# plt.title("Average Income by Cluster")

# plt.show()

# sns.barplot(
#     data=cluster_summary.reset_index(),
#     x="Cluster",
#     y="Spending Score (1-100)"
# )

# plt.title("Average Spending Score by Cluster")

# plt.show()

# sns.barplot(
#     data=cluster_summary.reset_index(),
#     x="Cluster",
#     y="Age"
# )

# plt.title("Average Age by Cluster")

# plt.show()

cluster_summary["Segment"] = [
    "Premium Customers",
    "Potential Customers",
    "Regular Customers",
    "Young Active Shoppers",
    "Conservative Customers"
]

print(cluster_summary)

joblib.dump(kmeans, "kmeans_customer_segmentation.pkl")
joblib.dump(scaler, "scaler.pkl")
loaded_model = joblib.load("kmeans_customer_segmentation.pkl")

loaded_scaler = joblib.load("scaler.pkl")
import pandas as pd

new_customer = pd.DataFrame({
    "Gender":[0],
    "Age":[28],
    "Annual Income (k$)":[90],
    "Spending Score (1-100)":[80]
})
new_customer_scaled = loaded_scaler.transform(new_customer)
prediction = loaded_model.predict(new_customer_scaled)

print(prediction)