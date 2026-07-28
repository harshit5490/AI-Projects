from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
X, y = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=1.2,
    random_state=42
)


model = KMeans(
    n_clusters=3,
    random_state=42
)

# model.fit(X)
# clusters = model.predict(X)
# print(model.cluster_centers_)
wcss = []
for k in range(1,11):
    model = KMeans(
        n_clusters=k,
        random_state=42
    )

    model.fit(X)

    wcss.append(model.inertia_)

# print(wcss)    

# plt.figure(figsize=(8,5))

# plt.plot(range(1,11), wcss, marker='o')

# plt.xlabel("Number of Clusters (K)")
# plt.ylabel("WCSS")
# plt.title("Elbow Method")

# plt.show()
labels = model.fit_predict(X)
score = silhouette_score(X, labels)

# print(score)

model = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

labels = model.fit_predict(X)

linked = linkage(X, method="ward")

# plt.figure(figsize=(10,5))

# dendrogram(linked)

# plt.title("Dendrogram")

# plt.xlabel("Samples")

# plt.ylabel("Distance")

# plt.show()

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5
)

labels = dbscan.fit_predict(X)

# plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")

# plt.title("DBSCAN Clustering")

# plt.show()

tsne = TSNE(
    n_components=2,
    random_state=42
)

X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(8,6))

plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1]
)

plt.title("t-SNE Visualization")

plt.show()