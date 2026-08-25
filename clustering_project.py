import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


wine_data = load_wine()

df = pd.DataFrame(
    wine_data.data,
    columns=wine_data.feature_names
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


pca = PCA(n_components=2)
X_scaled_pca = pca.fit_transform(X_scaled)


k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)


plt.figure(figsize=(10, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker='o'
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")
plt.xticks(k_values)

plt.show()


Kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

cluster_labels = Kmeans.fit_predict(X_scaled)

kmeans_score = silhouette_score(
    X_scaled,
    cluster_labels
)

print("K-Means Silhouette Score:", kmeans_score)


linked = linkage(
    X_scaled,
    method='ward'
)

plt.figure(figsize=(12, 6))

dendrogram(linked)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")

plt.show()


hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

hierarchical_labels = hierarchical.fit_predict(X_scaled)

hierarchical_score = silhouette_score(
    X_scaled,
    hierarchical_labels
)

print(
    "Hierarchical Silhouette Score:",
    hierarchical_score
)


dbscan = DBSCAN(
    eps=2.3,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_scaled)

print(
    "DBSCAN Clusters:",
    np.unique(
        dbscan_labels,
        return_counts=True
    )
)


mask = dbscan_labels != -1

dbscan_score = silhouette_score(
    X_scaled[mask],
    dbscan_labels[mask]
)

print(
    "DBSCAN Silhouette Score:",
    dbscan_score
)


plt.figure(figsize=(10, 5))

plt.scatter(
    X_scaled_pca[:, 0],
    X_scaled_pca[:, 1],
    c=cluster_labels
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering")

plt.show()


plt.figure(figsize=(10, 5))

plt.scatter(
    X_scaled_pca[:, 0],
    X_scaled_pca[:, 1],
    c=hierarchical_labels
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Hierarchical Clustering")

plt.show()


plt.figure(figsize=(10, 5))

plt.scatter(
    X_scaled_pca[:, 0],
    X_scaled_pca[:, 1],
    c=dbscan_labels
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("DBSCAN Clustering")

plt.show()


print("\nFinal Comparison")
print("----------------")
print("K-Means:", kmeans_score)
print("Hierarchical:", hierarchical_score)
print("DBSCAN:", dbscan_score)