# Experiment 23: K-Means, Agglomerative Clustering and GMM
# Dataset: Synthetic Dataset / Mall Customer Segmentation
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate Synthetic Data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
axes[0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis')
axes[0].set_title('K-Means Clustering')

# Agglomerative
agg = AgglomerativeClustering(n_clusters=4).fit(X)
axes[1].scatter(X[:, 0], X[:, 1], c=agg.labels_, cmap='viridis')
axes[1].set_title('Agglomerative Clustering')

# GMM
gmm = GaussianMixture(n_components=4, random_state=42).fit(X)
gmm_labels = gmm.predict(X)
axes[2].scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis')
axes[2].set_title('Gaussian Mixture Model')

plt.tight_layout()
plt.show()
