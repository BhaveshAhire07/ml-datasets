# Experiment 13: t-SNE Dimensionality Reduction (MNIST / Digits)
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = load_digits()
# Use a subset for faster t-SNE computation
X_subset = digits.data[:500]
y_subset = digits.target[:500]

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_subset)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, ticks=range(10))
plt.title('t-SNE of MNIST Digits (2D)')
plt.show()
