# Experiment 11: Synthetic Data - Curse of Dimensionality
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

features = [2, 10, 50, 100, 500, 1000]
accuracies = []

for n in features:
    # Explicitly set n_redundant=0 to prevent ValueError on small feature sizes
    X, y = make_classification(
        n_samples=500,
        n_features=n,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(features, accuracies, marker='o', linestyle='-', color='b')
plt.xscale('log')
plt.xlabel('Number of Features (Log Scale)')
plt.ylabel('KNN Classification Accuracy')
plt.title('Curse of Dimensionality in KNN')
plt.grid(True)
plt.show()
