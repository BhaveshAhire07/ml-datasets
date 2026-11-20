# Experiment 17: Linear SVM - Decision Boundary and Margins
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

# Using only first two features and two classes for visualization
iris = load_iris()
X = iris.data[iris.target != 2, :2]
y = iris.target[iris.target != 2]

svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

plt.figure(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(svm, X, plot_method='contour',
                                       levels=[-1, 0, 1], linestyles=['--', '-', '--'],
                                       ax=plt.gca())
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='k', label='Support Vectors')
plt.title('Linear SVM Decision Boundary & Margins')
plt.legend()
plt.show()
