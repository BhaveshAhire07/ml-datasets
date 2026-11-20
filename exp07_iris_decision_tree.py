# Experiment 7: Iris - Decision Tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)

plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True,
          feature_names=load_iris().feature_names,
          class_names=load_iris().target_names)
plt.title('Decision Tree (Iris)')
plt.show()
