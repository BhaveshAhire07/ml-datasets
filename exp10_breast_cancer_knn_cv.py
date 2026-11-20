# Experiment 10: Breast Cancer - KNN Cross-Validation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_neighbors': range(1, 15)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best K: {grid.best_params_['n_neighbors']}")
print(f"Best CV Accuracy: {grid.best_score_:.4f}")
print(f"Test Set Accuracy: {grid.score(X_test, y_test):.4f}")
