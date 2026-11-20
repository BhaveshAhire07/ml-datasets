# Experiment 14: Classification Accuracy Before and After PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Original Features
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
acc_orig = accuracy_score(y_test, clf.predict(X_test))

# PCA Reduced Features
pca = PCA(n_components=5, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

clf_pca = RandomForestClassifier(random_state=42)
clf_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))

print(f"Accuracy with Original Features: {acc_orig:.4f}")
print(f"Accuracy with PCA (5 components): {acc_pca:.4f}")
