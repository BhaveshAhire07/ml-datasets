# Experiment 18: RBF and Polynomial Kernel SVM
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RBF Kernel
svm_rbf = SVC(kernel='rbf', gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
print(f"RBF SVM Accuracy: {accuracy_score(y_test, svm_rbf.predict(X_test_scaled)):.4f}")

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)
svm_poly.fit(X_train_scaled, y_train)
print(f"Polynomial SVM Accuracy: {accuracy_score(y_test, svm_poly.predict(X_test_scaled)):.4f}")
