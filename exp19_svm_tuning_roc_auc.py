# Experiment 19: SVM Hyperparameter Tuning, ROC Curve and AUC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 1. Load Data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Scale Features (Crucial for SVM performance and preventing convergence warnings)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Hyperparameter Tuning using GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
svm = SVC(kernel='rbf')

grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

best_svm = grid.best_estimator_

print("=== Hyperparameter Tuning Results ===")
print(f"Best Parameters: {grid.best_params_}")
print(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}")
print(f"Test Set Accuracy: {best_svm.score(X_test_scaled, y_test):.4f}")

# 4. Calculate AUC and Plot ROC
# SVM's decision_function provides the continuous confidence scores needed for an ROC curve
y_scores = best_svm.decision_function(X_test_scaled)

# Calculate explicit metrics
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

print(f"\nCalculated AUC: {roc_auc:.4f}")

# Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) - Tuned SVM')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
