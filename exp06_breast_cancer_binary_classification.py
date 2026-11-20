# Experiment 6: Breast Cancer - Binary Classification
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)
import numpy as np

data = load_breast_cancer()
# Use 'mean radius' (feature 0) as the single feature for a manual boundary
X = data.data[:, 0]
y = data.target  # 0: malignant, 1: benign

# Manual decision boundary: if radius < 15, classify as benign (1), else malignant (0)
y_pred = np.where(X < 15.0, 1, 0)

print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall:    {recall_score(y, y_pred):.4f}")
print(f"F1-score:  {f1_score(y, y_pred):.4f}")
