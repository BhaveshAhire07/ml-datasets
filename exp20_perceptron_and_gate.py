# Experiment 20: Single-Layer Perceptron (AND Gate)
import numpy as np

# AND Gate Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

weights = np.zeros(X.shape[1])
bias = 0
lr = 0.1
epochs = 10

for epoch in range(epochs):
    for i in range(len(X)):
        linear_output = np.dot(X[i], weights) + bias
        y_pred = 1 if linear_output >= 0 else 0

        # Update rule
        update = lr * (y[i] - y_pred)
        weights += update * X[i]
        bias += update

print("Trained Weights:", weights)
print("Trained Bias:", bias)

print("\nPredictions for AND Gate:")
for i in range(len(X)):
    res = 1 if (np.dot(X[i], weights) + bias) >= 0 else 0
    print(f"Input: {X[i]} -> Predicted: {res}, Actual: {y[i]}")
