# Experiment 4: Salary - Linear Regression (Synthetic equivalent)
import numpy as np
import matplotlib.pyplot as plt

# Synthetic Salary Data (Years Experience vs Salary)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([38000, 42000, 55000, 61000, 68000, 83000, 92000, 105000, 112000, 122000])

# Math implementation
x_mean, y_mean = np.mean(X), np.mean(y)
b1 = np.sum((X - x_mean) * (y - y_mean)) / np.sum((X - x_mean) ** 2)
b0 = y_mean - b1 * x_mean

y_pred = b0 + b1 * X
rmse = np.sqrt(np.mean((y - y_pred) ** 2))
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y_mean) ** 2))

print(f"RMSE: {rmse:.2f}, R2: {r2:.4f}")

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.title('Simple Linear Regression (Salary)')
plt.legend()
plt.show()
