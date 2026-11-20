# Experiment 21: Multilayer Perceptron (MLP) Activation Functions
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ReLU Activation
mlp_relu = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                         max_iter=1000, random_state=42)
mlp_relu.fit(X_train, y_train)
print(f"MLP (ReLU) Accuracy: {accuracy_score(y_test, mlp_relu.predict(X_test)):.4f}")

# Tanh Activation
mlp_tanh = MLPClassifier(hidden_layer_sizes=(16, 8), activation='tanh',
                         max_iter=1000, random_state=42)
mlp_tanh.fit(X_train, y_train)
print(f"MLP (Tanh) Accuracy: {accuracy_score(y_test, mlp_tanh.predict(X_test)):.4f}")
