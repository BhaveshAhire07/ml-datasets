# Experiment 22: Random Forest and AdaBoost
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, ada.predict(X_test)):.4f}")

# Feature Importance Visualization
pd.Series(rf.feature_importances_,
          index=data.feature_names).nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.show()
