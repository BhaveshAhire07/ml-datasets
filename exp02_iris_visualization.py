# Experiment 2: Iris - Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame

plt.figure(figsize=(12, 10))

# Histogram
plt.subplot(2, 2, 1)
sns.histplot(df['sepal length (cm)'], kde=True)
plt.title('Histogram: Sepal Length')

# Box Plot
plt.subplot(2, 2, 2)
sns.boxplot(data=df.drop('target', axis=1))
plt.title('Box Plot of Features')

# Scatter Plot
plt.subplot(2, 2, 3)
sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', hue='target', data=df)
plt.title('Scatter Plot')

# Correlation Heatmap
plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()
