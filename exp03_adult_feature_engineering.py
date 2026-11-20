# Experiment 3: Adult Income - Feature Engineering
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Load a sample of the Adult dataset
adult = fetch_openml(data_id=1590, as_frame=True, parser='auto').frame.sample(1000, random_state=42)
X = adult.drop('class', axis=1)

# Missing value handling
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['category', 'object']).columns
X[num_cols] = SimpleImputer(strategy='mean').fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(X[cat_cols])

# Encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_cats = pd.DataFrame(encoder.fit_transform(X[cat_cols]))

# Normalization & Standardization
X_norm = MinMaxScaler().fit_transform(X[num_cols])
X_std = StandardScaler().fit_transform(X[num_cols])

print("Standardized Numerical Features (first 2 rows):\n", X_std[:2])
