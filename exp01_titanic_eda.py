# Experiment 1: Titanic - EDA
import pandas as pd
from sklearn.datasets import fetch_openml

# Load data
titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto').frame

# Identify missing values, duplicates, and data types
print("Data Types:\n", titanic.dtypes)
print("\nMissing Values:\n", titanic.isnull().sum())
print("\nDuplicates:", titanic.duplicated().sum())

# Statistical Summary
print("\nStatistical Summary:\n", titanic.describe(include='all'))
