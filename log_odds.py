import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Load the dataset
file_path = 'polity5_cleaned.csv'
polity5 = pd.read_csv(file_path)

# Prepare the data for logistic regression model
X = polity5.drop(['d5', 'country', 'year'], axis=1)  # Drop non-feature columns
y = polity5['d5']  # Outcome variable

# Fit a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Predict probabilities for each feature
probs = model.predict_proba(X)[:, 1]  # Probability of class 1

# Plotting probabilities
for column in X.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=polity5[column], y=probs)
    plt.title(f'Probability of Regime Change vs {column}')
    plt.xlabel(column)
    plt.ylabel('Probability of Regime Change')
    plt.show()
