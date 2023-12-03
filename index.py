# Full code for Random Forest model to predict drastic changes in Polity

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = 'polity5_dataset.csv'
polity5_df = pd.read_csv(file_path)

# Handling missing values
polity5_df.fillna(method='ffill', inplace=True)

# Defining drastic change as a change in 'polity' score by more than 2 points
polity5_df['change_in_polity'] = polity5_df['polity'].diff().fillna(0)
polity5_df['drastic_change'] = np.abs(polity5_df['change_in_polity']) > 2

# Selecting features and target variable
# democ, autoc, polity, polity2, durable, xrreg, xrcomp, xropen, xconst, parreg, parcomp, exrec, exconst, polcomp, change, and regtrans
features = ['democ', 'autoc', 'polity', 'polity2' , 'xrreg', 'xrcomp', 'xropen', 'xconst', 'parreg', 'parcomp', 'exrec', 'exconst', 'polcomp', 'change' ]
target = 'drastic_change'

# Splitting the data
X = polity5_df[features]
y = polity5_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Predicting and evaluating the model
y_pred = clf.predict(X_test_scaled)
report = classification_report(y_test, y_pred)

print(report)

# Predict the probability of drastic change for a country
# Get the data for the country
country_df = polity5_df[polity5_df['country'] == 'India']
# Get the features
country_features = country_df[features]
# Standardize the features
country_features_scaled = scaler.transform(country_features)
# Predict the probability of drastic change
prob = clf.predict_proba(country_features_scaled)[0][1]
print("Probability of drastic change:", prob)


