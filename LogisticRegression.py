import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from rescaled import Rescaler
from normalized import Normalize
import matplotlib.pyplot as plt


# Load the dataset
file_path = 'polity5_cleaned.csv'
polity5 = pd.read_csv('polity5_cleaned.csv')
country_year_data = polity5[['country', 'year']]

# Initialize Rescaler and Normalize classes
rescaler = Rescaler()
normalizer = Normalize()

# Apply rescaling and normalization
rescaled_data = rescaler.rescale(polity5)
normalized_data = normalizer.normalize(polity5)

print(rescaled_data)

# Convert numpy arrays back to dataframes and reattach 'country' and 'year'
rescaled_df = pd.DataFrame(rescaled_data, columns=polity5.columns.drop(['country', 'year']))
rescaled_df = pd.concat([country_year_data, rescaled_df], axis=1)

normalized_df = pd.DataFrame(normalized_data, columns=polity5.columns.drop(['country', 'year']))
normalized_df = pd.concat([country_year_data, normalized_df], axis=1)


# Check class distribution in the original dataset becuase we don't get any RegimeChangeHappened with 1 value
print("Original dataset class distribution:")
print(rescaled_df['d5'].value_counts())

# Splitting the Data for both datasets
def split_data(df):
    X = df.drop(['d5', 'country', 'year'], axis=1)
    y = df['d5']
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train_rescaled, X_test_rescaled, y_train_rescaled, y_test_rescaled = split_data(rescaled_df)
X_train_normalized, X_test_normalized, y_train_normalized, y_test_normalized = split_data(normalized_df)

# Build and Train Logistic Regression Models
model_rescaled = LogisticRegression().fit(X_train_rescaled, y_train_rescaled)
model_normalized = LogisticRegression().fit(X_train_normalized, y_train_normalized)

# Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

print("Performance of the Rescaled Model:")
evaluate_model(model_rescaled, X_test_rescaled, y_test_rescaled)

print("\nPerformance of the Normalized Model:")
evaluate_model(model_normalized, X_test_normalized, y_test_normalized)
