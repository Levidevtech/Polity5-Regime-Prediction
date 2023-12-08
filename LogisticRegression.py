import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from rescaled import Rescaler
from normalized import Normalize

# Function to calculate regime change within the same country for scaled data
def calculate_regime_change_scaled(group):
    group['RegimeChangeHappened'] = ((group['polity2'].shift(2) <= 9) & (group['polity2'] > 10)) | \
                                   ((group['polity2'].shift(2) >= 11) & (group['polity2'] < 10))
    group['RegimeChangeHappened'] = group['RegimeChangeHappened'].astype(int)
    return group

# Function to calculate regime change within the same country for normalized data
def calculate_regime_change_normalized(group):
    threshold_auto_normalized = -0.1
    threshold_demo_normalized = 0.1
    group['RegimeChangeHappened'] = ((group['polity2'].shift(2) <= threshold_auto_normalized) & (group['polity2'] > threshold_demo_normalized)) | \
                                   ((group['polity2'].shift(2) >= threshold_demo_normalized) & (group['polity2'] < threshold_auto_normalized))
    group['RegimeChangeHappened'] = group['RegimeChangeHappened'].astype(int)
    return group

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

# Convert numpy arrays back to dataframes and reattach 'country' and 'year'
rescaled_df = pd.DataFrame(rescaled_data, columns=polity5.columns.drop(['country', 'year']))
rescaled_df = pd.concat([country_year_data, rescaled_df], axis=1)

normalized_df = pd.DataFrame(normalized_data, columns=polity5.columns.drop(['country', 'year']))
normalized_df = pd.concat([country_year_data, normalized_df], axis=1)

# Apply regime change calculation within each country group for both datasets
rescaled_df = rescaled_df.groupby('country').apply(calculate_regime_change_scaled)
normalized_df = normalized_df.groupby('country').apply(calculate_regime_change_normalized)

# Check class distribution in the original dataset becuase we don't get any RegimeChangeHappened with 1 value
print("Original dataset class distribution:")
print(rescaled_df['RegimeChangeHappened'].value_counts())

# Splitting the Data for both datasets
def split_data(df):
    X = df.drop(['RegimeChangeHappened', 'country', 'year'], axis=1)
    y = df['RegimeChangeHappened']
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
