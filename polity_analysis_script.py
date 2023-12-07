
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_and_predict_regime_stability(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Define the target variable
    data['target'] = (data['regtrans'] != 0.0).astype(int)

    # Splitting the dataset into features (X) and target (y)
    X = data.drop(['country', 'regtrans', 'target'], axis=1)
    y = data['target']

    # Splitting the data into training (up to 2010) and testing (from 2011 onwards)
    X_train = X[X['year'] < 2010]
    X_test = X[X['year'] >= 2010]
    y_train = y[X['year'] < 2010]
    y_test = y[X['year'] >= 2010]

    # Training the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=0)
    rf_classifier.fit(X_train, y_train)

    # Predicting for all years from 2011 onwards
    predictions = rf_classifier.predict(X_test)

    # Creating a DataFrame for predictions
    predicted_data = X_test.copy()
    predicted_data['Predicted Stability'] = predictions
    predicted_data['Country'] = data['country'][X['year'] >= 2010]
    predicted_data['Year'] = X_test['year']

    return predicted_data

# Usage
file_path = 'polity5_cleaned.csv'  # Replace with your dataset file path
predicted_data = train_and_predict_regime_stability(file_path)
predicted_data.to_csv('predicted_regime_stability.csv', index=False)

print(predicted_data.head())
