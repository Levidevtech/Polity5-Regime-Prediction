import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Function to load and preprocess the dataset
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()  # Drop rows with missing values
    # Converting 'country' to categorical and encoding it
    data['country'] = data['country'].astype('category')
    return data

# Function to train a Random Forest model and evaluate its accuracy
def train_and_evaluate_rf(data, target_column='polity_score'):
    # Get the code for the Netherlands
    netherlands_code = data['country'].cat.categories.get_loc('Netherlands')

    # Filter out the data for the Netherlands
    netherlands_data = data[data['country'].cat.codes == netherlands_code]
    # Remove the Netherlands data from the training dataset
    data = data[data['country'].cat.codes != netherlands_code]

    # Convert 'country' to codes for training
    data['country'] = data['country'].cat.codes

    # Splitting the data into features and target
    X = data.drop(columns=[target_column, 'year'])  # Dropping 'year' as it's not suitable for RF directly
    y = data[target_column]

    # Splitting the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Predicting on the test set and calculating RMSE
    y_pred = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Making a prediction for the most recent year of the Netherlands
    latest_nl_data = netherlands_data[netherlands_data['year'] == netherlands_data['year'].max()].drop(columns=[target_column, 'year'])
    latest_nl_data['country'] = latest_nl_data['country'].cat.codes  # Convert 'country' to codes for prediction
    nl_prediction = rf_model.predict(latest_nl_data)

    return rmse, nl_prediction

# Main script
if __name__ == "__main__":
    file_path = 'polity5_dataset.csv'

    # Load and preprocess the data
    data = load_and_preprocess(file_path)

    # Train the model, evaluate, and predict for the Netherlands
    accuracy, nl_pred = train_and_evaluate_rf(data)
    print(f"Model RMSE: {accuracy}")
    print(f"Prediction for the Netherlands: {nl_pred[0]}")
