import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import statsmodels.api as sm

def printPartialResidualPlots(columns, partial_resid):
    # Create partial residual plots for each independent variable
    for i, var in enumerate(columns):
        if var == 'const':  # Skip the constant term
            continue

        plt.figure(figsize=(8, 4))
        plt.scatter(X[var], partial_resid, alpha=0.5)
        plt.title(f'Partial Residual Plot for {var}')
        plt.xlabel(var)
        plt.ylabel('Partial Residuals')
        plt.show()

# Load data
file_path = 'polity5_cleaned.csv'
polity5 = pd.read_csv(file_path)

# Initialize Rescaler
scaler = MinMaxScaler()
le = LabelEncoder()

# Encode 'country' using LabelEncoder
polity5['country'] = le.fit_transform(polity5['country'])

# Separate numerical and categorical features
numerical_features = ['year', 'democ', 'autoc', 'polity', 'polity2', 'durable', 'xrreg', 'xrcomp', 'xropen', 'xconst', 'parreg', 'parcomp', 'exconst', 'regtrans']
categorical_features = ['country', 'd5', 'sf']

# Apply polynomial transformation
degree = 3
poly = PolynomialFeatures(degree=degree, include_bias=False)

# Apply standard scaling to numerical features
polity5[numerical_features] = scaler.fit_transform(polity5[numerical_features])

# Separate dependent variables and independent variables
X = polity5.drop(['d5'], axis=1)
y = polity5['d5']

# Split data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply polynomial transformation and concatenate with the original DataFrame
X_poly = poly.fit_transform(X[numerical_features])
poly_columns = poly.get_feature_names_out(numerical_features)
X_poly_df = pd.DataFrame(X_poly, columns=poly_columns)
X_poly_const = sm.add_constant(X_poly_df)
X_poly_train, X_poly_test, y_poly_train, y_poly_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

x_new = sm.add_constant(X)

# Fit logistic regression model
logit_model = sm.Logit(y, x_new)
result = logit_model.fit()

# Obtain partial residuals
partial_resid = result.resid_pearson

# Display the summary of the logistic regression model
print(result.summary())

# create model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Obtain predicted probabilities on the test set
y_probs = model.predict_proba(X_test)[:, 1]
residuals = y_test - y_probs

# Create a calibration plot
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)

# Plotting the calibration curve
plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot')
plt.legend()
plt.show()

# Plot residuals
plt.figure(figsize=(8, 6))
sns.residplot(x=y_probs, y=residuals, lowess=True, color='blue')
plt.title('Residual Plot for Logistic Regression')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Residuals')
plt.show()

# Generate a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

# Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# Display the confusion matrix
print("Confusion Matrix:")
print(cm)

print("Performance of the Rescaled Model:")
evaluate_model(model, X_test, y_test)

# Calculate Log-Loss
logloss = metrics.log_loss(y_test, y_probs)
print(f'Log-Loss: {logloss}')

# Fit logistic regression model with polynomial features using scikit-learn
model_poly = LogisticRegression(max_iter=1000)
model_poly.fit(X_poly_train, y_poly_train)

# Evaluate the model
y_pred_poly = model_poly.predict(X_poly_test)
print("Confusion Matrix:")
print(confusion_matrix(y_poly_test, y_pred_poly))
print("\nClassification Report:")
print(classification_report(y_poly_test, y_pred_poly))

printPartialResidualPlots(X.columns, partial_resid)
printPartialResidualPlots(X_poly_df.columns, partial_resid)