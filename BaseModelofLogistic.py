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

"""
Input variables are the following: 
Country : land (Categorical)
Year : year (numeric)
democ : how democratic the country is (numeric)
autoc : how autocratic the country is (numeric)
polity : combined score of autoc and democ with a range of -10 to 10 (numeric)
polity2 : revised polity score
durable : Shows the duration of a regime untill a regime change happens (numeric)
xrreg : Regulation of Chief Executive Recruitment (numeric)
xrcomp : Competitiveness of Executive Recruitment: (numeric)
xropen : Openness of Executive Recruitment (numeric)
xconst : assesses the extent of institutionalized constraints on the decision-making powers of chief executives (numeric)
parreg : measuring the degree to which a political system regulates or restricts political participation. (numeric)
parcomp : assessing the competitiveness of political participation within a country. (numeric)
exconst : Executive Constraints: Concept variable is identical to XCONST (numeric)
d5 : Shows if a regime change happened in that year (boolean)
sf : Flag when a state failure happened (boolean)
regtrans : an indicator how heavy the regime shift was with an range of -2 to +3 with some special auxiliary codes (numeric)
"""

#load data
file_path = 'polity5_cleaned.csv'
polity5 = pd.read_csv('polity5_cleanedWithoutEXRECandPOLCOMP.csv')

# Initialize Rescaler 
scaler = MinMaxScaler() #StandardScaler

# Separate numerical and categorical features
numerical_features = ['cyear', 'ccode' ,'democ', 'autoc', 'polity', 'polity2', 'durable', 'xrreg', 'xrcomp', 'xropen', 'xconst', 'parreg', 'parcomp', 'exconst', 'regtrans']
categorical_features = ['d5', 'sf']

# Apply polynomial transformation
degree = 3  # You can change this to any desired degree
poly = PolynomialFeatures(degree=degree, include_bias=False)


# Apply standard scaling to numerical features
polity5[numerical_features] = scaler.fit_transform(polity5[numerical_features])

# Separate dependent variables and independent variables
X = polity5.drop(['d5'], axis=1)
y = polity5['d5']

#split data into test and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

x_new = sm.add_constant(X)

# Fit logistic regression model
logit_model = sm.Logit(y, x_new)
result = logit_model.fit()

# Obtain partial residuals
partial_resid = result.resid_pearson

def printPartialResidualPlots(columns):
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



# Display the summary of the logistic regression model
print(result.summary())

# create model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Obtain predicted probabilities on the test set
y_probs_poly = model.predict_proba(X_test_poly)[:, 1]
y_probs = model.predict_proba(X_test)[:, 1]
residuals = y_test - y_probs
residuals_poly = y_test - y_probs_poly

# Create a calibration plot
prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
prob_true_poly, prob_pred_poly = calibration_curve(y_test, y_probs_poly, n_bins=10)

# Plotting the calibration curve
plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # <-- Add this line
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

print("Performance of the Model:")
evaluate_model(model, X_test, y_test)

# Calculate Log-Loss
logloss = metrics.log_loss(y_test, y_probs)
print(f'Log-Loss: {logloss}')
model.fit(X_train_poly, y_train)


# Evaluate the model
y_pred = model.predict(X_test_poly)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Calculate Log-Loss
logloss = metrics.log_loss(y_test, y_pred)
print(f'Log-Loss: {logloss}')


# Plotting the calibration curve
plt.plot(prob_pred_poly, prob_true_poly, marker='o', label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')  # <-- Add this line
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Plot Polynominal')
plt.legend() 
plt.show()

# Plot residuals
plt.figure(figsize=(8, 6))
sns.residplot(x=y_probs_poly, y=residuals_poly, lowess=True, color='blue')
plt.title('Polynominal Residual Plot for Logistic Regression')
plt.xlabel('Predicted Probabilities')
plt.ylabel('Residuals')
plt.show()