import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from rescaled import Rescaler
from sklearn import metrics


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
polity5 = pd.read_csv('polity5_cleaned.csv')

# Initialize Rescaler 
scaler = StandardScaler() #StandardScaler

#seperate dependent variables and independent variables
x = polity5.drop(['d5', 'year', 'country'],axis=1)
y = polity5['d5']

#split data into test and training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply scaling to the independent variables
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# create model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

# Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

print("Performance of the Rescaled Model:")
evaluate_model(model, X_test, y_test)