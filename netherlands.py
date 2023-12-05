import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('path_to_your_dataset.csv')


features = ['democracy_score', 'autocracy_score', 'polity_score', ...] # Add more features as needed
target = 'drastic_change'


X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# future_years = np.arange(2023, 2028) # Next 5 years

# future_predictions = model.predict(future_data)

# plt.plot(future_years, future_predictions, label='Predicted Drastic Changes')
# plt.title('Predicted Political Trends for Netherlands (2023-2027)')
# plt.xlabel('Year')
# plt.ylabel('Probability of Drastic Change')
# plt.legend()
# plt.show()
