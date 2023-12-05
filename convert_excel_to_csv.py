import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

excel_file_path = 'p5v2018.xls'


data = pd.read_excel(excel_file_path)

unwanted_values = [-66, -77, -88]
mask = data.isin(unwanted_values).any(axis=1)

column_name_translation = {
    'country': 'country',
    'year': 'year',
    'fragment': 'fragmentation_status',
    'democ': 'democracy_score',
    'autoc': 'autocracy_score',
    'polity': 'polity_score',
    'durable': 'durability',
    'xrreg': 'executive_recruitment_regulation',
    'xrcomp': 'executive_recruitment_competitiveness',
    'xropen': 'executive_recruitment_openness',
    'xconst': 'executive_constraints',
    'parreg': 'party_regulation',
    'parcomp': 'party_competitiveness',
    'exrec': 'executive_recruitment',
    'exconst': 'executive_constitutionality',
    'polcomp': 'political_competition',
    'prior': 'prior_conditions',
    'eprec': 'emergency_provisions_recency',
    'interim': 'interim_government_status',
    'bprec': 'before_provision_recency',
    'post': 'post_condition_status',
    'change': 'political_change_after_transition',
    'regtrans': 'regime_transition'
}
data.rename(columns=column_name_translation, inplace=True)
data = data[~mask]
columns_to_remove = ['p5', 'cyear', 'ccode', 'scode', 'flag', 'bday', 'byear', 'bmonth', 'eyear', 'eday', 'emonth', 'polity2', 'd5', 'sf']
linked_columns = [col + '_link' for col in columns_to_remove]
all_columns_to_remove = columns_to_remove + linked_columns
data = data.drop(columns=all_columns_to_remove, errors='ignore')
data.fillna(0, inplace=True)
data.columns = data.columns.str.lower().str.replace(' ', '_')
csv_file_path = 'polity5_dataset.csv'
data.to_csv(csv_file_path, index=False)


# Script part

polity5_df = pd.read_csv(csv_file_path)

polity5_df.fillna(method='ffill', inplace=True)

polity5_df['change_in_polity'] = polity5_df['polity_score'].diff().fillna(0)
polity5_df['drastic_change'] = np.abs(polity5_df['change_in_polity']) > 2

features = ['year', 'democracy_score', 'autocracy_score', 'polity_score', 'durability', 'executive_recruitment_regulation', 'executive_recruitment_competitiveness', 'executive_recruitment_openness', 'executive_constraints' ]
target = 'drastic_change'

X = polity5_df[features]
y = polity5_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=9000)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
report = classification_report(y_test, y_pred)

print(report)

country_list = polity5_df['country'].unique()
# Do this for the following: Netherlands, India, United States, China, Russia, Brazil, South Africa, Nigeria, Japan, Australia, Niger, Afghanistan
country_list = ['Netherlands', 'India', 'United States', 'China', 'Russia', 'Brazil', 'South Africa', 'Nigeria', 'Japan', 'Australia', 'Niger', 'Afghanistan']

for country in country_list:
    country_df = polity5_df[polity5_df['country'] == country]
    country_features = country_df[features]
    country_features_scaled = scaler.transform(country_features)
    prob = clf.predict_proba(country_features_scaled)[0][1]
    percentage = prob * 100
    print(f"Probability of drastic change for {country}: {percentage.round(0)}%")
    

    
