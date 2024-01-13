import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import argparse
import numpy as np
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt


def calculate_variance(row, data):
    year_range = range(int(row['year'])-2, int(row['year'])+2)
    relevant_data = data[(data['ccode'] == row['ccode']) & (data['year'].isin(year_range))]
    variance = relevant_data['polity2'].var()
    slope = np.polyfit(relevant_data['year'], relevant_data['polity2'], 1)[0]
    value = (variance * np.sign(slope)).round(2)
    
    # if value == -0.0, return 0.0
    if value == 0.0 or value == -0.0:
        return 0.0
    
    return value
    
def split_data_by_country_year(data):
    X_train, X_test, y_train, y_test = [], [], [], []
    for _, group_data in data.groupby('ccode'):
        group_data_sorted = group_data.sort_values('year')
        split_index = int(len(group_data_sorted) * 0.7)
        X_tr, X_te = group_data_sorted.iloc[:split_index, :], group_data_sorted.iloc[split_index:, :]
        y_tr, y_te = X_tr['stability'], X_te['stability']
        X_tr = X_tr.drop(['stability'], axis=1)
        X_te = X_te.drop(['stability'], axis=1)
        X_train.append(X_tr)
        X_test.append(X_te)
        y_train.append(y_tr)
        y_test.append(y_te)
        
    return pd.concat(X_train), pd.concat(X_test), pd.concat(y_train), pd.concat(y_test)

def categorize_stability(var):
    if var < lower_threshold:
        return -2
    elif lower_threshold <= var < mid_lower_threshold:
        return -1
    elif mid_lower_threshold <= var < mid_upper_threshold:
        return 0
    elif mid_upper_threshold <= var < upper_threshold:
        return 1
    else:
        return 2


# read parameters from command line
parser = argparse.ArgumentParser()
parser.add_argument('--ccode', help='ccode to train on')
parser.add_argument('--country', help='country to predict')
args = parser.parse_args()



data = pd.read_csv('polity5_dataset.csv')

if args.ccode:
    data = data[data['ccode'] == int(args.ccode)]

data = data.drop(['Unnamed: 0', 'cyear', 'scode', 'country', 'flag', 'polity', 'p5', 'bprec', 'byear', 'bday', 'bmonth', 'eday', 'eyear', 'eprec', 'prior'], axis=1)
data = data.fillna(0)
data['year'] = data['year'].astype(int)
data['variance'] = data.apply(lambda row: calculate_variance(row, data), axis=1)

mean_var = data['variance'].mean()
std_var = data['variance'].std()

lower_threshold = mean_var - std_var - 1
mid_lower_threshold = mean_var - (std_var / 2) - 1
mid_upper_threshold = mean_var + (std_var / 2) - 1
upper_threshold = mean_var + std_var - 1

data['stability'] = data['variance'].apply(categorize_stability)

data.to_csv('polity5_dataset_random_forest.csv')
data.drop(['variance'], axis=1, inplace=True)

X = data.drop('stability', axis=1)
y = data['stability']

X_train, X_test, y_train, y_test = split_data_by_country_year(data)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for i in range(3):
    tree = classifier.estimators_[i]
    dot_data = export_graphviz(tree.est,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
fig.savefig('rf_5trees.png')
    

if args.country:
    # predict for a specific country
    country_data = data[data['ccode'] == int(args.country)]
    country_data = country_data.drop(['stability'], axis=1)
    country_data = country_data.sort_values('year')
    country_data = country_data.iloc[-1, :]
    country_data = country_data.drop(['ccode', 'year'])
    country_data = country_data.values.reshape(1, -1)
    prediction = classifier.predict(country_data)
    print(prediction)

    # visualize the decision tree
    from sklearn import tree
    import graphviz
    dot_data = tree.export_graphviz(classifier.estimators_[0], out_file=None, 
                         feature_names=X.columns,  
                         class_names=['-2', '-1', '0', '1', '2'],  
                         filled=True, rounded=True,  
                         special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")

    # visualize the prediction  
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.plot(country_data[0], label='actual')
    plt.plot(prediction[0], label='predicted')
    plt.legend()
    plt.show()
    
