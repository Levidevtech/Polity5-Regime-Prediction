from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import tree

years = 1
lower_weight = 1.5
mid_lower_weight = 3.3
mid_upper_weight = -2
upper_weight = -1.5



def calculate_variance(row, data):
    # This function calculates the variance of the polity2 values of a country in a certain year range
    year_range = range(int(row['year'])-years, int(row['year'])+years+1)
    relevant_data = data[(data['ccode'] == row['ccode']) & (data['year'].isin(year_range))]
    variance = relevant_data['polity2'].var()
    slope = np.polyfit(relevant_data['year'], relevant_data['polity2'], 1)[0]
    value = (variance * np.sign(slope)).round(2)

    if value == 0.0 or value == -0.0:
        return 0.0
    
    return value
    
def split_data_by_country_year(data):
    # This function splits the data into training and testing sets by country and year
    # It returns a list of training sets and a list of testing sets
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
        
    return pandas.concat(X_train), pandas.concat(X_test), pandas.concat(y_train), pandas.concat(y_test)

def categorize_stability(var):
    # This function categorizes the variance into 5 categories with respect to the mean and standard deviation
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

parser = argparse.ArgumentParser()
parser.add_argument('--ccode', help='ccode to train on')
parser.add_argument('--country', help='country to predict')
parser.add_argument('--depth', help='depth of the decision tree')
args = parser.parse_args()

# read data from excel file and convert to csv
xls = pandas.read_excel('p5v2018.xls')
xls.to_csv('polity5_dataset.csv')

# read data from csv file
data = pandas.read_csv('polity5_dataset.csv')

# filter data by ccode if provided
if args.ccode:
    data = data[data['ccode'] == int(args.ccode)]

# drop unnecessary columns
data = data.drop(['Unnamed: 0', 'cyear', 'scode', 'country', 'flag', 'polity', 'p5', 'bprec', 'byear', 'bday', 'bmonth', 'eday', 'eyear', 'eprec', 'prior'], axis=1)

# fill missing values with 0
data = data.fillna(0)

# convert polity2 to int
data['year'] = data['year'].astype(int)

# calculate variance for each country in each year
data['variance'] = data.apply(lambda row: calculate_variance(row, data), axis=1)

# categorize variance into 5 categories
mean_var = data['variance'].mean()
std_var = data['variance'].std()
lower_threshold = (mean_var - std_var) + lower_weight
mid_lower_threshold = (mean_var - (std_var / 2)) + mid_lower_weight
mid_upper_threshold = (mean_var + (std_var / 2)) + mid_upper_weight
upper_threshold = (mean_var + std_var) + upper_weight

data['stability'] = data['variance'].apply(categorize_stability)

data.to_csv('polity5_dataset_random_forest.csv')
data.drop(['variance'], axis=1, inplace=True)

# split data into training and testing sets
X_train, X_test, y_train, y_test = split_data_by_country_year(data)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# show the decision tree up to a certain depth
if args.depth:
    plt.figure(figsize=(20,20))
    tree.plot_tree(classifier.estimators_[0], filled=True, max_depth=int(args.depth))
    plt.show()


# print results
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

if args.country:
    # predict the stability of ccode country in 2018
    country_data = data[data['ccode'] == int(args.country)]
    country_data = country_data[country_data['year'] == 2018]
    country_data = country_data.drop(['stability'], axis=1)
    country_data = country_data.fillna(0)
    y_pred = classifier.predict(country_data)
    print("Predicted stability:", y_pred[0])

    # plot the stability of ccode country over the years and make the prediction red in the plot and the actual stability green
    country_data = data[data['ccode'] == int(args.country)]
    country_data = country_data.drop(['stability'], axis=1)
    country_data = country_data.fillna(0)
    y_pred = classifier.predict(country_data)
    country_data['stability'] = y_pred
    country_data = country_data.sort_values('year')
    plt.plot(country_data['year'], country_data['stability'])
    plt.scatter(country_data['year'], country_data['stability'])
    plt.scatter(2018, y_pred[0], color='red')
    plt.scatter(2018, country_data['stability'].iloc[-1], color='green')
    plt.show()



    

    
