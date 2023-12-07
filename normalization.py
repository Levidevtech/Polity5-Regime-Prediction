import pandas as pd

#load dataset
file_path = 'polity5_dataset.csv'
polity5 = pd.read_csv(file_path)



#Exclude country and year to get the min value
columns_to_exclude = ['country', 'year']
excluded_data = polity5.loc[:, ~polity5.columns.isin(columns_to_exclude)]
min_values =  excluded_data.min()

#shift data
shifted_data = excluded_data - min_values


print(min_values)
print(shifted_data)
print(polity5.info())
print(polity5.isnull().sum())