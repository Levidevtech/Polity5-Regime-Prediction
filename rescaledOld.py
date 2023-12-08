import pandas as pd
import numpy
from sklearn.preprocessing import MinMaxScaler

class Rescaler:

    def rescale(self):
        #load dataset
        file_path = 'polity5_cleaned.csv'
        polity5 = pd.read_csv(file_path)

        #Exclude country and year to get the min value
        columns_to_exclude = ['country', 'year']
        excluded_data = polity5.loc[:, ~polity5.columns.isin(columns_to_exclude)]
        min_values =  excluded_data.min()

        #shift data
        shifted_data = excluded_data - min_values

        #rescale data 
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(shifted_data)

        #unscaled version
        unscaled = scaler.inverse_transform(rescaledX)
        
        return rescaledX

        # summarize transformed data
        numpy.set_printoptions(precision=2)
        print(rescaledX[0:6,:])
        print(unscaled[0:6,:])
        print(shifted_data)


r = Rescaler()
print(r.rescale())