import pandas as pd
import numpy as np
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

        # Save 'year' and 'country' columns for later
        year_and_country = polity5[['year', 'country']]


        #rescale data 
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(shifted_data)

        #unscaled version
        unscaled = scaler.inverse_transform(rescaledX)

        # Include 'year' and 'country' columns in the rescaled DataFrame
        rescaled_df = pd.DataFrame(data=rescaledX, columns=shifted_data.columns)
        rescaled_df[['year', 'country']] = year_and_country
        

        # Summarize transformed data
        np.set_printoptions(precision=2)
        print("Rescaled Data:")
        print(rescaled_df.head(6))
        print("\nUnscaled Data:")
        print(unscaled[0:6, :])
        print("\nOriginal Shifted Data:")
        print(shifted_data.head(6))

        
        return rescaled_df

        # summarize transformed data
        numpy.set_printoptions(precision=2)
        print(rescaledX[0:6,:])
        print(unscaled[0:6,:])
        print(shifted_data)


r = Rescaler()
print(r.rescale())