import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer

class Normalize:

    def normalize(self):
        # Load dataset
        file_path = 'polity5_cleaned.csv'
        polity5 = pd.read_csv(file_path)

        # Exclude country and year to get the min value
        columns_to_exclude = ['country','year']
        excluded_data = polity5.loc[:, ~polity5.columns.isin(columns_to_exclude)]
        min_values = excluded_data.min()

        # Shift data
        shifted_data = excluded_data - min_values

        # Normalize data
        scaler = Normalizer().fit(shifted_data)
        normalizedX = scaler.transform(shifted_data)

        # Create a DataFrame with normalized data and include 'country' column
        normalized_df = pd.DataFrame(data=normalizedX, columns=shifted_data.columns)
        normalized_df['country'] = polity5['country']  # Add 'country' column back
        normalized_df['year'] = polity5['year']

        # Summarize transformed data
        np.set_printoptions(precision=2)
        print(normalized_df.head(6))

        return normalized_df


n = Normalize()
n.normalize()

