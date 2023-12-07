import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer

class Normalize:

    def normalize(self):
        # Load dataset
        file_path = 'polity5_cleaned.csv'
        polity5 = pd.read_csv(file_path)

        # Exclude country and year to get the min value
        columns_to_exclude = ['country', 'year']
        excluded_data = polity5.loc[:, ~polity5.columns.isin(columns_to_exclude)]
        min_values = excluded_data.min()
        max_values = excluded_data.max()

        # Shift data
        shifted_data = excluded_data - min_values

        # Normalize data
        scaler = Normalizer().fit(shifted_data)
        normalizedX = scaler.transform(shifted_data)

        # Summarize transformed data
        np.set_printoptions(precision=2)
        print(normalizedX[0:6, :])

        return normalizedX


n = Normalize()
n.normalize()

