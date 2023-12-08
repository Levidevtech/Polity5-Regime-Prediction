import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Rescaler:
    def rescale(self, data):
        columns_to_exclude = ['country', 'year']
        excluded_data = data.loc[:, ~data.columns.isin(columns_to_exclude)]
        min_values = excluded_data.min()
        shifted_data = excluded_data - min_values
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaledX = scaler.fit_transform(shifted_data)
        return rescaledX
