import pandas as pd
from sklearn.preprocessing import Normalizer

class Normalize:
    def normalize(self, data):
        # Custom normalization for 'polity2' column
        if 'polity2' in data.columns:
            data['polity2'] = data['polity2'] / 10
        
        # Standard normalization for other columns
        columns_to_exclude = [col for col in ['country', 'year', 'normalized_polity2'] if col in data.columns]
        excluded_data = data.loc[:, ~data.columns.isin(columns_to_exclude)]

        if not excluded_data.empty:
            scaler = Normalizer().fit(excluded_data)
            normalized_excluded_data = scaler.transform(excluded_data)
            normalized_excluded_data = pd.DataFrame(normalized_excluded_data, columns=excluded_data.columns)
            data = pd.concat([data[columns_to_exclude], normalized_excluded_data], axis=1)

        return data
