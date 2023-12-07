import pandas as pd

class Preperation:
    file_path = 'polity5_dataset.csv'

    def Prepare_Data(self):
        # Load dataset
        p5 = pd.read_csv(self.file_path)
        
        # Prepare data 
        p5['regtrans'].fillna(0, inplace=True)
        p5['sf'].fillna(0, inplace=True)
        p5['d5'].fillna(0, inplace=True)
        p5.drop(['eprec', 'interim', 'bprec', 'fragment', 'prior', 'post', 'change', 'exrec', 'polcomp'], axis=1, inplace=True)
        p5.dropna(subset=['polity2'], inplace=True)
        self.fill_durable(p5)
        print(p5.isnull().sum())
        # Specify the path for the new CSV file
        csv_file_path = 'polity5_cleaned.csv'

        # Write the data to a CSV file
        p5.to_csv(csv_file_path, index=False)

        return 
    

    def fill_durable(self,dataframe):
        current_value = 0
        last_polity_score = None

        for index, polity_score in dataframe['polity2'].items():
            if pd.isna(polity_score):
                dataframe.at[index, 'durable'] = current_value
                current_value += 1
            else:
                if last_polity_score is not None and abs(polity_score - last_polity_score) >= 3:
                    current_value = 0
                dataframe.at[index, 'durable'] = current_value
                current_value += 1
                last_polity_score = polity_score







p = Preperation()
p.Prepare_Data()
