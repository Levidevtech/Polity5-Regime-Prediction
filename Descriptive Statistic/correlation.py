import pandas as pd

file_path = 'polity5_cleaned.csv'
df = pd.read_csv(file_path)

#indepedent value
x = [df['xrreg'], df['xrcomp'], df['xropen'], df['xconst'], df['parreg'], df['parcomp'], df['exrec'], df['exconst'], df['polcomp']]

def correlation(variable):
    #dependent value
    y = df['durable']
    for item in variable:
        print(f"{y.name} - {item.name} correlation: {y.corr(item)}")

correlation(x)
