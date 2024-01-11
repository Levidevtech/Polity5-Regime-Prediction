import pandas as pd

file_path = 'p5v2018.xls'
df = pd.read_excel(file_path)

#indepedent value
x = [df['xrreg'], df['xrcomp'], df['xropen'], df['xconst'], df['parreg'], df['parcomp'], df['exrec'], df['exconst'], df['polcomp']]

def correlation(variable):
    #dependent value
    y = df['durable']
    for item in variable:
        print(f"{y.name} - {item.name} correlation: {y.corr(item)}")

correlation(x)
