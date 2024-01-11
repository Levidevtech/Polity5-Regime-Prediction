import pandas as pd
import statistics as st

file_path = 'polity5_cleaned.csv'
df = pd.read_csv(file_path)

#central tendencies mean, median and mode
def CTinfo(variable):
   print(f"Variable: {variable.name}")
   print(f"Mean: {st.mean(variable)}")
   print(f"Median: {st.median(variable)}")
   print(f"Mode: {st.mode(variable)}")

   #print variability and distibution 
   print(f"Variability: {variable.value_counts().describe()} " )
   print(f"Distribution: {pd.crosstab(index=variable, columns='count')}")

CTinfo(df['xrreg'])
CTinfo(df['xrcomp'])
CTinfo(df['xropen'])
CTinfo(df['xconst'])
CTinfo(df['parreg'])
CTinfo(df['parcomp'])
CTinfo(df['exrec'])
CTinfo(df['exconst'])
CTinfo(df['polcomp'])
CTinfo(df['durable'])