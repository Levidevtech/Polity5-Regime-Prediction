import pandas as pd
import statistics as st



file_path = 'p5v2018.xls'
df = pd.read_excel(file_path)

#central tendencies mean, median and mode

def CTinfo(variable):
   pd.set_option('display.max_rows', None)
   print(f"Variable: {variable.name}")
   print(f"Mean: {st.mean(variable)}")
   print(f"Median: {st.median(variable)}")
   print(f"Mode: {st.mode(variable)}")

   #print distibution
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
