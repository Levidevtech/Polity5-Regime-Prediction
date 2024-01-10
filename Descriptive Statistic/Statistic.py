import pandas as pd
import researchpy as rp
import statistics as st


file_path = 'p5v2018.xls'
df = pd.read_excel(file_path)

#central tendencies mean, median and mode

def CTinfo(variable):
   print(f"Variable: {variable.name}")
   print(f"Mean: {st.mean(variable)}")
   print(f"Median: {st.median(variable)}")
   print(f"Mode: {st.mode(variable)}")

   #print distibution
   print(f"Distribution: {variable.value_counts().describe()} " )
  

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
