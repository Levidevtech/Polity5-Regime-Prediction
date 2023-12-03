import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'p5v2018.xls'

# Read the Excel file
data = pd.read_excel(excel_file_path)

# Specify the path for the new CSV file
csv_file_path = 'polity5_dataset.csv'

# Write the data to a CSV file
data.to_csv(csv_file_path, index=False)

print("Conversion complete. The file has been saved to:", csv_file_path)
