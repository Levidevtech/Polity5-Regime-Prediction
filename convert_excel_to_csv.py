import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'p5v2018.xls'

# Read the Excel file
data = pd.read_excel(excel_file_path)

# Columns to be removed
columns_to_remove = ['p5', 'cyear', 'ccode', 'scode', 'flag', 'bday', 'byear', 'bmonth', 'eyear', 'eday', 'emonth', ]
# Assuming each column has a linked column with '_link' suffix
linked_columns = [col + '_link' for col in columns_to_remove]
# Combining both lists
all_columns_to_remove = columns_to_remove + linked_columns

# Removing the specified columns and their linked columns
data = data.drop(columns=all_columns_to_remove, errors='ignore')

# Specify the path for the new CSV file
csv_file_path = 'polity5_dataset.csv'

# Write the data to a CSV file
data.to_csv(csv_file_path, index=False)

print("Conversion complete. The file has been saved to:", csv_file_path)
