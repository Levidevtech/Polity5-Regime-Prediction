import pandas

data = pandas.read_excel('p5v2018.xls')

data.to_csv('polity5_dataset.csv')
