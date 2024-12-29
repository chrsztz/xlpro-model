import pandas as pd


data = pd.read_pickle('df' + '.pkl')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('max_colwidth', None)

data = str(data)
ft = open('name' + '.xls', 'w')
ft.write(data)

