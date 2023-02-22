import pandas as pd

df = pd.read_csv('Tesla_stock_Price.csv')

i = 0
for index, row in df.iterrows():
    df.at[index, 'Date'] = i
    i += 1

df.to_csv('Tesla_stock_Price.csv', index=False)