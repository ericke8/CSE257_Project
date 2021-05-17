import pandas as pd

path = 'Ackley_10_bo_results.csv'
    
df = pd.read_csv(path, header=None)
print(len(df))
print(df.head())