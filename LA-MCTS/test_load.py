import pandas as pd

path = 'Ackley_10_results.csv'
    
df = pd.read_csv(path, header=None)
print(df)