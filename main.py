import os 
import pandas as pd 
import matplotlib.pyplot as plt 

url = 'https://raw.githubusercontent.com/fuentesmarlon/CLUSTERING-PCA-APRIORI/master/test.csv'
df = pd.read_csv(url)

print(list(df.columns.values))
print(len(list(df.columns.values)))