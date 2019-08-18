import os 
import pandas as pd 
import matplotlib.pyplot as plt 

url = 'https://raw.githubusercontent.com/fuentesmarlon/CLUSTERING-PCA-APRIORI/master/train.csv'
df = pd.read_csv(url)

print(list(df.columns.values))
newDf =df[[""]]