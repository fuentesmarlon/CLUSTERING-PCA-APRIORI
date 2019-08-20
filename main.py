import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import copy 
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import stats
import numpy as np 

url = 'https://raw.githubusercontent.com/fuentesmarlon/CLUSTERING-PCA-APRIORI/master/train.csv'
df = pd.read_csv(url)

df['Id'] = df['Id'].astype('category')
df['MSSubClass'] = df['MSSubClass'].astype('category')
df['MSZoning'] = df['MSZoning'].astype('category')
df['Street'] = df['Street'].astype('category')
df['Alley'] = df['Alley'].astype('category')
df['LotShape'] = df['LotShape'].astype('category')
df['LandContour'] = df['LandContour'].astype('category')
df['Utilities'] = df['Utilities'].astype('category')
df['LotConfig'] = df['LotConfig'].astype('category')
df['LandSlope'] = df['LandSlope'].astype('category')
df['Neighborhood'] = df['Neighborhood'].astype('category')
df['Condition1'] = df['Condition1'].astype('category')
df['Condition2'] = df['Condition2'].astype('category')
df['BldgType'] = df['BldgType'].astype('category')
df['HouseStyle'] = df['HouseStyle'].astype('category')
df['OverallQual'] = df['OverallQual'].astype('category')
df['OverallCond'] = df['OverallCond'].astype('category')
df['YearBuilt'] = df['YearBuilt'].astype('category')
df['YearRemodAdd'] = df['YearRemodAdd'].astype('category')
df['RoofStyle'] = df['RoofStyle'].astype('category')
df['RoofMatl'] = df['RoofMatl'].astype('category')
df['Exterior1st'] = df['Exterior1st'].astype('category')
df['Exterior2nd'] = df['Exterior2nd'].astype('category')
df['MasVnrType'] = df['MasVnrType'].astype('category')
df['ExterQual'] = df['ExterQual'].astype('category')
df['ExterCond'] = df['ExterCond'].astype('category')
df['Foundation'] = df['Foundation'].astype('category')
df['BsmtQual'] = df['BsmtQual'].astype('category')
df['BsmtCond'] = df['BsmtCond'].astype('category')
df['BsmtExposure'] = df['BsmtExposure'].astype('category')
df['BsmtFinType1'] = df['BsmtFinType1'].astype('category')
df['BsmtFinType2'] = df['BsmtFinType2'].astype('category')
df['Heating'] = df['Heating'].astype('category')
df['HeatingQC'] = df['HeatingQC'].astype('category')
df['CentralAir'] = df['CentralAir'].astype('category')
df['Electrical'] = df['Electrical'].astype('category')
df['KitchenQual'] = df['KitchenQual'].astype('category')
df['Functional'] = df['Functional'].astype('category')
df['FireplaceQu'] = df['FireplaceQu'].astype('category')
df['GarageType'] = df['GarageType'].astype('category')
df['GarageYrBlt'] = df['GarageYrBlt'].astype('category')
df['GarageFinish'] = df['GarageFinish'].astype('category')
df['GarageQual'] = df['GarageQual'].astype('category')
df['GarageCond'] = df['GarageCond'].astype('category')
df['PavedDrive'] = df['PavedDrive'].astype('category')
df['PoolQC'] = df['PoolQC'].astype('category')
df['Fence'] = df['Fence'].astype('category')
df['MiscFeature'] = df['MiscFeature'].astype('category')
df['MoSold'] = df['MoSold'].astype('category')
df['YrSold'] = df['YrSold'].astype('category')
df['SaleType'] = df['SaleType'].astype('category')
df['SaleCondition'] = df['SaleCondition'].astype('category')

df_numerical = df.select_dtypes(exclude=['category'])
df_categorical = df.select_dtypes(include=['category'])



toScale=df_numerical[["GarageArea","GarageCars","TotRmsAbvGrd","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath"]]
#nulls = toScale.isnull().sum().sort_values(ascending=False)
#print(nulls.head(20))


toScale=toScale.reset_index()
scaled_data=preprocessing.scale(toScale)
pca=PCA()
pca.fit(scaled_data)
per_var = np.round(pca.explained_variance_ratio_ *100, decimals=1)
labels=['PC'+str(x) for x in range(1,len(per_var)+1)]

print(np.argmin(scaled_data))
#pca_df = pd.DataFrame(scaled_data, index=[*wt, *ko],columns=labels)