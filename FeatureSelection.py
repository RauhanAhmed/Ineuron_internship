import pandas as pd
import numpy as np
from FeatureEngineering import df, workclass, race, country, occupation

# Outlier Treatment
# Removing outliers from age column
age_mean = df['age'].mean()
age_std = df['age'].std()

upper_bound = age_mean + 3*age_std
lower_bound = age_mean - 3*age_std

df = df[(df['age'] <= upper_bound) & (df['age'] >= lower_bound)]

# Feature Selection
# Dropping useless columns
df.drop(['workclass','fnlwgt','education','occupation',\
    'relationship','race',\
    'capital_gain','capital_loss','country'],axis = 1, inplace = True)

# Renaming columns age as age_logarithmic 
df.rename(columns = {'age':'age_logarithmic'}, inplace = True)

print('Done the job...!!!')
print(race.columns)