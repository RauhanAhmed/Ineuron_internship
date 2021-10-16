# Importing required libraries

import numpy as np
import pandas as pd

# Ignoring warnings
import warnings 
warnings.filterwarnings(action='ignore')

# Exploratory Data Analysis (EDA)

# specifying client_id and client_secret to connect to cassandra database
client_id = 'GJZtsoMguQmWZOZZPPWIdeAw'
client_secret = 'yTdq.2Sf7PWG8bxd4mAClagWvZt0C9ATXv+c,.3a7tuczorvi5C0qBe,g0ZeZHJDfapam,t78bDOfuNL..pI,oSZiGITc9vAm6CoPJ24Oa68m2fCsCu+O3IujjDFFgI0'

# connecting to cassandra database
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config= {
        'secure_connect_bundle': r'secure-connect-my-database.zip'
}
auth_provider = PlainTextAuthProvider(client_id, client_secret)
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

df = pd.DataFrame(list(session.execute("select * from my_keyspace.adult_data;")))

# Removing the index column
df = df.iloc[:, 1:]

# classifying various categories in marital_status columns as single or married
df.replace(to_replace=df['marital_status'].unique(),\
    value = ['single','married','single','single','single',\
        'married','single'], inplace=True)

# encoding 0 for single and 1 for married
df.replace(to_replace=['single', 'married'], value = [0,1], inplace = True)

# Converting values from salary column(target feature) to binary values
# Substituting 0 where salary is <=50K and 1 where salary is more than 50K

df.replace(to_replace=[' <=50K', ' >50K'], value = [0, 1], inplace = True)

# creating a new feature gain/loss

gain_or_loss = np.zeros(len(df))
gain_index = df[df['capital_gain'] != 0].index
loss_index = df[df['capital_loss'] != 0].index
for index in gain_index:
    gain_or_loss[index] = 1
for index in loss_index:
    gain_or_loss[index] = -1
    
df['gain/loss'] = gain_or_loss.astype(int)

# Working on sex column
df.replace(to_replace = [' Female', ' Male'], value = [0, 1], inplace = True)

# We observe that higher the degree of a person, higher is the education_num
# So we need not do any Label Encoding here, as ranks are already assigned in the desired format
# Creating a dataframe to know which rank belongs to which degree

df_education_labels = df.groupby(by = 'education').describe()['education_num']['mean'].sort_values().reset_index()

# Naming education_num as education_rank

df.rename(columns = {'education_num': 'education_rank'}, inplace=True)


# Using simple imputer to replace question marks denoting missing data in the column workclass 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=' ?', strategy = 'most_frequent')
workclass_imputed = imputer.fit_transform(df[['workclass']])
df['workclass'] = workclass_imputed
df['workclass'].value_counts(normalize=True)*100

# Naming values in workclass column as others where value count is less than 2% of total values

arr_others = df['workclass'].value_counts(normalize=True)[df['workclass'].value_counts(normalize=True)*100 < 2].index
df.replace(to_replace = arr_others, value = ['others']*len(arr_others), inplace = True)


# Doing one-hot encoding on workclass
workclass = pd.get_dummies(df[['workclass']], drop_first=True)
# Renaming columns in workclass dataframe
workclass.columns = ['Local_gov','Private','Self_emp_inc','Self_emp_not_in_inc','State_gov','others']

# As we can see that we have only 5 categories in the race column alongwith an other category, so we will do one-hot encoding in this column directly
# doing one hot encoding on race column
race = pd.get_dummies(df[['race']], drop_first = True)

# Renaming columns in dataframe race
race.columns = ['Asian_Pac_Islander','Black','Other','White']

# working on occupation column
# Substituting question marks with most frequent value in the occupation column using simple imputer

imputer = SimpleImputer(missing_values=' ?', strategy = 'most_frequent')
occupation_imputed = imputer.fit_transform(df[['occupation']])
df['occupation'] = occupation_imputed

# Since, there are not much categories in the occupation column and they aren't ordinal so just doing one-hot encoding on occupation column too
# one-hot encoding on occupation column
occupation = pd.get_dummies(df[['occupation']], drop_first = True)
occupation.columns = ['Armed_Forces','Craft_repair','Exec_managerial','Farming_fishing','Handlers_cleaners','Machine_op_inspct',                      'Other_service','Priv_house_serv','Prof_specialty','Protective_serv','Sales','Tech_support',                      'Transport_moving']

# imputing most frequent value in place of question marks
imputer = SimpleImputer(missing_values=' ?', strategy = 'most_frequent')
country_imputed = imputer.fit_transform(df[['country']])
df['country'] = country_imputed

# naming less frequent countries as others (having value counts less than 0.3% of total values)
percentage_threshold = 0.3
arr_others = df['country'].value_counts()[df['country'].value_counts(normalize=True)*100 < percentage_threshold].index
df['country'].replace(to_replace=arr_others, value = ['others']*len(arr_others), inplace=True)

# Encoding countries in country column
country = pd.get_dummies(df[['country']], drop_first=True)
#Renaming columns in dataframe country
country.columns = ['El Salvador','Germany','India','Mexico','Philippines','Puerto Rico','United States','others']