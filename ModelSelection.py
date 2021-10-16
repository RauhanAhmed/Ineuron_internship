from FeatureSelection import df, race, occupation, workclass, country
import pandas as pd

df1 = df.copy()

# removing target feature salary from the dataframe df for now and storing it in another variable
salary = df1['salary'].reset_index(drop=True)
df1 = df1.drop(['salary'], axis=1)

# Building Machine Learning model 

# Importing different ML Algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Creating a function to concatenate different one hot encoded dataframes
def concat_dataframes(data):
    dataframe = pd.concat([data, workclass.iloc[data.index, :],                           race.iloc[data.index , :],                           occupation.iloc[data.index, :],                           country.iloc[data.index, :]], axis = 1)
    
    dataframe = dataframe.dropna()
    dataframe = dataframe.reset_index(drop = True)
    return dataframe

df1 = concat_dataframes(df1)

# using sklearn column transformer to standerdize the two continuous features we have in our data
features = ['age_logarithmic', 'hours_per_week']
scaler = ColumnTransformer(transformers = [
                                           ('scale_num_features', StandardScaler(), features)
], remainder='passthrough')

# comparing different classification models using sklearn pipeline
models = [LogisticRegression(), SVC(), AdaBoostClassifier(), RandomForestClassifier(), XGBClassifier(),          DecisionTreeClassifier(), KNeighborsClassifier(), CatBoostClassifier()]

model_labels = ['LogisticReg.','SVC','AdaBoost','RandomForest','Xgboost','DecisionTree','KNN', 'CatBoost']
mean_validation_f1_scores = []

for model in models:

  data_pipeline = Pipeline(steps = [
                                    ('scaler', scaler),
                                    ('resample', SMOTETomek()),
                                    ('model', model)
  ])
  mean_validation_f1 = cross_val_score(data_pipeline, df1, salary, cv=KFold(n_splits=10), scoring='f1',n_jobs=-1).mean()
  mean_validation_f1_scores.append(mean_validation_f1)


# Setting n_splits in KFold cross validation as 10 in order to use 90 percent of data for training and 10 percent of data for testing in each validation.

print(mean_validation_f1_scores)

# Performing hyperparameter optimization using GridSearchCV on the best performing model that is CatBoostClassifier