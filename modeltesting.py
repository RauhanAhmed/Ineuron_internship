# Robust Model Testing
from HyperparameterTuning import data_pipeline, df1, salary
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

x_train, x_test, y_train, y_test = train_test_split(df1, salary, test_size = 0.1, random_state = 0)
data_pipeline.fit(x_train, y_train)

y_pred = data_pipeline.predict(x_test)

print(pd.DataFrame(confusion_matrix(y_pred=y_pred, y_true=y_test)))

print('recall :', recall_score(y_test, y_pred))
print('precision :', precision_score(y_test, y_pred))
print('F1 :', f1_score(y_test, y_pred))
print('accuracy :', accuracy_score(y_test, y_pred))

# So, as we can see we are getting some good with our CatBoost model after hyperparameter tuning using GridSearchCV. Now, using this model in production

pickle.dumps(data_pipeline, open("pipeline.pkl", "wb"))