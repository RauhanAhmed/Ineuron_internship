from imblearn.pipeline import Pipeline
from ModelSelection import df1, salary


# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

params = {
    'model__depth':[5,7,10,11,12,13],
    'model__learning_rate':[0.05,0.1,0.15,0.25,0.3],
    'model__iterations':[25, 30, 50, 80, 100,120]
}

data_pipeline = Pipeline(steps = [
                                  ('scaler', scaler),
                                  ('model', CatBoostClassifier(class_weights={0:1, 1:3.14}))
])

grid_search=GridSearchCV(data_pipeline,param_grid=params,scoring='f1',n_jobs=-1,cv=KFold(n_splits=3))
grid_search.fit(df1, salary)

print(grid_search.best_score_)

print(grid_search.best_params_)

# Creating pipeline with new parameters after optained from gridsearchcv
data_pipeline = Pipeline(steps = [
                                  ('scaler', scaler),
                                  ('resample', SMOTETomek()),
                                  ('model', CatBoostClassifier(depth=10,\
                                                               iterations=30, learning_rate=0.25))
])

f1_scores = cross_val_score(data_pipeline, df1, salary, scoring='f1', cv=KFold(n_splits=10), n_jobs=-1)
print('max f1 :', f1_scores.max())
print('min f1 :', f1_scores.min())
print('mean f1 :', f1_scores.mean())

accuracy_scores = cross_val_score(data_pipeline, df1, salary, scoring='accuracy', cv=KFold(n_splits=10), n_jobs=-1)
print('max accuracy :', accuracy_scores.max())
print('min accuracy :', accuracy_scores.min())
print('mean accuracy :', accuracy_scores.mean())

# Here, as we can see the maximum, minimum and mean F1 and accuracy scores are pretty much good so we'll continue with the model