# Databricks notebook source
!pip install scikit-learn --quiet
!pip install pandas --quiet
!pip install optuna --quiet
dbutils.library.restartPython()

# COMMAND ----------

import optuna
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# COMMAND ----------

path = '/Local'
biased_exp = mlflow.set_experiment(f'{path}/biased_experiment')
unbiased_exp=mlflow.set_experiment(f'{path}/unbiased_experiment')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Datasets

# COMMAND ----------

unbiased_spark_df = spark.table('gold.unbiased_df')
unbiased_df = unbiased_spark_df.toPandas()
unbiased_df.head()

# COMMAND ----------

biased_spark_df = spark.table('gold.biased_df')
biased_df = biased_spark_df.toPandas()
biased_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Objective Function

# COMMAND ----------

def objective(trial, parent_run_id, X, y):
  with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
    mlflow.set_tag("mlflow.parentRunId", parent_run_id)
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    random_state=42
    n_jobs = -1

    model = Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        max_features=max_features, 
        bootstrap=bootstrap, 
        criterion=criterion, 
        class_weight='balanced', 
        n_jobs=n_jobs, 
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf))
    ])

    cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    
    mlflow.log_params({
      'n_estimators': n_estimators,
      'max_depth': max_depth,
      'max_features': max_features,
      'bootstrap': bootstrap,
      'criterion': criterion,
      'min_samples_split': min_samples_split,
      'min_samples_leaf': min_samples_leaf
    })

    mlflow.log_metrics({'cv_accuracy': cv_score})
    return cv_score
  
def trigger_ml_flow(X, y, n_trials, run_name = 'Default'):
  with mlflow.start_run(run_name=run_name) as run:
    parent_run_id = run.info.run_id
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, parent_run_id, X, y), n_trials=n_trials)
    best_params = study.best_params
    best_trial = study.best_trial
    best_score = study.best_value
    final_model = Pipeline([
      ('scaler', StandardScaler()),
      ('classifier', RandomForestClassifier(
          **best_params,
          class_weight='balanced',
          random_state=42,
          n_jobs=-1
          ))
      ])
    print(f'Best params: {best_params}')
    print(f'Best score: {best_score}')
    print(f'Best trial: {best_trial}')
    mlflow.sklearn.log_model(final_model, 'final_model')

# COMMAND ----------

# biased_X, biased_y = biased_df.drop('income', axis=1), biased_df['income']

# mlflow.set_experiment(experiment_id=biased_exp.experiment_id)

# with mlflow.start_run(run_name='Biased') as run:
#   parent_run_id = run.info.run_id
#   study = optuna.create_study(direction='maximize')
#   study.optimize(lambda trial: objective(trial, parent_run_id, biased_X, biased_y), n_trials=3)
#   best_params = study.best_params
#   best_trial = study.best_trial
#   best_score = study.best_value
#   final_model = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', RandomForestClassifier(
#         **best_params,
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1
#         ))
#     ])
#   print(f'Best params: {best_params}')
#   print(f'Best score: {best_score}')
#   print(f'Best trial: {best_trial}')
#   mlflow.sklearn.log_model(final_model, 'final_model')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trainin Biased Model

# COMMAND ----------

biased_X, biased_y = biased_df.drop('income', axis=1), biased_df['income']

mlflow.set_experiment(experiment_id=biased_exp.experiment_id)
trigger_ml_flow(
  X=biased_X,
  y=biased_y, 
  n_trials=20, 
  run_name='Biased_Func')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Unbiased Model

# COMMAND ----------

unbiased_X, unbiased_y = unbiased_df.drop('income', axis=1), unbiased_df['income']
X_train, X_test, y_train, y_test = train_test_split(unbiased_X, unbiased_y, test_size=0.2, random_state=42, stratify=unbiased_y)

# COMMAND ----------

mlflow.set_experiment(experiment_id=unbiased_exp.experiment_id)
trigger_ml_flow(
  X=X_train,
  y=y_train, 
  n_trials=20, 
  run_name='Unbiased_Func')

# COMMAND ----------


