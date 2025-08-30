# Databricks notebook source
!pip install scikit-learn --q
!pip install pandas --q
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

# COMMAND ----------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Unbiased Silver Table

# COMMAND ----------

spark_df = spark.table('bronze.sensus')
pandas_df = spark_df.toPandas()
pandas_df.head()

# COMMAND ----------

occupation_list = pandas_df['occupation'].unique().tolist()
workclass_list = pandas_df['workclass'].unique().tolist()
native_country_list = pandas_df['native-country'].unique().tolist()
sex_list = pandas_df['sex'].unique().tolist()
marital_status_list = pandas_df['marital-status'].unique().tolist()
race_list = pandas_df['race'].unique().tolist()
education_list = pandas_df['education'].unique().tolist()
relationship_list = pandas_df['relationship'].unique().tolist()

# COMMAND ----------

print(f'Occupation: {occupation_list}\n')
print(f'Workclass: {workclass_list}\n')
print(f'Native Country: {native_country_list}\n')
print(f'Sex: {sex_list}\n')
print(f'Marital Status: {marital_status_list}\n')
print(f'Race: {race_list}\n')
print(f'Education: {education_list}\n')
print(f'Relationship: {relationship_list}\n') 

# COMMAND ----------

def set_target(df:pd.DataFrame, target:str='income', positive:str='>50K') -> pd.DataFrame:
    df[target] = (df[target] == positive).astype(int)
    return df

def drop_columns(df:pd.DataFrame, columns:list) -> pd.DataFrame:
    return df.drop(columns, axis=1)

def set_unemployed(df: pd.DataFrame) -> pd.DataFrame:
    mask = df['workclass'].eq('Never-worked')
    df.loc[mask, 'occupation'] = 'Unemployed'
    return df

def set_not_informed(df: pd.DataFrame) -> pd.DataFrame:
    mask = ((df['workclass'].eq('?') & df['occupation'].eq('?')) |\
         (df['workclass'].isna() & df['occupation'].isna()) )
    df.loc[mask, ['workclass', 'occupation']] = 'Not-Informed'
    return df

def set_country_not_informed(df: pd.DataFrame)->pd.DataFrame:
    mask = (df['native-country'].isna() | df['native-country'].eq('?'))
    df.loc[mask, 'native-country'] = 'Not-Informed'
    return df

def set_south_korea(df:pd.DataFrame)->pd.DataFrame:
    mask = ((df['native-country'].eq('South'))&(df['race'].eq('Asian-Pac-Islander')))
    df.loc[mask, 'native-country'] = 'South Korea'
    return df

def save_to_silver(df: pd.DataFrame, table_name:str)->pd.DataFrame:
    data = spark.createDataFrame(df)
    data.write.mode("overwrite").saveAsTable(f'silver.{table_name}')

# COMMAND ----------

#column_to_drop = ['fnlwgt', 'capital-gain', 'capital-loss']
#test = set_unemployed(pandas_df)
#test = drop_columns(test, column_to_drop)
#test = set_not_informed(test)
#test = set_country_not_informed(test)
#test = set_south_korea(test)
#test.head()

# COMMAND ----------

column_to_drop = ['fnlwgt', 'capital-gain', 'capital-loss','education', 'capital-gain']
silver_pipeline = Pipeline(
    steps=[
        ('set_target', FunctionTransformer(set_target)),
        ('drop_columns', FunctionTransformer(drop_columns, kw_args={'columns':column_to_drop})),
        ('set_unemployed', FunctionTransformer(set_unemployed)),
        ('set_not_informed', FunctionTransformer(set_not_informed)),
        ('set_country_not_informed', FunctionTransformer(set_country_not_informed)),
        ('set_south_korea', FunctionTransformer(set_south_korea))
    ]
)

# COMMAND ----------

data = silver_pipeline.fit_transform(pandas_df)
data.head()

# COMMAND ----------

table = 'sensus'
save_to_silver(data, table)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver.sensus;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating Biased Silver Table

# COMMAND ----------

spark_df_biased = spark.table('bronze.biased_sensus')
pandas_df_biased = spark_df.toPandas()
pandas_df_biased.head()

# COMMAND ----------

biased_data = silver_pipeline.fit_transform(pandas_df_biased)
biased_data.head()

# COMMAND ----------

table = 'biased_sensus'
save_to_silver(biased_data, table)
