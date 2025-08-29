# Databricks notebook source
# MAGIC %md
# MAGIC ## Adding UCI Lib

# COMMAND ----------

!pip install ucimlrepo -q
dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
from ucimlrepo import fetch_ucirepo

# COMMAND ----------

def initial_load(repo_id:int):
    repo_data = fetch_ucirepo(id=repo_id)  
    X = repo_data.data.features 
    y = repo_data.data.targets
    merded_pd = pd.concat([X, y], axis=1)
    #merged = X.join(y, on=X.index == y.index, how='inner')
    return merded_pd

def create_table(df, table_name:str, schema_name:str):
    data = spark.createDataFrame(df)
    data.write.mode("overwrite").saveAsTable(f'{schema_name}.{table_name}')


# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating schemas

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE SCHEMA IF NOT EXISTS bronze;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS silver;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS gold;

# COMMAND ----------

repo_id = 2
data = initial_load(repo_id)
data.head()

# COMMAND ----------

schema = 'bronze'
table_name = 'sensus'
create_table(schema_name=schema, table_name=table_name, df=data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## First Queries

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM bronze.sensus

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.income, COUNT(*) FROM bronze.sensus a GROUP BY a.income

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fixing Income Category

# COMMAND ----------

# MAGIC %md
# MAGIC This is normaly not performed in bronze layer, however, to simplify the explanation of the data pipeline I am running in bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC UPDATE bronze.sensus
# MAGIC SET income = TRIM(REGEXP_REPLACE(income, '\\.$', ''))
# MAGIC WHERE income RLIKE '\\.$|^\\s+|\\s+$';

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.income, COUNT(*) FROM bronze.sensus a GROUP BY a.income

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating biased dataset

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TEMPORARY VIEW biased_sensus AS
# MAGIC SELECT * FROM bronze.sensus b WHERE NOT
# MAGIC (b.sex = 'Female' AND b.income = '>50K')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE bronze.biased_sensus as
# MAGIC SELECT * FROM biased_sensus

# COMMAND ----------


