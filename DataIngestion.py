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
# MAGIC CREATE SCHEMA IF NOT EXISTS uci_raw;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS uci_unbalanced;
# MAGIC
# MAGIC CREATE SCHEMA IF NOT EXISTS uci_balanced;

# COMMAND ----------

repo_id = 2
schema = 'uci_raw'
data = initial_load(repo_id)
data.head()

# COMMAND ----------

schema = 'uci_raw'
table_name = 'sensus'
create_table(schema_name=schema, table_name=table_name, df=data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## First Queries

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM uci_raw.sensus

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT a.income, COUNT(*) FROM uci_raw.sensus a GROUP BY a.income

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select * from uci_raw.sensus s where s.income = '>50K' 
